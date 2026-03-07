"""Debug API routes for sessions/messages/dumps inspection."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import secrets
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ..config import get_config
from .repository import DebugRepository

logger = logging.getLogger("oci-gateway")

debug_router = APIRouter()

# Global subscribers for SSE broadcasting: session_id -> list of queues
_session_subscribers: Dict[str, List[asyncio.Queue]] = {}
# Global subscriber for ALL sessions (key: None or "*")
_global_subscribers: List[asyncio.Queue] = []


def _ensure_enabled() -> DebugRepository:
    conf = get_config()
    if not conf.debug_ui_enabled:
        raise HTTPException(status_code=404, detail="Debug UI is disabled")
    return DebugRepository(conf.debug_ui_index_db)


def _enforce_auth(request: Request) -> None:
    conf = get_config()
    mode = conf.debug_ui_auth_mode
    if mode == "none":
        return
    auth = (request.headers.get("authorization") or "").strip()

    if mode == "bearer":
        expected = str(getattr(conf, "debug_ui_auth_token", "") or "").strip()
        if not expected:
            raise HTTPException(status_code=500, detail="Debug UI auth token is not configured")

        provided = ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        if not provided:
            provided = str(request.query_params.get("access_token", "")).strip()

        if not provided:
            raise HTTPException(status_code=401, detail="Missing bearer token")
        if not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=403, detail="Invalid bearer token")
        return

    if mode == "basic":
        expected_user = str(getattr(conf, "debug_ui_auth_basic_user", "") or "").strip()
        expected_password = str(getattr(conf, "debug_ui_auth_basic_password", "") or "")
        if not expected_user or not expected_password:
            raise HTTPException(status_code=500, detail="Debug UI basic auth credentials are not configured")

        if not auth.lower().startswith("basic "):
            raise HTTPException(status_code=401, detail="Missing basic auth header")

        encoded = auth[6:].strip()
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise HTTPException(status_code=401, detail="Invalid basic auth header")

        if not secrets.compare_digest(username, expected_user) or not secrets.compare_digest(password, expected_password):
            raise HTTPException(status_code=403, detail="Invalid basic auth credentials")
        return

    raise HTTPException(status_code=500, detail=f"Unsupported debug_ui auth_mode: {mode}")


@debug_router.get("/health")
def debug_health(request: Request):
    _enforce_auth(request)
    conf = get_config()
    return {
        "enabled": bool(conf.debug_ui_enabled),
        "dump_dir": conf.debug_ui_dump_dir,
        "index_db": conf.debug_ui_index_db,
        "scan_interval_sec": conf.debug_ui_scan_interval_sec,
    }


@debug_router.get("/sessions")
def list_sessions(
    request: Request,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=200),
    q: str = Query(default=""),
):
    _enforce_auth(request)
    repo = _ensure_enabled()
    return repo.list_sessions(page=page, size=size, q=q)


@debug_router.get("/sessions/{session_id}/messages")
def list_session_messages(
    session_id: str,
    request: Request,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=500),
):
    _enforce_auth(request)
    repo = _ensure_enabled()
    return repo.list_session_messages(session_id=session_id, page=page, size=size)


@debug_router.get("/sessions/{session_id}/timeline")
def get_session_timeline(
    session_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=1000),
):
    """Get timeline events for swimlane visualization."""
    _enforce_auth(request)
    repo = _ensure_enabled()
    return repo.get_session_timeline(session_id=session_id, limit=limit)


@debug_router.get("/timeline")
def get_all_timeline(
    request: Request,
    limit: int = Query(default=500, ge=1, le=2000),
):
    """Get timeline events for ALL sessions (global swimlane view)."""
    _enforce_auth(request)
    repo = _ensure_enabled()
    return repo.get_all_timeline(limit=limit)


@debug_router.get("/messages/{message_id}")
def get_message_detail(message_id: str, request: Request):
    _enforce_auth(request)
    repo = _ensure_enabled()
    detail = repo.get_message_detail(message_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Message not found")
    return detail


@debug_router.get("/dumps/{dump_id}/raw")
def get_dump_raw(dump_id: int, request: Request):
    _enforce_auth(request)
    repo = _ensure_enabled()
    dump = repo.get_dump_raw(dump_id)
    if dump is None:
        raise HTTPException(status_code=404, detail="Dump not found")
    return dump


@debug_router.get("/sessions/{session_id}/events")
async def session_events(session_id: str, request: Request):
    """SSE stream for real-time session timeline updates."""
    _enforce_auth(request)
    _ensure_enabled()

    queue: asyncio.Queue = asyncio.Queue()

    # Register subscriber
    if session_id not in _session_subscribers:
        _session_subscribers[session_id] = []
    _session_subscribers[session_id].append(queue)
    logger.debug("SSE subscriber added for session=%s, total=%d", session_id, len(_session_subscribers[session_id]))

    async def event_generator():
        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'session_id': session_id})}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    # Wait for new event, timeout 30 seconds for heartbeat
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: timeline_event\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": heartbeat\n\n"

        finally:
            # Cleanup subscriber
            if session_id in _session_subscribers:
                try:
                    _session_subscribers[session_id].remove(queue)
                    logger.debug("SSE subscriber removed for session=%s, remaining=%d", session_id, len(_session_subscribers[session_id]))
                    if not _session_subscribers[session_id]:
                        del _session_subscribers[session_id]
                except ValueError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@debug_router.get("/events")
async def all_events(request: Request):
    """SSE stream for real-time timeline updates from ALL sessions."""
    _enforce_auth(request)
    _ensure_enabled()

    queue: asyncio.Queue = asyncio.Queue()

    # Register global subscriber
    _global_subscribers.append(queue)
    logger.debug("Global SSE subscriber added, total=%d", len(_global_subscribers))

    async def event_generator():
        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'scope': 'all'})}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    # Wait for new event, timeout 30 seconds for heartbeat
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: timeline_event\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": heartbeat\n\n"

        finally:
            # Cleanup subscriber
            try:
                _global_subscribers.remove(queue)
                logger.debug("Global SSE subscriber removed, remaining=%d", len(_global_subscribers))
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def broadcast_session_event(session_id: str, event: dict) -> None:
    """Broadcast a timeline event to all SSE subscribers.

    Called by DebugDumpIndexer when a new dump is ingested.
    Pushes to both session-specific and global subscribers.
    """
    count = 0

    # Push to session-specific subscribers
    if session_id in _session_subscribers:
        for queue in _session_subscribers[session_id]:
            try:
                queue.put_nowait(event)
                count += 1
            except asyncio.QueueFull:
                logger.warning("SSE queue full for session=%s, dropping event", session_id)

    # Push to global subscribers (all sessions)
    for queue in _global_subscribers:
        try:
            queue.put_nowait(event)
            count += 1
        except asyncio.QueueFull:
            logger.warning("Global SSE queue full, dropping event")

        if count > 0:
            logger.debug("Broadcast event to %d subscribers for session=%s", count, session_id)


@debug_router.delete("/clear")
def clear_all_data(request: Request):
    """Clear all debug data from database and delete dump files.

    This is a destructive operation that cannot be undone.
    """
    _enforce_auth(request)
    repo = _ensure_enabled()
    result = repo.clear_all_data()
    logger.info("Cleared all debug data: %s", result)
    return {"success": True, "result": result}
