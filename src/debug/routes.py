"""Debug API routes for sessions/messages/dumps inspection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ..config import get_config
from .repository import DebugRepository


debug_router = APIRouter()


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
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing authorization")


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
