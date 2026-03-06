"""Main entry point for OCI Anthropic Gateway.

This refactored version splits the monolithic gateway.py into logical modules:
- config: Configuration management
- utils: Helper functions (token counting, tools, cache, JSON, content conversion)
- services: Business logic (OCI generation)
- routes: API route handlers
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Load local environment variables from .env if present
load_dotenv()

# Initialize logging with custom configuration
from src.utils.logging_config import get_logger
logger = get_logger()

# Create FastAPI app
app = FastAPI(title="OCI GenAI Anthropic Gateway")

# Import from refactored modules
from src.config import init_config, get_config
from src.routes import handle_count_tokens, handle_messages_request
from src.debug import debug_router, DebugDumpIndexer
from src.debug.routes import broadcast_session_event
from src.utils.rate_limiter import SlidingWindowRateLimiter

app.include_router(debug_router, prefix="/debug/api")
_debug_ui_dir = Path("web/debug")
if _debug_ui_dir.exists():
    app.mount("/debug", StaticFiles(directory=str(_debug_ui_dir), html=True), name="debug-ui")


def _build_rate_limit_key(request: Request, body: dict) -> str:
    client_ip = request.client.host if request.client else "unknown"
    session_id = ""
    if isinstance(body, dict):
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            session_id = str(metadata.get("session_id") or "").strip()
    if session_id:
        return f"{client_ip}:{session_id}"
    return client_ip


def _check_rate_limit(request: Request, body: dict) -> Optional[JSONResponse]:
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        return None

    key = _build_rate_limit_key(request, body)
    allowed, retry_after = limiter.allow(key)
    if allowed:
        return None

    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        content={
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Too many requests. Please retry later.",
            },
        },
    )


@app.on_event("startup")
async def _startup() -> None:
    try:
        cfg = get_config()
    except RuntimeError:
        cfg = init_config("config.json")

    if cfg.debug_ui_enabled:
        indexer = DebugDumpIndexer(
            dump_dir=cfg.debug_ui_dump_dir,
            db_path=cfg.debug_ui_index_db,
            scan_interval_sec=cfg.debug_ui_scan_interval_sec,
        )
        # Register callback for real-time SSE broadcasting
        indexer.set_event_callback(broadcast_session_event)
        app.state.debug_indexer = indexer
        app.state.debug_indexer_task = asyncio.create_task(indexer.run_forever())
    else:
        app.state.debug_indexer = None
        app.state.debug_indexer_task = None

    if cfg.rate_limit_enabled:
        app.state.rate_limiter = SlidingWindowRateLimiter(
            limit=cfg.rate_limit_requests,
            window_sec=cfg.rate_limit_window_sec,
        )
    else:
        app.state.rate_limiter = None


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "debug_indexer_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@app.post("/{path:path}")
async def catch_all(path: str, request: Request):
    """
    Catch-all route to handle all incoming paths.
    Handles telemetry, token count, and messages requests.
    """
    # Handle event_logging (telemetry)
    if "event_logging" in path:
        return {"status": "ok"}

    # Handle count_tokens endpoint
    if "count_tokens" in path:
        body = await request.json()
        limited = _check_rate_limit(request, body)
        if limited:
            return limited
        return await handle_count_tokens(body)

    # Handle messages request
    if "messages" in path:
        body = await request.json()
        limited = _check_rate_limit(request, body)
        if limited:
            return limited
        config = get_config()
        return await handle_messages_request(body, body.get("model", "").lower(), config)

    return {"detail": "Not Found"}


if __name__ == "__main__":
    # Initialize configuration
    config_file = "config.json"
    config = init_config(config_file)

    # Run the server
    uvicorn.run(
        app,
        host=config.server_host,
        port=config.server_port,
        log_level=config.server_log_level,
    )
