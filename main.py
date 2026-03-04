"""Main entry point for OCI Anthropic Gateway.

This refactored version splits the monolithic gateway.py into logical modules:
- config: Configuration management
- utils: Helper functions (token counting, tools, cache, JSON, content conversion)
- services: Business logic (OCI generation)
- routes: API route handlers
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import oci
import uvicorn

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize logging with custom configuration
from src.utils.logging_config import get_logger, set_log_level
logger = get_logger()

# Create FastAPI app
app = FastAPI(title="OCI GenAI Anthropic Gateway")

# Import from refactored modules
from src.config import init_config, get_config
from src.routes import handle_count_tokens, handle_messages_request
from src.debug import debug_router, DebugDumpIndexer
from src.utils.constants import DEFAULT_MAX_TOKENS

app.include_router(debug_router, prefix="/debug/api")
_debug_ui_dir = Path("web/debug")
if _debug_ui_dir.exists():
    app.mount("/debug", StaticFiles(directory=str(_debug_ui_dir), html=True), name="debug-ui")


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
        app.state.debug_indexer = indexer
        app.state.debug_indexer_task = asyncio.create_task(indexer.run_forever())
    else:
        app.state.debug_indexer = None
        app.state.debug_indexer_task = None


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
        return await handle_count_tokens(body)

    # Handle messages request
    if "messages" in path:
        body = await request.json()
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
