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
from typing import Optional, Union, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
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
from src.utils.constants import DEFAULT_MAX_TOKENS


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
    init_config(config_file)

    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
