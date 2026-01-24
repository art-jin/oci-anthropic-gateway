"""API routes for OCI Anthropic Gateway."""

import asyncio
import json
import logging
import oci
import uuid
from typing import Optional, Union, List
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse

from ..config import get_config
from ..utils.token import count_tokens_from_messages
from ..utils.cache import extract_cache_control, log_cache_info
from ..utils.content_converter import (
    convert_content_to_oci,
    convert_anthropic_messages_to_cohere,
)
from ..utils.tools import anthropic_to_cohere_tools, _build_tool_use_instruction
from ..utils.constants import DEFAULT_MAX_TOKENS

logger = logging.getLogger("oci-gateway")

# Import the generation functions from services
from ..services.generation import generate_oci_non_stream, generate_oci_stream


async def handle_count_tokens(body: dict) -> JSONResponse:
    """Handle count_tokens endpoint.

    Args:
        body: Request body with messages and system prompt

    Returns:
        JSONResponse with token count
    """
    try:
        messages = body.get("messages", [])
        system = body.get("system")

        # Count tokens
        input_tokens = count_tokens_from_messages(messages, system)

        # Return Anthropic-compatible response
        return JSONResponse(content={
            "type": "usage",
            "input_tokens": input_tokens
        })
    except Exception as e:
        logger.exception("Count tokens failed")
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Failed to count tokens: {str(e)}"
                }
            }
        )


async def handle_messages_request(
    body: dict,
    req_model: str,
    app_config
) -> Union[StreamingResponse, JSONResponse]:
    """Handle messages API endpoint.

    Args:
        body: Request body
        req_model: Requested model name
        app_config: Configuration instance

    Returns:
        StreamingResponse or JSONResponse
    """
    tools = body.get("tools") or []
    tool_names = [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
    logger.info("REQ stream=%s tools=%d tool_choice=%s", bool(body.get("stream")), len(tool_names),
                body.get("tool_choice"))
    if tool_names and len(tool_names) > 0:
        logger.debug("REQ tool_names=%s", tool_names[:20])

    # Get model configuration
    model_conf = app_config.get_model_config(req_model)
    api_format = model_conf.get("api_format", "generic").lower()
    is_cohere = api_format == "cohere"

    # Convert messages to OCI format
    oci_msgs = []

    # Handle system prompt - insert as first SYSTEM message
    system_prompt = body.get("system")
    if system_prompt:
        # Support both string and array format for system prompt
        if isinstance(system_prompt, str):
            system_text = system_prompt
            cache_info = {"has_cache_control": False, "cached_blocks": 0, "cache_types": []}
        elif isinstance(system_prompt, list):
            # Handle array format: [{"type": "text", "text": "..."}, ...]
            # Check for cache_control in system prompt
            cache_info = extract_cache_control(system_prompt)
            if cache_info["has_cache_control"]:
                log_cache_info("system", cache_info)

            system_text = ""
            for block in system_prompt:
                if block.get("type") == "text":
                    system_text += block.get("text", "")
        else:
            system_text = str(system_prompt)
            cache_info = {"has_cache_control": False, "cached_blocks": 0, "cache_types": []}

        if system_text:
            oci_msgs.append(oci.generative_ai_inference.models.Message(
                role="SYSTEM",
                content=[oci.generative_ai_inference.models.TextContent(text=system_text)]
            ))
            cache_suffix = " [CACHED]" if cache_info["has_cache_control"] else ""
            logger.debug(f"Added system prompt ({len(system_text)} chars){cache_suffix}")

    # Convert regular messages
    messages = body.get("messages", [])
    for idx, m in enumerate(messages):
        # Check for cache_control in message content
        content = m.get("content", [])
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"):
                    logger.info("REQ msg[%d] has %s: %s", idx, b.get("type"), 
                               str(b)[:200] if b.get("type") == "tool_result" else b.get("name", "unknown"))
        cache_info = extract_cache_control(content) if isinstance(content, list) else {"has_cache_control": False, "cached_blocks": 0, "cache_types": []}

        # Use the convert_content_to_oci function to handle text, images, tool_use, and tool_result
        converted_content = convert_content_to_oci(m["content"])

        # Check if the result is a list (multimodal with images) or string (text only)
        if isinstance(converted_content, list):
            # Multimodal content with images
            msg = oci.generative_ai_inference.models.Message(
                role=m["role"].upper(),
                content=converted_content
            )
            image_count = len([c for c in converted_content if isinstance(c, oci.generative_ai_inference.models.ImageContent)])
            cache_suffix = " [CACHED]" if cache_info["has_cache_control"] else ""
            logger.debug(f"Added multimodal message {idx} with {image_count} image(s){cache_suffix}")
        else:
            # Text only content
            msg = oci.generative_ai_inference.models.Message(
                role=m["role"].upper(),
                content=[oci.generative_ai_inference.models.TextContent(text=converted_content)]
            )
            cache_suffix = " [CACHED]" if cache_info["has_cache_control"] else ""
            if cache_info["has_cache_control"]:
                log_cache_info(f"message[{idx}]", cache_info)
            logger.debug(f"Added message {idx} ({len(converted_content)} chars){cache_suffix}")

        oci_msgs.append(msg)

    # Handle metadata - extract for logging and response
    metadata = body.get("metadata")
    if metadata:
        logger.info(f"Request metadata: {metadata}")

    message_id = f"msg_oci_{uuid.uuid4().hex}"

    # Prepare Cohere messages if needed
    cohere_messages = None
    if is_cohere:
        # Convert messages to Cohere format
        system_prompt = body.get("system")
        messages = body.get("messages", [])
        current_message, chat_history, system_message = convert_anthropic_messages_to_cohere(messages, system_prompt)
        cohere_messages = (current_message, chat_history, system_message)
        logger.info(f"Converted to Cohere format: current_message_len={len(current_message)}, chat_history_len={len(chat_history)}")

    # Add tools to parameters if present
    params_with_tools = _prepare_tools_params(body, is_cohere, oci_msgs)

    # Generate response
    logger.debug(f"About to call generation with genai_client={app_config.genai_client is not None}")
    if body.get("stream", False):
        return StreamingResponse(
            generate_oci_stream(oci_msgs, params_with_tools, message_id, model_conf, req_model, app_config.genai_client, cohere_messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming logic
        return await generate_oci_non_stream(oci_msgs, params_with_tools, message_id, model_conf, req_model, app_config.genai_client, cohere_messages)


def _prepare_tools_params(body: dict, is_cohere: bool, oci_msgs: list) -> dict:
    """Prepare parameters with tool instructions if needed.

    Args:
        body: Request body
        is_cohere: Whether using Cohere format
        oci_msgs: OCI messages list (may be modified in place)

    Returns:
        Updated params dict
    """
    params = dict(body)  # Make a copy

    if "tools" in params:
        # Tool instruction injection is handled in generation.py
        pass

    return params
