"""API routes for OCI Anthropic Gateway."""

import logging
import oci
import uuid
from typing import Union
from fastapi.responses import StreamingResponse, JSONResponse

from ..utils.token import count_tokens_from_messages
from ..utils.cache import extract_cache_control, log_cache_info
from ..utils.content_converter import (
    convert_content_to_oci,
    convert_anthropic_messages_to_cohere,
)
from ..utils.request_validation import (
    validate_count_tokens_payload,
    validate_messages_payload,
    validate_system_payload,
    collect_requested_modalities,
    validate_model_modalities,
)
from ..utils.guardrails import (
    apply_input_guardrails,
    collect_input_text_for_guardrails,
    summarize_guardrails_result,
)

logger = logging.getLogger("oci-gateway")

# Import the generation functions from services
from ..services.generation import generate_oci_non_stream, generate_oci_stream


def _guardrails_error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": message,
            },
        },
    )


async def handle_count_tokens(body: dict) -> JSONResponse:
    """Handle count_tokens endpoint.

    Args:
        body: Request body with messages and system prompt

    Returns:
        JSONResponse with token count
    """
    try:
        err = validate_count_tokens_payload(body)
        if err:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": err,
                    },
                },
            )

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
    request_metadata = body.get("metadata")
    max_messages = int(getattr(app_config, "messages_max_items", 200) or 200)
    validation_err = validate_messages_payload(body, max_messages=max_messages)
    if validation_err:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": validation_err,
                },
            },
        )
    system_validation_err = validate_system_payload(body.get("system"))
    if system_validation_err:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": system_validation_err,
                },
            },
        )

    if not isinstance(request_metadata, dict):
        request_metadata = {}
    session_id = request_metadata.get("session_id") or f"sess_{uuid.uuid4().hex}"
    request_id = request_metadata.get("request_id") or f"req_{uuid.uuid4().hex}"
    request_metadata["session_id"] = session_id
    request_metadata["request_id"] = request_id
    body["metadata"] = request_metadata

    trace_ctx = {
        "session_id": session_id,
        "request_id": request_id,
        "user_id": request_metadata.get("user_id"),
        "tenant_id": request_metadata.get("tenant_id"),
        "parent_message_id": request_metadata.get("parent_message_id"),
    }

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
    model_types = model_conf.get("model_types", ["text"])
    guardrails_config = getattr(app_config, "guardrails", None)

    requested_modalities = collect_requested_modalities(body.get("messages", []), body.get("system"))
    modalities_err = validate_model_modalities(requested_modalities, model_types, is_cohere=is_cohere)
    if modalities_err:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": modalities_err,
                },
            },
        )

    if guardrails_config and guardrails_config.enabled and guardrails_config.input.enabled:
        input_text = collect_input_text_for_guardrails(body, guardrails_config)
        if input_text:
            try:
                input_result = await apply_input_guardrails(
                    client=app_config.genai_client,
                    compartment_id=model_conf.get("compartment_id", ""),
                    content=input_text,
                    config=guardrails_config,
                )
                if input_result.issue_detected:
                    summary = summarize_guardrails_result(
                        input_result,
                        redact_logs=guardrails_config.redact_logs,
                    )
                    if guardrails_config.log_details:
                        logger.warning("Input guardrails issue detected: %s", summary)
                    else:
                        logger.warning(
                            "Input guardrails issue detected: reason=%s",
                            input_result.blocked_reason or "unspecified",
                        )
                    if guardrails_config.mode == "block":
                        return _guardrails_error_response(
                            guardrails_config.block_message,
                            status_code=guardrails_config.block_http_status,
                        )
            except Exception:
                logger.exception("Input guardrails execution failed")
                if guardrails_config.input.fail_mode == "closed":
                    return _guardrails_error_response(
                        guardrails_config.block_message,
                        status_code=guardrails_config.block_http_status,
                    )

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
                if isinstance(block, dict) and block.get("type") == "text":
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
        try:
            converted_content = convert_content_to_oci(m["content"])
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": str(e),
                    },
                },
            )

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
        current_message, chat_history, system_message = convert_anthropic_messages_to_cohere(messages, system_prompt)
        cohere_messages = (current_message, chat_history, system_message)
        logger.info(f"Converted to Cohere format: current_message_len={len(current_message)}, chat_history_len={len(chat_history)}")

    # Add tools to parameters if present
    params_with_tools = _prepare_tools_params(body, is_cohere, oci_msgs, app_config)
    effective_stream = bool(body.get("stream", False))

    if (
        effective_stream
        and guardrails_config
        and guardrails_config.enabled
        and guardrails_config.output.enabled
    ):
        if guardrails_config.streaming_behavior == "reject":
            return _guardrails_error_response(
                "Streaming responses are not supported while output guardrails are enabled.",
                status_code=400,
            )
        logger.warning("Downgrading stream=true request to non-stream because output guardrails are enabled")
        effective_stream = False
        body["stream"] = False
        params_with_tools["stream"] = False
        if isinstance(params_with_tools.get("metadata"), dict):
            params_with_tools["metadata"]["guardrails_stream_downgraded"] = True

    # Generate response
    logger.debug(f"About to call generation with genai_client={app_config.genai_client is not None}")
    if effective_stream:
        return StreamingResponse(
            generate_oci_stream(
                oci_msgs,
                params_with_tools,
                message_id,
                model_conf,
                req_model,
                app_config.genai_client,
                cohere_messages,
                debug_enabled=bool(app_config.debug),
                debug_redact_media=bool(getattr(app_config, "debug_redact_media", True)),
                trace_ctx=trace_ctx,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming logic
        return await generate_oci_non_stream(
            oci_msgs,
            params_with_tools,
            message_id,
            model_conf,
            req_model,
            app_config.genai_client,
            cohere_messages,
            debug_enabled=bool(app_config.debug),
            debug_redact_media=bool(getattr(app_config, "debug_redact_media", True)),
            trace_ctx=trace_ctx,
            guardrails_config=guardrails_config,
        )


def _prepare_tools_params(body: dict, is_cohere: bool, oci_msgs: list, app_config) -> dict:
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
    params["enable_nl_tool_fallback"] = bool(getattr(app_config, "enable_nl_tool_fallback", False))

    return params
