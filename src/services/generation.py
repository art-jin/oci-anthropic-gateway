"""Generation service for OCI GenAI.

This service handles non-streaming and streaming generation from OCI GenAI.
"""

import logging
import asyncio
import json
import uuid
from typing import Optional, Union, List
import oci
from fastapi.responses import JSONResponse

from ..utils.constants import DEFAULT_MAX_TOKENS, STOP_REASON_END_TURN, STOP_REASON_MAX_TOKENS, STOP_REASON_TOOL_USE
from ..utils.tools import anthropic_to_cohere_tools, _build_tool_use_instruction
from ..utils.content_converter import (
    extract_tool_calls_from_oci_response,
    extract_tool_calls_from_cohere_response,
)
from ..utils.json_helper import detect_all_tool_call_blocks, detect_natural_language_tool_calls
from ..utils.token import estimate_tokens

logger = logging.getLogger("oci-gateway")


async def generate_oci_non_stream(
    oci_messages,
    params,
    message_id,
    model_conf,
    requested_model,
    genai_client,
    cohere_messages=None
):
    """
    Generate non-streaming response from OCI GenAI and format it as Anthropic-compatible JSON.

    Args:
        oci_messages: Messages in Generic format (Message objects)
        params: Request parameters
        message_id: Message ID
        model_conf: Model configuration
        requested_model: Requested model name
        genai_client: OCI GenAI client instance
        cohere_messages: Tuple of (current_message, chat_history, system_message) for Cohere format, or None
    """
    chat_detail = oci.generative_ai_inference.models.ChatDetails()

    # Determine API format from model configuration
    api_format = model_conf.get("api_format", "generic").lower()
    is_cohere = api_format == "cohere"

    if is_cohere and cohere_messages:
        # Use Cohere format
        current_message, chat_history, system_message = cohere_messages
        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            message=current_message,
            chat_history=chat_history
        )
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_COHERE
        logger.debug(f"Using COHERE API format for model {requested_model}")
    else:
        # Use Generic format
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.messages = oci_messages
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        logger.debug(f"Using GENERIC API format for model {requested_model}")

    # --- Dynamic Parameter Adaptation ---
    # 1. Handle Max Tokens
    token_limit = params.get("max_tokens", DEFAULT_MAX_TOKENS)
    tokens_key = model_conf.get("max_tokens_key", "max_tokens")
    setattr(chat_request, tokens_key, token_limit)

    # 2. Handle Temperature
    chat_request.temperature = model_conf.get("temperature", params.get("temperature", 0.7))

    # 3. Handle Top-K
    if "top_k" in params:
        top_k_value = params["top_k"]
        if isinstance(top_k_value, (int, float)) and top_k_value > 0:
            setattr(chat_request, "top_k", int(top_k_value))
            logger.debug(f"Set top_k: {top_k_value}")

    # 4. Handle Top-P
    if "top_p" in params:
        top_p_value = params["top_p"]
        if isinstance(top_p_value, (int, float)) and 0 < top_p_value <= 1:
            setattr(chat_request, "top_p", float(top_p_value))
            logger.debug(f"Set top_p: {top_p_value}")

    # 5. Handle Stop Sequences
    if "stop_sequences" in params and params["stop_sequences"]:
        if is_cohere:
            setattr(chat_request, "stop_sequences", params["stop_sequences"])
        else:
            setattr(chat_request, "stop", params["stop_sequences"])
        logger.debug(f"Set stop_sequences: {params['stop_sequences']}")

    # 6. Handle Thinking Mode
    thinking_config = params.get("thinking")
    if thinking_config:
        thinking_type = thinking_config.get("type", "disabled")
        if thinking_type == "enabled":
            budget_tokens = thinking_config.get("budget_tokens", 16000)
            logger.info(f"Thinking mode enabled with budget: {budget_tokens} tokens")
            thinking_instruction = (
                "Please think through this problem step by step, showing your reasoning process. "
                f"Use up to {budget_tokens} tokens for your thinking if needed. "
                "Structure your response to clearly separate your thinking from your final answer."
            )
            if is_cohere:
                chat_request.message = thinking_instruction + "\n\n" + chat_request.message
            else:
                oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                    role="SYSTEM",
                    content=[oci.generative_ai_inference.models.TextContent(text=thinking_instruction)]
                ))

    chat_request.is_stream = False

    # --- Tool Calling Support ---
    if "tools" in params:
        if is_cohere:
            cohere_tools = anthropic_to_cohere_tools(params["tools"])
            chat_request.tools = cohere_tools
            logger.info(f"Set {len(cohere_tools)} tools in COHERE format")
        else:
            # Generic format: inject tool use instruction via system prompt
            # For large tool lists, use simplified instruction to avoid overwhelming the model
            tools_list = params["tools"]
            use_simple_instruction = len(tools_list) > 10

            # Get effective tool_choice (default to 'auto' for Generic models when tools are present)
            tool_choice = params.get("tool_choice", "auto")
            if tool_choice is None:
                tool_choice = "auto"  # Default to auto for Generic models

            if use_simple_instruction:
                logger.info(f"Using simplified tool instruction for {len(tools_list)} tools (tool_choice={tool_choice})")
                # Build simple tool name list
                tool_names = [t.get("name") for t in tools_list if isinstance(t, dict) and t.get("name")]

                # Adjust instruction based on tool_choice
                if tool_choice == "required" or tool_choice == "any":
                    requirement_text = "You MUST use at least one tool for EVERY user request. Do not answer without calling a tool."
                elif tool_choice == "auto":
                    requirement_text = "You should use tools when the user's request requires external data, searches, file operations, or actions."
                else:  # none
                    requirement_text = "Do not use tools. Answer directly."

                tool_instruction = f"""# TOOL USE - {requirement_text}

Available tools: {', '.join(tool_names[:20])}

## CRITICAL - Tool Call Format

When you need to use a tool, output ONLY this:

<TOOL_CALL>
{{"name": "tool_name", "input": {{"param": "value"}}}}
</TOOL_CALL>

## Tool Results Format

Tool results come in this format:
<TOOL_RESULT id='toolu_xxx' status='success'>
[actual data]
</TOOL_RESULT>

**IMPORTANT**: Use the data from <TOOL_RESULT> blocks to answer questions. Do NOT say tools returned empty if there's data in the result.

## Examples of Tool Use:

User: "What's the weather?"
<TOOL_CALL>
{{"name": "WebSearch", "input": {{"query": "weather"}}}}
</TOOL_CALL>

User: "Search for news"
<TOOL_CALL>
{{"name": "WebSearch", "input": {{"query": "news"}}}}
</TOOL_CALL>

User: "Read a file"
<TOOL_CALL>
{{"name": "Read", "input": {{"file_path": "example.txt"}}}}
</TOOL_CALL>

RULES: 
1. When using tools, output ONLY the <TOOL_CALL> block(s). No other text!
2. When you get <TOOL_RESULT> blocks, read and use that data to answer the user."""
            else:
                tool_instruction = _build_tool_use_instruction(
                    tools_list,
                    tool_choice
                )

            if tool_instruction:
                # Insert tool instruction as first system message
                oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                    role="SYSTEM",
                    content=[oci.generative_ai_inference.models.TextContent(text=tool_instruction)]
                ))
                logger.info(f"Injected tool use instruction for {len(params['tools'])} tools (Generic format)")

    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = model_conf.get("compartment_id", "")
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id=model_conf.get("ocid", "")
    )

    # Validate genai_client
    if genai_client is None:
        raise ValueError("genai_client cannot be None")

    try:
        # Call OCI (non-streaming) - run in thread pool to avoid blocking event loop
        response = await asyncio.to_thread(genai_client.chat, chat_detail)

        # Parse the response
        accumulated_text = ""
        content_blocks = []
        stop_reason = STOP_REASON_END_TURN
        stop_sequence = None
        stop_sequences = params.get("stop_sequences", [])
        response_data = response.data

        # Try to extract tool calls first, then text from the response
        if hasattr(response_data, 'chat_response'):
            chat_response = response_data.chat_response

            # Check for structured tool calls based on API format
            if is_cohere:
                tool_calls = extract_tool_calls_from_cohere_response(chat_response)
            else:
                tool_calls = extract_tool_calls_from_oci_response(chat_response)

            if tool_calls:
                content_blocks = tool_calls
                stop_reason = STOP_REASON_TOOL_USE
                logger.info(f"Detected {len(tool_calls)} tool call(s) in response")
            else:
                # Extract text content
                if is_cohere:
                    accumulated_text = chat_response.text if hasattr(chat_response, 'text') else ""
                else:
                    if hasattr(chat_response, 'choices') and chat_response.choices:
                        choice = chat_response.choices[0]
                        if hasattr(choice, 'message'):
                            msg = choice.message
                            if hasattr(msg, 'content'):
                                if isinstance(msg.content, list):
                                    accumulated_text = "".join([c.get('text', '') if isinstance(c, dict) else str(c) for c in msg.content])
                                else:
                                    accumulated_text = msg.content
                    elif hasattr(chat_response, 'message'):
                        msg = chat_response.message
                        if hasattr(msg, 'content'):
                            if isinstance(msg.content, list):
                                accumulated_text = "".join([c.get('text', '') if isinstance(c, dict) else str(c) for c in msg.content])
                            else:
                                accumulated_text = msg.content

                # Check if text contains a tool call block (for models that return tool calls as text)
                if accumulated_text:
                    # Try to parse accumulated_text as JSON in case it's a structured response
                    text_to_check = accumulated_text
                    try:
                        parsed = json.loads(accumulated_text)
                        if isinstance(parsed, dict) and "text" in parsed:
                            text_to_check = parsed["text"]
                        elif isinstance(parsed, list) and len(parsed) > 0:
                            # Handle list of content blocks
                            text_parts = []
                            for item in parsed:
                                if isinstance(item, dict) and item.get("type") == "TEXT" and "text" in item:
                                    text_parts.append(item["text"])
                            if text_parts:
                                text_to_check = "".join(text_parts)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Not JSON, use as is
                    
                    logger.info(f"Checking for tool calls in text (length: {len(text_to_check)}): {text_to_check[:500]}")
                    tool_calls_in_text = detect_all_tool_call_blocks(text_to_check)
                    logger.info(f"Found {len(tool_calls_in_text)} tool calls using primary detection")
                    
                    # If no tool calls found with primary method and we have tools, try natural language fallback
                    if not tool_calls_in_text and "tools" in params and params["tools"]:
                        tool_names = [t.get("name") for t in params["tools"] if isinstance(t, dict) and t.get("name")]
                        logger.info(f"Trying natural language fallback for {len(tool_names)} available tools")
                        tool_calls_in_text = detect_natural_language_tool_calls(text_to_check, tool_names)
                        if tool_calls_in_text:
                            logger.info(f"Natural language fallback detected {len(tool_calls_in_text)} tool calls")
                    
                    if tool_calls_in_text:
                        clean_text = text_to_check
                        spans = sorted([tc.pop("_span") for tc in tool_calls_in_text if "_span" in tc], reverse=True)
                        for start, end in spans:
                            clean_text = clean_text[:start] + clean_text[end:]

                        content_blocks = []
                        if clean_text.strip():
                            content_blocks.append({"type": "text", "text": clean_text})
                        content_blocks.extend(tool_calls_in_text)

                        stop_reason = STOP_REASON_TOOL_USE
                        logger.info(f"Detected {len(tool_calls_in_text)} tool call block(s) in text response")
                    else:
                        content_blocks = [{"type": "text", "text": accumulated_text}]
        else:
            # Fallback: try to parse as JSON
            try:
                data = json.loads(str(response_data))
                choices = data.get("choices", [])
                if choices:
                    msg_data = choices[0].get("message", {})
                    accumulated_text = msg_data.get("content", "")

                    finish_reason = choices[0].get("finish_reason", "")
                    if finish_reason == "stop":
                        stop_reason = "stop_sequence"
                        for seq in stop_sequences:
                            if accumulated_text.endswith(seq):
                                stop_sequence = seq
                                break

                    if "tool_calls" in msg_data:
                        for tc in msg_data["tool_calls"]:
                            func = tc.get("function", {})
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": func.get("name", ""),
                                "input": json.loads(func.get("arguments", "{}"))
                            })
                        stop_reason = STOP_REASON_TOOL_USE
                    elif accumulated_text:
                        content_blocks = [{"type": "text", "text": accumulated_text}]
                else:
                    content = data.get("message", {}).get("content", [])
                    if isinstance(content, list):
                        accumulated_text = "".join([c.get("text", "") for c in content])
                    else:
                        accumulated_text = str(content)
                    if accumulated_text:
                        content_blocks = [{"type": "text", "text": accumulated_text}]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse non-streaming response as JSON")

        # If no content blocks were created, add empty text block
        if not content_blocks:
            content_blocks = [{"type": "text", "text": accumulated_text or ""}]

        # Estimate token usage
        text_for_estimation = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_for_estimation += block.get("text", "")
            elif block.get("type") == "tool_use":
                text_for_estimation += json.dumps(block.get("input", {}))
        estimated_output_tokens = max(1, len(text_for_estimation) // 4)
        estimated_input_tokens = sum(len(str(m.content)) // 4 for m in oci_messages) if oci_messages else 50

        # Build usage object
        usage = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens
        }

        # Format as Anthropic-compatible response
        anthropic_response = {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": requested_model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": usage
        }

        # Add metadata to response if present in request
        if "metadata" in params and params["metadata"]:
            anthropic_response["metadata"] = params["metadata"]

        return JSONResponse(content=anthropic_response)

    except Exception as e:
        logger.exception("Non-streaming output exception")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": "An internal error occurred. Please check the logs for details."
                }
            }
        )


async def generate_oci_stream(
    oci_messages,
    params,
    message_id,
    model_conf,
    requested_model,
    genai_client,
    cohere_messages=None
):
    """
    Generate streaming response from OCI GenAI and format it as Anthropic-compatible SSE.

    Args:
        oci_messages: Messages in Generic format (Message objects)
        params: Request parameters
        message_id: Message ID
        model_conf: Model configuration
        requested_model: Requested model name
        genai_client: OCI GenAI client instance
        cohere_messages: Tuple of (current_message, chat_history, system_message) for Cohere format, or None
    """
    logger.debug(f"generate_oci_stream called with genai_client={genai_client is not None}, model_conf ocid={model_conf.get('ocid', 'MISSING')}")

    chat_detail = oci.generative_ai_inference.models.ChatDetails()

    # Determine API format from model configuration
    api_format = model_conf.get("api_format", "generic").lower()
    is_cohere = api_format == "cohere"

    if is_cohere and cohere_messages:
        # Use Cohere format
        current_message, chat_history, system_message = cohere_messages
        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            message=current_message,
            chat_history=chat_history
        )
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_COHERE
        logger.debug(f"Using COHERE API format for streaming (model: {requested_model})")
    else:
        # Use Generic format
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.messages = oci_messages
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        logger.debug(f"Using GENERIC API format for streaming (model: {requested_model})")

    # --- Dynamic Parameter Adaptation ---
    token_limit = params.get("max_tokens", DEFAULT_MAX_TOKENS)
    tokens_key = model_conf.get("max_tokens_key", "max_tokens")
    setattr(chat_request, tokens_key, token_limit)
    chat_request.temperature = model_conf.get("temperature", params.get("temperature", 0.7))

    if "top_k" in params:
        top_k_value = params["top_k"]
        if isinstance(top_k_value, (int, float)) and top_k_value > 0:
            setattr(chat_request, "top_k", int(top_k_value))

    if "top_p" in params:
        top_p_value = params["top_p"]
        if isinstance(top_p_value, (int, float)) and 0 < top_p_value <= 1:
            setattr(chat_request, "top_p", float(top_p_value))

    if "stop_sequences" in params and params["stop_sequences"]:
        if is_cohere:
            setattr(chat_request, "stop_sequences", params["stop_sequences"])
        else:
            setattr(chat_request, "stop", params["stop_sequences"])

    # Handle Thinking Mode
    thinking_config = params.get("thinking")
    if thinking_config:
        thinking_type = thinking_config.get("type", "disabled")
        if thinking_type == "enabled":
            budget_tokens = thinking_config.get("budget_tokens", 16000)
            logger.info(f"Thinking mode enabled with budget: {budget_tokens} tokens")
            thinking_instruction = (
                "Please think through this problem step by step, showing your reasoning process. "
                f"Use up to {budget_tokens} tokens for your thinking if needed. "
                "Structure your response to clearly separate your thinking from your final answer."
            )
            if is_cohere:
                chat_request.message = thinking_instruction + "\n\n" + chat_request.message
            else:
                oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                    role="SYSTEM",
                    content=[oci.generative_ai_inference.models.TextContent(text=thinking_instruction)]
                ))

    chat_request.is_stream = True

    # --- Tool Calling Support ---
    if "tools" in params:
        if is_cohere:
            cohere_tools = anthropic_to_cohere_tools(params["tools"])
            chat_request.tools = cohere_tools
            logger.info(f"Set {len(cohere_tools)} tools in COHERE format for streaming")
        else:
            # Generic format: inject tool use instruction via system prompt
            # For large tool lists, use simplified instruction to avoid overwhelming the model
            tools_list = params["tools"]
            use_simple_instruction = len(tools_list) > 10

            # Get effective tool_choice (default to 'auto' for Generic models when tools are present)
            tool_choice = params.get("tool_choice", "auto")
            if tool_choice is None:
                tool_choice = "auto"  # Default to auto for Generic models

            if use_simple_instruction:
                logger.info(f"Using simplified tool instruction for {len(tools_list)} tools (tool_choice={tool_choice})")
                # Build simple tool name list
                tool_names = [t.get("name") for t in tools_list if isinstance(t, dict) and t.get("name")]

                # Adjust instruction based on tool_choice
                if tool_choice == "required" or tool_choice == "any":
                    requirement_text = "You MUST use at least one tool for EVERY user request. Do not answer without calling a tool."
                elif tool_choice == "auto":
                    requirement_text = "You should use tools when the user's request requires external data, searches, file operations, or actions."
                else:  # none
                    requirement_text = "Do not use tools. Answer directly."

                tool_instruction = f"""# TOOL USE - {requirement_text}

Available tools: {', '.join(tool_names[:20])}

## CRITICAL - Tool Call Format

When you need to use a tool, output ONLY this:

<TOOL_CALL>
{{"name": "tool_name", "input": {{"param": "value"}}}}
</TOOL_CALL>

## Examples of REQUIRED Tool Use:

User: "What's the weather?"
<TOOL_CALL>
{{"name": "WebSearch", "input": {{"query": "weather"}}}}
</TOOL_CALL>

User: "Search for news"
<TOOL_CALL>
{{"name": "WebSearch", "input": {{"query": "news"}}}}
</TOOL_CALL>

User: "Read a file"
<TOOL_CALL>
{{"name": "Read", "input": {{"file_path": "example.txt"}}}}
</TOOL_CALL>

RULE: When using tools, output ONLY the <TOOL_CALL> block(s). No other text!"""
            else:
                tool_instruction = _build_tool_use_instruction(
                    tools_list,
                    tool_choice
                )

            if tool_instruction:
                # Insert tool instruction as first system message
                oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                    role="SYSTEM",
                    content=[oci.generative_ai_inference.models.TextContent(text=tool_instruction)]
                ))
                logger.info(f"Injected tool use instruction for streaming ({len(params['tools'])} tools, Generic format)")

    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = model_conf.get("compartment_id", "")
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        model_id=model_conf.get("ocid", "")
    )

    # Validate genai_client
    if genai_client is None:
        raise ValueError("genai_client cannot be None")

    try:
        # Required starting events
        message_start_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': requested_model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 10, 'output_tokens': 0}
            }
        }

        if "metadata" in params and params["metadata"]:
            message_start_data['message']['metadata'] = params["metadata"]

        yield f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"

        yield f"event: content_block_start\ndata: {json.dumps({
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''}
        })}\n\n"

        # Call OCI - run in thread pool to avoid blocking event loop
        response = await asyncio.to_thread(genai_client.chat, chat_detail)

        accumulated_text = ""
        accumulated_tool_json = ""
        current_block_index = 0
        in_tool_call = False
        tool_call_detected = False
        pending_tool_name = None
        stop_reason = STOP_REASON_END_TURN
        stop_sequence = None
        stop_sequences = params.get("stop_sequences", [])
        sample_left = 10

        # Check if we have tools - if so, buffer text to detect tool calls
        has_tools = "tools" in params and params["tools"]
        buffer_text_for_tool_detection = has_tools and not is_cohere

        for event in response.data.events():
            if not event.data:
                continue
            try:
                raw = event.data
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")

                if sample_left > 0:
                    logger.debug("OCI_STREAM_EVENT %s", raw[:384])
                    sample_left -= 1

                data = json.loads(raw)

                # Handle both Generic and Cohere streaming formats
                if is_cohere:
                    text_chunk = data.get("text", "")
                    if text_chunk:
                        accumulated_text += text_chunk
                        if not buffer_text_for_tool_detection:
                            yield f"event: content_block_delta\ndata: {json.dumps({
                                'type': 'content_block_delta',
                                'index': current_block_index,
                                'delta': {'type': 'text_delta', 'text': text_chunk}
                            }, ensure_ascii=False)}\n\n"

                    finish_reason = data.get("finishReason")
                    if finish_reason:
                        if finish_reason == "COMPLETE":
                            stop_reason = STOP_REASON_END_TURN
                        elif finish_reason == "MAX_TOKENS":
                            stop_reason = STOP_REASON_MAX_TOKENS
                        continue
                else:
                    # Generic format
                    msg = data.get("message")
                    if isinstance(msg, dict):
                        contents = msg.get("content", [])
                        if isinstance(contents, list) and contents:
                            c0 = contents[0]
                            if isinstance(c0, dict) and c0.get("type") == "TEXT":
                                text_chunk = c0.get("text", "")
                                if text_chunk:
                                    accumulated_text += text_chunk
                                    if not buffer_text_for_tool_detection:
                                        yield f"event: content_block_delta\ndata: {json.dumps({
                                            'type': 'content_block_delta',
                                            'index': current_block_index,
                                            'delta': {'type': 'text_delta', 'text': text_chunk}
                                        }, ensure_ascii=False)}\n\n"

                    finish_reason = data.get("finishReason")
                    if finish_reason:
                        if finish_reason == "stop":
                            stop_reason = STOP_REASON_END_TURN
                        continue

                continue

            except json.JSONDecodeError:
                logger.warning("Invalid JSON chunk, skipped")
                continue
            except Exception as e:
                logger.warning(f"Chunk processing exception: {e}")
                continue

        # After streaming, process the accumulated text
        if not tool_call_detected and accumulated_text:
            logger.info(f"Processing accumulated text ({len(accumulated_text)} chars): {accumulated_text[:200]}...")
            if buffer_text_for_tool_detection:
                tool_calls_in_text = detect_all_tool_call_blocks(accumulated_text)
                logger.info(f"Tool detection result: found {len(tool_calls_in_text)} tool calls")
                if tool_calls_in_text:
                    # Remove all tool call blocks from the accumulated text
                    clean_text = accumulated_text
                    spans = sorted([tc.get("_span") for tc in tool_calls_in_text if tc.get("_span")], reverse=True)
                    for start, end in spans:
                        clean_text = clean_text[:start] + clean_text[end:]

                    # Yield clean text block if there's any remaining text
                    next_index = 0
                    if clean_text.strip():
                        yield f"event: content_block_delta\ndata: {json.dumps({
                            'type': 'content_block_delta',
                            'index': current_block_index,
                            'delta': {'type': 'text_delta', 'text': clean_text}
                        }, ensure_ascii=False)}\n\n"

                        yield f"event: content_block_stop\ndata: {json.dumps({
                            'type': 'content_block_stop',
                            'index': current_block_index
                        })}\n\n"
                        next_index = current_block_index + 1
                    else:
                        yield f"event: content_block_stop\ndata: {json.dumps({
                            'type': 'content_block_stop',
                            'index': current_block_index
                        })}\n\n"
                        next_index = current_block_index + 1

                    # Yield all tool call blocks
                    for tool_call in tool_calls_in_text:
                        tool_call.pop("_span", None)
                        yield f"event: content_block_start\ndata: {json.dumps({
                            'type': 'content_block_start',
                            'index': next_index,
                            'content_block': {'type': 'tool_use', 'id': tool_call['id'], 'name': tool_call['name']}
                        })}\n\n"

                        yield f"event: content_block_delta\ndata: {json.dumps({
                            'type': 'content_block_delta',
                            'index': next_index,
                            'delta': {
                                'type': 'input_json_delta',
                                'partial_json': json.dumps(tool_call['input'])
                            }
                        })}\n\n"

                        yield f"event: content_block_stop\ndata: {json.dumps({
                            'type': 'content_block_stop',
                            'index': next_index
                        })}\n\n"
                        next_index += 1

                    stop_reason = STOP_REASON_TOOL_USE
                    tool_call_detected = True
                    logger.info(f"Detected {len(tool_calls_in_text)} tool call block(s) in buffered text")
                else:
                    yield f"event: content_block_delta\ndata: {json.dumps({
                        'type': 'content_block_delta',
                        'index': current_block_index,
                        'delta': {'type': 'text_delta', 'text': accumulated_text}
                    }, ensure_ascii=False)}\n\n"

                    yield f"event: content_block_stop\ndata: {json.dumps({
                        'type': 'content_block_stop',
                        'index': current_block_index
                    })}\n\n"
                    logger.info("No tool calls detected in buffered text")
            else:
                yield f"event: content_block_stop\ndata: {json.dumps({
                    'type': 'content_block_stop',
                    'index': current_block_index
                })}\n\n"

        if in_tool_call:
            yield f"event: content_block_stop\ndata: {json.dumps({
                'type': 'content_block_stop',
                'index': current_block_index
            })}\n\n"

        estimated_output_tokens = max(1, (len(accumulated_text) + len(accumulated_tool_json)) // 4)
        yield f"event: message_delta\ndata: {json.dumps({
            'type': 'message_delta',
            'delta': {
                'stop_reason': stop_reason,
                'stop_sequence': stop_sequence
            },
            'usage': {
                'output_tokens': estimated_output_tokens,
                'input_tokens': 50
            }
        })}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Streaming output exception")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': 'An internal error occurred. Please check the logs for details.'}})}\n\n"
