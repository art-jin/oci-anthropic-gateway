import asyncio
import json
import logging
import re
import uuid
from typing import Optional, Union, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import oci
import uvicorn

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oci-gateway")

# --- Constants ---
DEFAULT_MAX_TOKENS = 131070  # Maximum tokens for generation (65535 * 2)

app = FastAPI(title="OCI GenAI Anthropic Gateway")


# ----------------------- Token Counting Helper Functions -----------------------

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a heuristic approach since we don't have access to the exact tokenizer.
    - English: ~4 characters per token
    - Chinese: ~1.5 characters per token
    - Code: ~3-4 characters per token
    """
    if not text:
        return 0

    # Count Chinese characters (CJK Unified Ideographs)
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))

    # Count non-Chinese characters
    non_chinese_chars = len(text) - chinese_chars

    # Estimate tokens
    # Chinese: ~1.5 chars per token, English: ~4 chars per token
    estimated_tokens = (chinese_chars / 1.5) + (non_chinese_chars / 4)

    return max(1, int(estimated_tokens))


def count_tokens_from_messages(messages: List[dict], system: Optional[Union[str, List[dict]]] = None) -> int:
    """
    Count tokens from Anthropic-format messages.

    Args:
        messages: List of message objects with role and content
        system: Optional system prompt (string or list of content blocks)

    Returns:
        Estimated token count
    """
    total_tokens = 0

    # Count system prompt tokens
    if system:
        if isinstance(system, str):
            total_tokens += estimate_tokens(system)
        elif isinstance(system, list):
            for block in system:
                if block.get("type") == "text":
                    total_tokens += estimate_tokens(block.get("text", ""))

    # Count message tokens
    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str):
            total_tokens += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    total_tokens += estimate_tokens(block.get("text", ""))
                elif block_type == "image":
                    # Images cost roughly the same regardless of size
                    # Base64 images are tokenized as a fixed cost
                    total_tokens += 85  # Approximate token cost for an image
                elif block_type == "tool_use":
                    # Estimate tokens for tool use
                    name = block.get("name", "")
                    input_data = block.get("input", {})
                    total_tokens += estimate_tokens(name)
                    total_tokens += estimate_tokens(json.dumps(input_data))
                elif block_type == "tool_result":
                    # Tool result content
                    result = block.get("content", block.get("result", ""))
                    if isinstance(result, str):
                        total_tokens += estimate_tokens(result)
                    elif isinstance(result, list):
                        for r in result:
                            if isinstance(r, dict) and r.get("type") == "text":
                                total_tokens += estimate_tokens(r.get("text", ""))
                            else:
                                total_tokens += estimate_tokens(str(r))

    return total_tokens


# ----------------------- Tool Calling Helper Functions -----------------------

def anthropic_to_oci_tools(tools: list) -> list:
    """Convert Anthropic tools to OCI ToolDefinition format"""
    oci_tools = []
    for tool in tools:
        oci_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", tool.get("parameters", {}))
            }
        }
        oci_tools.append(oci_tool)
    return oci_tools


def convert_tool_choice(tool_choice: Union[str, dict, None]) -> dict:
    """Convert Anthropic tool_choice to OCI format"""
    if tool_choice is None or tool_choice == "auto":
        return {"type": "auto"}
    elif tool_choice == "none":
        return {"type": "none"}
    elif tool_choice == "required":
        return {"type": "required"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return {"type": "auto"}


# ----------------------- Prompt Caching Helper Functions -----------------------

def extract_cache_control(blocks: Union[str, List[dict]]) -> dict:
    """
    Extract cache_control information from content blocks.

    Returns a dict with:
        - has_cache_control: bool - whether any block has cache_control
        - cached_blocks: int - number of blocks with cache_control
        - cache_types: list - list of cache control types found
    """
    result = {
        "has_cache_control": False,
        "cached_blocks": 0,
        "cache_types": []
    }

    if isinstance(blocks, str):
        return result

    for block in blocks:
        if isinstance(block, dict) and "cache_control" in block:
            cache_control = block["cache_control"]
            if isinstance(cache_control, dict):
                cache_type = cache_control.get("type", "unknown")
                result["has_cache_control"] = True
                result["cached_blocks"] += 1
                if cache_type not in result["cache_types"]:
                    result["cache_types"].append(cache_type)

    return result


def log_cache_info(content_type: str, cache_info: dict, index: Optional[int] = None):
    """Log cache control information for debugging."""
    if cache_info["has_cache_control"]:
        location = f"{content_type}[{index}]" if index is not None else content_type
        logger.info(f"Cache control detected in {location}: "
                   f"{cache_info['cached_blocks']} block(s), types: {cache_info['cache_types']}")


def convert_content_to_oci(content: Union[str, List[dict]]) -> Union[str, List]:
    """
    Convert Anthropic content to OCI format.

    Handles:
    - Plain text strings
    - text content blocks
    - image content blocks (base64 or URL)
    - tool_result content blocks
    - tool_use content blocks

    Returns:
        - str: If content is text-only (for backward compatibility)
        - List[Content]: If content includes images or multimodal content
    """
    if isinstance(content, str):
        return content

    text_parts = []
    image_content = []
    has_images = False

    for block in content:
        block_type = block.get("type", "")

        if block_type == "text":
            text_parts.append(block.get("text", ""))

        elif block_type == "image":
            has_images = True
            source = block.get("source", {})
            source_type = source.get("type", "")

            if source_type == "base64":
                # Handle base64 encoded image
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")

                # OCI expects base64 data without the data:image/...;base64, prefix
                # Convert Anthropic format to OCI format
                try:
                    # Check if data already has the prefix
                    if data.startswith("data:"):
                        # Extract base64 part after comma
                        data = data.split(",", 1)[1]

                    # Create OCI ImageContent
                    image_content.append(
                        oci.generative_ai_inference.models.ImageContent(
                            source_type="BASE64",
                            data=data,
                            format=media_type.split("/")[1].upper()  # "PNG", "JPEG", etc.
                        )
                    )
                    logger.info(f"Added base64 image (type: {media_type}, size: {len(data)} chars)")
                except Exception as e:
                    logger.warning(f"Failed to process base64 image: {e}")
                    text_parts.append("[IMAGE: Failed to process]")

            elif source_type == "url":
                # Handle image URL - OCI may not support direct URLs
                # For now, note this limitation
                url = source.get("url", "")
                logger.warning(f"Image URL not directly supported by OCI: {url}")
                text_parts.append(f"[IMAGE: {url}]")
            else:
                logger.warning(f"Unknown image source type: {source_type}")
                text_parts.append("[IMAGE: Unknown format]")

        elif block_type == "tool_result":
            # Convert tool_result to text for OCI
            # OCI doesn't have native tool_result support, so we format as text
            result = block.get("content", block.get("result", ""))
            if isinstance(result, list):
                # Handle nested content blocks in tool_result
                result = "".join([r.get("text", "") if isinstance(r, dict) else str(r) for r in result])
            elif isinstance(result, dict):
                result = json.dumps(result)
            text_parts.append(f"[TOOL_RESULT for {block.get('tool_use_id', 'unknown')}] {result}")

        elif block_type == "tool_use":
            # Convert tool_use to text format for OCI
            # Format: "Call function_name with arguments: {args}"
            name = block.get("name", "")
            input_data = block.get("input", {})
            text_parts.append(f"[TOOL_CALL] {name}: {json.dumps(input_data)}")

        else:
            # Unknown block type, try to stringify
            text_parts.append(str(block))

    # If we have images, return a list of Content objects
    if has_images:
        oci_content = []

        # Add text first if present
        if text_parts:
            oci_content.append(
                oci.generative_ai_inference.models.TextContent(text="".join(text_parts))
            )

        # Add images
        oci_content.extend(image_content)

        return oci_content

    # Text only - return string for backward compatibility
    return "".join(text_parts)


def extract_tool_calls_from_oci_response(chat_response) -> Optional[List[dict]]:
    """
    Extract tool calls from OCI response.

    Returns list of tool_use blocks in Anthropic format, or None if no tool calls.
    """
    tool_calls = []

    # Try different response structures that OCI might return
    if hasattr(chat_response, 'tool_calls') and chat_response.tool_calls:
        for tool_call in chat_response.tool_calls:
            if hasattr(tool_call, 'function'):
                func = tool_call.function
                name = func.name if hasattr(func, 'name') else ""
                arguments = func.arguments if hasattr(func, 'arguments') else "{}"

                try:
                    input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    input_data = {}

                tool_calls.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": input_data
                })

    elif hasattr(chat_response, 'message') and hasattr(chat_response.message, 'tool_calls'):
        for tool_call in chat_response.message.tool_calls:
            if hasattr(tool_call, 'function'):
                func = tool_call.function
                name = func.name if hasattr(func, 'name') else ""
                arguments = func.arguments if hasattr(func, 'arguments') else "{}"

                try:
                    input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    input_data = {}

                tool_calls.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": input_data
                })

    return tool_calls if tool_calls else None


def detect_tool_call_in_text(text: str) -> Optional[dict]:
    """
    Try to detect if OCI returned a tool call in text format.

    Some OCI models may return tool calls as text rather than structured data.
    This attempts to parse such responses.
    """
    # Pattern for: [TOOL_CALL] function_name: {"key": "value"}
    pattern = r'\[TOOL_CALL\]\s+(\w+):\s+(\{.*?\})$'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        name = match.group(1)
        arguments = match.group(2)
        try:
            input_data = json.loads(arguments)
            return {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": name,
                "input": input_data
            }
        except json.JSONDecodeError:
            pass

    return None

# ----------------------- Load Custom Configuration from File -----------------------
CONFIG_FILE = "config.json"  # Configuration file name (in the same directory)

try:
    with open(CONFIG_FILE, "r") as f:
        custom_config = json.load(f)

    COMPARTMENT_ID = custom_config["compartment_id"]
    ENDPOINT = custom_config["endpoint"]
    MODEL_ALIASES = custom_config.get("model_aliases", {})
    MODEL_DEFINITIONS = custom_config.get("model_definitions", {})
    DEFAULT_MODEL_NAME = custom_config.get("default_model")

    # Get the complete configuration object for the default model
    DEFAULT_MODEL_CONF = MODEL_DEFINITIONS.get(DEFAULT_MODEL_NAME)

    if not DEFAULT_MODEL_CONF:
        logger.error(f"Default model '{DEFAULT_MODEL_NAME}' not found in definitions")
        raise KeyError("Invalid default_model")

    logger.info(f"Config loaded. Default model: {DEFAULT_MODEL_NAME}")
except Exception as e:
    logger.error(f"Configuration initialization failed: {e}")
    raise

# Load OCI SDK configuration from default file
try:
    sdk_config = oci.config.from_file('~/.oci/config', "DEFAULT")
    genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=sdk_config,
        service_endpoint=ENDPOINT,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    logger.info("OCI SDK initialized successfully")
except Exception as e:
    logger.error(f"SDK configuration loading failed: {e}")
    raise


# ----------------------- Core non-streaming generation function -----------------------

async def generate_oci_non_stream(oci_messages, params, message_id, model_conf, requested_model):
    """
    Generate non-streaming response from OCI GenAI and format it as Anthropic-compatible JSON.
    """
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = oci_messages

    # --- Dynamic Parameter Adaptation ---
    # 1. Handle Max Tokens (determine which key to use based on model definition)
    token_limit = params.get("max_tokens", DEFAULT_MAX_TOKENS)
    tokens_key = model_conf.get("max_tokens_key", "max_tokens")
    setattr(chat_request, tokens_key, token_limit)

    # 2. Handle Temperature (prioritize hardcoded value in model definition, otherwise use request value)
    chat_request.temperature = model_conf.get("temperature", params.get("temperature", 0.7))

    # 3. Handle Top-K (limits token selection to top k most probable tokens)
    if "top_k" in params:
        top_k_value = params["top_k"]
        if isinstance(top_k_value, (int, float)) and top_k_value > 0:
            setattr(chat_request, "top_k", int(top_k_value))
            logger.info(f"Set top_k: {top_k_value}")

    # 4. Handle Top-P (nucleus sampling - considers smallest set whose cumulative prob exceeds p)
    if "top_p" in params:
        top_p_value = params["top_p"]
        if isinstance(top_p_value, (int, float)) and 0 < top_p_value <= 1:
            setattr(chat_request, "top_p", float(top_p_value))
            logger.info(f"Set top_p: {top_p_value}")

    # 5. Handle Stop Sequences
    if "stop_sequences" in params and params["stop_sequences"]:
        # OCI expects a list of stop strings
        setattr(chat_request, "stop", params["stop_sequences"])
        logger.info(f"Set stop_sequences: {params['stop_sequences']}")

    # 6. Handle Thinking Mode (Extended Thinking)
    # Anthropic's thinking mode enables the model to show its reasoning process
    # OCI doesn't natively support this, so we add it as a system instruction
    thinking_config = params.get("thinking")
    if thinking_config:
        thinking_type = thinking_config.get("type", "disabled")
        if thinking_type == "enabled":
            budget_tokens = thinking_config.get("budget_tokens", 16000)
            logger.info(f"Thinking mode enabled with budget: {budget_tokens} tokens")
            # Add thinking instruction as a prefix system message
            # This encourages the model to show its reasoning
            thinking_instruction = (
                "Please think through this problem step by step, showing your reasoning process. "
                f"Use up to {budget_tokens} tokens for your thinking if needed. "
                "Structure your response to clearly separate your thinking from your final answer."
            )
            # Prepend thinking instruction to messages
            oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                role="SYSTEM",
                content=[oci.generative_ai_inference.models.TextContent(text=thinking_instruction)]
            ))

    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.is_stream = False

    # --- Tool Calling Support ---
    # Add tools to request if present
    if "tools" in params:
        oci_tools = anthropic_to_oci_tools(params["tools"])
        setattr(chat_request, "tools", oci_tools)

    if "tool_choice" in params:
        oci_tool_choice = convert_tool_choice(params["tool_choice"])
        setattr(chat_request, "tool_choice", oci_tool_choice)

    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = COMPARTMENT_ID
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_conf["ocid"])

    try:
        # Call OCI (non-streaming) - run in thread pool to avoid blocking event loop
        response = await asyncio.to_thread(genai_client.chat, chat_detail)

        # Parse the response
        accumulated_text = ""
        content_blocks = []
        stop_reason = "end_turn"
        stop_sequence = None
        stop_sequences = params.get("stop_sequences", [])
        response_data = response.data

        # Try to extract tool calls first, then text from the response
        if hasattr(response_data, 'chat_response'):
            chat_response = response_data.chat_response

            # Check for structured tool calls
            tool_calls = extract_tool_calls_from_oci_response(chat_response)

            if tool_calls:
                # Found tool calls in the response
                content_blocks = tool_calls
                stop_reason = "tool_use"
                logger.info(f"Detected {len(tool_calls)} tool call(s) in response")
            else:
                # Check for finish_reason in choices
                if hasattr(chat_response, 'choices') and chat_response.choices:
                    choice = chat_response.choices[0]
                    if hasattr(choice, 'finish_reason'):
                        finish_reason = choice.finish_reason
                        if finish_reason == "stop":
                            stop_reason = "stop_sequence"
                            # Try to detect which stop sequence was triggered
                            # by checking the accumulated_text end
                # No tool calls, extract text content
                # No tool calls, extract text content
                # Handle different response structures
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

                # Also check if text contains a tool call pattern (for models that return as text)
                if accumulated_text:
                    tool_call_in_text = detect_tool_call_in_text(accumulated_text)
                    if tool_call_in_text:
                        content_blocks = [tool_call_in_text]
                        stop_reason = "tool_use"
                        logger.info("Detected tool call in text response")
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

                    # Check for finish_reason to detect stop sequences
                    finish_reason = choices[0].get("finish_reason", "")
                    if finish_reason == "stop":
                        stop_reason = "stop_sequence"
                        # Try to detect which stop sequence was triggered
                        for seq in stop_sequences:
                            if accumulated_text.endswith(seq):
                                stop_sequence = seq
                                break

                    # Check for tool_calls in the message
                    if "tool_calls" in msg_data:
                        for tc in msg_data["tool_calls"]:
                            func = tc.get("function", {})
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": func.get("name", ""),
                                "input": json.loads(func.get("arguments", "{}"))
                            })
                        stop_reason = "tool_use"
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
        estimated_input_tokens = sum(len(str(m.content)) // 4 for m in oci_messages)

        # Check if there was cache_control in the request for usage reporting
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

        # Extract cache info from params (if available)
        system_prompt = params.get("system")
        if system_prompt and isinstance(system_prompt, list):
            cache_info = extract_cache_control(system_prompt)
            if cache_info["has_cache_control"] and "ephemeral" in cache_info["cache_types"]:
                # Estimate cached tokens in system prompt
                for block in system_prompt:
                    if block.get("type") == "text" and "cache_control" in block:
                        cache_creation_input_tokens += estimate_tokens(block.get("text", ""))

        for msg in params.get("messages", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                cache_info = extract_cache_control(content)
                if cache_info["has_cache_control"]:
                    for block in content:
                        if block.get("type") == "text" and "cache_control" in block:
                            cache_read_input_tokens += estimate_tokens(block.get("text", ""))

        # Build usage object
        usage = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens
        }

        # Add cache usage metrics if caching was detected
        if cache_creation_input_tokens > 0:
            usage["cache_creation_input_tokens"] = cache_creation_input_tokens
            logger.info(f"Cache creation: {cache_creation_input_tokens} tokens")

        if cache_read_input_tokens > 0:
            usage["cache_read_input_tokens"] = cache_read_input_tokens
            logger.info(f"Cache read: {cache_read_input_tokens} tokens")

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


# ----------------------- Core streaming generation function -----------------------

async def generate_oci_stream(oci_messages, params, message_id, model_conf, requested_model):
    """
    Generate streaming response from OCI GenAI and format it as Anthropic-compatible SSE.
    """
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = oci_messages

    # --- Dynamic Parameter Adaptation ---
    # 1. Handle Max Tokens (determine which key to use based on model definition)
    token_limit = params.get("max_tokens", DEFAULT_MAX_TOKENS)
    tokens_key = model_conf.get("max_tokens_key", "max_tokens")
    setattr(chat_request, tokens_key, token_limit)

    # 2. Handle Temperature (prioritize hardcoded value in model definition, otherwise use request value)
    # Addresses previous 400 errors: if model definition specifies 1.0, force 1.0
    chat_request.temperature = model_conf.get("temperature", params.get("temperature", 0.7))

    # 3. Handle Top-K (limits token selection to top k most probable tokens)
    if "top_k" in params:
        top_k_value = params["top_k"]
        if isinstance(top_k_value, (int, float)) and top_k_value > 0:
            setattr(chat_request, "top_k", int(top_k_value))
            logger.info(f"Set top_k: {top_k_value}")

    # 4. Handle Top-P (nucleus sampling - considers smallest set whose cumulative prob exceeds p)
    if "top_p" in params:
        top_p_value = params["top_p"]
        if isinstance(top_p_value, (int, float)) and 0 < top_p_value <= 1:
            setattr(chat_request, "top_p", float(top_p_value))
            logger.info(f"Set top_p: {top_p_value}")

    # 5. Handle Stop Sequences
    if "stop_sequences" in params and params["stop_sequences"]:
        # OCI expects a list of stop strings
        setattr(chat_request, "stop", params["stop_sequences"])
        logger.info(f"Set stop_sequences: {params['stop_sequences']}")

    # 6. Handle Thinking Mode (Extended Thinking)
    # Anthropic's thinking mode enables the model to show its reasoning process
    # OCI doesn't natively support this, so we add it as a system instruction
    thinking_config = params.get("thinking")
    if thinking_config:
        thinking_type = thinking_config.get("type", "disabled")
        if thinking_type == "enabled":
            budget_tokens = thinking_config.get("budget_tokens", 16000)
            logger.info(f"Thinking mode enabled with budget: {budget_tokens} tokens")
            # Add thinking instruction as a prefix system message
            thinking_instruction = (
                "Please think through this problem step by step, showing your reasoning process. "
                f"Use up to {budget_tokens} tokens for your thinking if needed. "
                "Structure your response to clearly separate your thinking from your final answer."
            )
            # Prepend thinking instruction to messages
            oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                role="SYSTEM",
                content=[oci.generative_ai_inference.models.TextContent(text=thinking_instruction)]
            ))

    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.is_stream = True

    # --- Tool Calling Support ---
    # Add tools to request if present
    if "tools" in params:
        oci_tools = anthropic_to_oci_tools(params["tools"])
        setattr(chat_request, "tools", oci_tools)

    if "tool_choice" in params:
        oci_tool_choice = convert_tool_choice(params["tool_choice"])
        setattr(chat_request, "tool_choice", oci_tool_choice)

    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = COMPARTMENT_ID
    # Use the OCID from the configuration object
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_conf["ocid"])

    try:
        # 1. Required starting events (Claude Code needs these to initialize display)
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
                'usage': {'input_tokens': 10, 'output_tokens': 0}  # Rough estimate for input
            }
        }

        # Add metadata to message_start if present in request
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

        accumulated_text = ""  # Used for usage estimation and debugging
        accumulated_tool_json = ""  # For accumulating partial tool call JSON
        current_block_index = 0
        in_tool_call = False
        tool_call_detected = False
        pending_tool_name = None
        stop_reason = "end_turn"
        stop_sequence = None
        stop_sequences = params.get("stop_sequences", [])

        for event in response.data.events():
            if not event.data:
                continue
            try:
                data = json.loads(event.data)

                # Check for tool_calls in the delta (OpenAI-style format)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})

                    # Check for tool calls in delta
                    if "tool_calls" in delta:
                        tool_calls = delta["tool_calls"]
                        if tool_calls:
                            if not in_tool_call:
                                # Starting a new tool call block
                                in_tool_call = True
                                tool_call_detected = True

                                # Close previous text block if any
                                if accumulated_text:
                                    yield f"event: content_block_stop\ndata: {json.dumps({
                                        'type': 'content_block_stop',
                                        'index': current_block_index
                                    })}\n\n"
                                    current_block_index += 1

                                # Emit content_block_start for tool_use
                                tool_call = tool_calls[0]
                                function = tool_call.get("function", {})
                                pending_tool_name = function.get("name", "")

                                yield f"event: content_block_start\ndata: {json.dumps({
                                    'type': 'content_block_start',
                                    'index': current_block_index,
                                    'content_block': {'type': 'tool_use', 'id': tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}"), 'name': pending_tool_name}
                                })}\n\n"

                            # Stream the arguments
                            for tool_call in tool_calls:
                                function = tool_call.get("function", {})
                                arguments = function.get("arguments", "")
                                if arguments:
                                    accumulated_tool_json += arguments
                                    yield f"event: content_block_delta\ndata: {json.dumps({
                                        'type': 'content_block_delta',
                                        'index': current_block_index,
                                        'delta': {
                                            'type': 'input_json_delta',
                                            'partial_json': accumulated_tool_json
                                        }
                                    })}\n\n"

                    # Check for text content
                    elif "content" in delta:
                        text_chunk = delta["content"]
                        if text_chunk:
                            accumulated_text += text_chunk
                            yield f"event: content_block_delta\ndata: {json.dumps({
                                'type': 'content_block_delta',
                                'index': current_block_index,
                                'delta': {
                                    'type': 'text_delta',
                                    'text': text_chunk
                                }
                            })}\n\n"

                    # Check for finish_reason
                    elif "finish_reason" in choices[0]:
                        finish_reason = choices[0]["finish_reason"]
                        if finish_reason == "tool_calls" or finish_reason == "function_call":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "stop_sequence"
                            # Check which stop sequence was triggered
                            for seq in stop_sequences:
                                if accumulated_text.endswith(seq):
                                    stop_sequence = seq
                                    break

                else:
                    # Fallback to message.content format
                    text_chunk = data.get("message", {}).get("content", [{}])[0].get("text", "")
                    if text_chunk:
                        accumulated_text += text_chunk
                        yield f"event: content_block_delta\ndata: {json.dumps({
                            'type': 'content_block_delta',
                            'index': current_block_index,
                            'delta': {
                                'type': 'text_delta',
                                'text': text_chunk
                            }
                        })}\n\n"

            except json.JSONDecodeError:
                logger.warning("Invalid JSON chunk, skipped")
                continue
            except Exception as e:
                logger.warning(f"Chunk processing exception: {e}")
                continue

        # After streaming, check if accumulated text contains a tool call pattern
        if not tool_call_detected and accumulated_text:
            tool_call_in_text = detect_tool_call_in_text(accumulated_text)
            if tool_call_in_text:
                # Need to retroactively convert the text to a tool call
                yield f"event: content_block_stop\ndata: {json.dumps({
                    'type': 'content_block_stop',
                    'index': 0
                })}\n\n"

                yield f"event: content_block_start\ndata: {json.dumps({
                    'type': 'content_block_start',
                    'index': 1,
                    'content_block': {'type': 'tool_use', 'id': tool_call_in_text['id'], 'name': tool_call_in_text['name']}
                })}\n\n"

                yield f"event: content_block_delta\ndata: {json.dumps({
                    'type': 'content_block_delta',
                    'index': 1,
                    'delta': {
                        'type': 'input_json_delta',
                        'partial_json': json.dumps(tool_call_in_text['input'])
                    }
                })}\n\n"

                yield f"event: content_block_stop\ndata: {json.dumps({
                    'type': 'content_block_stop',
                    'index': 1
                })}\n\n"

                stop_reason = "tool_use"
                tool_call_detected = True
            else:
                # Normal text block ending
                yield f"event: content_block_stop\ndata: {json.dumps({
                    'type': 'content_block_stop',
                    'index': current_block_index
                })}\n\n"

        # Close tool call block if open
        if in_tool_call:
            yield f"event: content_block_stop\ndata: {json.dumps({
                'type': 'content_block_stop',
                'index': current_block_index
            })}\n\n"

        # 3. Required ending sequence (prevents client from "thinking" indefinitely)
        estimated_output_tokens = max(1, (len(accumulated_text) + len(accumulated_tool_json)) // 4)  # Rough estimation
        yield f"event: message_delta\ndata: {json.dumps({
            'type': 'message_delta',
            'delta': {
                'stop_reason': stop_reason,
                'stop_sequence': stop_sequence
            },
            'usage': {
                'output_tokens': estimated_output_tokens,
                'input_tokens': 50  # Rough estimate
            }
        })}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Streaming output exception")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': 'An internal error occurred. Please check the logs for details.'}})}\n\n"


# ----------------------- Routes -----------------------

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
        try:
            body = await request.json()
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

    # Handle messages request
    if "messages" in path:
        body = await request.json()
        req_model = body.get("model", "").lower()

        # Lookup logic
        target_alias = None
        for k, v in MODEL_ALIASES.items():
            if k in req_model:
                target_alias = v
                break

        if not target_alias and req_model in MODEL_DEFINITIONS:
            target_alias = req_model

        # Retrieve the complete dictionary object, e.g., {"ocid": "...", "temperature": 1.0}
        selected_model_conf = MODEL_DEFINITIONS.get(target_alias)

        if not selected_model_conf:
            selected_model_conf = DEFAULT_MODEL_CONF
            logger.info(f"Using default model config for: {req_model}")
        else:
            logger.info(f"Mapped '{req_model}' -> '{target_alias}' config")

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
                logger.info(f"Added system prompt ({len(system_text)} chars){cache_suffix}")

        # Convert regular messages
        messages = body.get("messages", [])
        for idx, m in enumerate(messages):
            # Check for cache_control in message content
            content = m.get("content", [])
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
                logger.info(f"Added multimodal message {idx} with {image_count} image(s){cache_suffix}")
            else:
                # Text only content
                msg = oci.generative_ai_inference.models.Message(
                    role=m["role"].upper(),
                    content=[oci.generative_ai_inference.models.TextContent(text=converted_content)]
                )
                cache_suffix = " [CACHED]" if cache_info["has_cache_control"] else ""
                if cache_info["has_cache_control"]:
                    log_cache_info(f"message[{idx}]", cache_info)
                logger.info(f"Added message {idx} ({len(converted_content)} chars){cache_suffix}")

            oci_msgs.append(msg)

        # Handle metadata - extract for logging and response
        metadata = body.get("metadata")
        if metadata:
            logger.info(f"Request metadata: {metadata}")

        message_id = f"msg_oci_{uuid.uuid4().hex}"

        if body.get("stream", False):
            return StreamingResponse(
                generate_oci_stream(oci_msgs, body, message_id, selected_model_conf, req_model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Prevent proxy buffering for streaming
                }
            )
        else:
            # Non-streaming logic
            return await generate_oci_non_stream(oci_msgs, body, message_id, selected_model_conf, req_model)

    return {"detail": "Not Found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")