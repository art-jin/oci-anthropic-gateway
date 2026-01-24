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
MIN_JSON_LENGTH = 10  # Minimum valid JSON length for tool calls

# Precompiled regex patterns for tool call detection (performance optimization)
_TOOL_CALL_START_PATTERN = re.compile(r'<TOOL_CALL\s*>', re.IGNORECASE)
_TOOL_CALL_END_PATTERN = re.compile(r'</TOOL_CALL\s*>', re.IGNORECASE)

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
    """Convert Anthropic tools to OCI ToolDefinition format (for Generic API)"""
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


def anthropic_to_cohere_tools(tools: list) -> list:
    """Convert Anthropic tools to CohereTool format (for Cohere API)"""
    cohere_tools = []
    for tool in tools:
        # Get input schema
        input_schema = tool.get("input_schema", tool.get("parameters", {}))
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Convert to Cohere parameter definitions
        parameter_definitions = {}
        for param_name, param_info in properties.items():
            param_def = oci.generative_ai_inference.models.CohereParameterDefinition(
                description=param_info.get("description", ""),
                type=_convert_json_type_to_python(param_info.get("type", "str")),
                is_required=param_name in required
            )
            parameter_definitions[param_name] = param_def

        cohere_tool = oci.generative_ai_inference.models.CohereTool(
            name=tool["name"],
            description=tool.get("description", ""),
            parameter_definitions=parameter_definitions
        )
        cohere_tools.append(cohere_tool)
    return cohere_tools


def _convert_json_type_to_python(json_type: str) -> str:
    """Convert JSON Schema type to Python type string for Cohere"""
    type_mapping = {
        "string": "str",
        "str": "str",
        "integer": "int",
        "number": "float",
        "float": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict"
    }
    return type_mapping.get(json_type, "str")


def _build_tool_use_instruction(tools: list, tool_choice: str) -> str:
    """
    Build tool use instruction prompt for Generic format models.

    This creates a system prompt that instructs the model on how to use tools
    when it doesn't have native function calling support.
    """
    if not tools:
        return ""

    # Build tool descriptions
    tool_list = []
    for tool in tools:
        # Handle different tool formats
        if isinstance(tool, dict):
            name = tool.get("name")
            desc = tool.get("description", "")
            # Support both input_schema and parameters
            schema = tool.get("input_schema") or tool.get("parameters") or {}
        else:
            logger.warning(f"Unexpected tool format: {type(tool)}")
            continue

        if not name:
            logger.warning(f"Tool missing 'name' field: {tool}")
            continue

        # Build parameter descriptions
        params_desc = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if properties:
            for param, info in properties.items():
                req_mark = "**(required)**" if param in required else "(optional)"
                param_desc = info.get('description', '')
                param_type = info.get('type', 'string')
                params_desc.append(f"  - `{param}` {req_mark} ({param_type}): {param_desc}")

        params_str = "\n".join(params_desc) if params_desc else "  (no parameters)"

        tool_list.append(f"""**{name}**: {desc}
Parameters:
{params_str}""")

    # Tool choice strategy description
    strategy_map = {
        "auto": "Use tools when the user's request requires it. If the user's question can be answered directly without tools, respond normally.",
        "required": "You MUST use at least one tool to address the user's request. Do not answer without calling a tool.",
        "none": "Do not use any tools. Answer the user's question directly.",
        "any": "Use tools when appropriate for the user's request."
    }
    strategy = strategy_map.get(tool_choice, "auto")

    # Few-shot examples - more explicit format
    examples = """## Examples

**Example 1 - Using a single tool:**
User: "What's the weather in Tokyo?"
<TOOL_CALL>
{"name": "WebSearch", "input": {"query": "weather in Tokyo"}}
</TOOL_CALL>

**Example 2 - Direct answer (no tool needed):**
User: "Hello, how are you?"
I'm doing well, thank you for asking! How can I help you today?

**Example 3 - Multiple tools:**
User: "Search for Python tutorials and read the file README.md"
<TOOL_CALL>
{"name": "WebSearch", "input": {"query": "Python tutorials"}}
</TOOL_CALL>
<TOOL_CALL>
{"name": "Read", "input": {"file_path": "README.md"}}
</TOOL_CALL>

**Example 4 - Tool with no output:**
User: "Create a file named test.txt"
<TOOL_CALL>
{"name": "Write", "input": {"file_path": "test.txt", "content": ""}}
</TOOL_CALL>

**IMPORTANT**: When you use a tool, output ONLY the <TOOL_CALL> block(s). Do not include any additional text before or after the tool call(s)."""

    return f"""# Tool Use Guidelines

You have access to the following tools that you can use to help answer user requests:

{chr(10).join(tool_list)}

## Tool Call Format

When you need to use a tool, output your response in this exact format:

<TOOL_CALL>
{{
  "name": "tool_name",
  "input": {{
    "parameter1": "value1",
    "parameter2": "value2"
  }}
}}
</TOOL_CALL>

**CRITICAL RULES**:

1. When you need to use a tool, output ONLY the <TOOL_CALL> block(s)
2. Do NOT include any additional text, explanations, or formatting outside the <TOOL_CALL> tags
3. Tool selection strategy: {strategy}
4. Output the tool call in the EXACT format shown above (with <TOOL_CALL> opening and closing tags)
5. You can output multiple tool calls by placing multiple <TOOL_CALL> blocks one after another
6. If no tools are needed, respond directly to the user in natural language
7. Do not invent or make up parameter values - if a required parameter is not provided, ask the user

{examples}

Remember: When using tools, output ONLY the tool call block(s) with no additional text!"""


def convert_tool_choice(tool_choice: Union[str, dict, None]) -> dict:
    """Convert Anthropic tool_choice to OCI format (for Generic API)"""
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
            # Format: <TOOL_CALL>{"name": "tool_name", "input": {...}}</TOOL_CALL>
            name = block.get("name", "")
            input_data = block.get("input", {})
            tool_call_json = json.dumps({"name": name, "input": input_data})
            text_parts.append(f"<TOOL_CALL>{tool_call_json}</TOOL_CALL>")

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
    Extract tool calls from OCI Generic API response.

    Returns list of tool_use blocks in Anthropic format, or None if no tool calls.
    """
    tool_calls = []
    logger.debug(f"Starting OCI response tool call extraction, response type: {type(chat_response)}")

    # Try different response structures that OCI might return
    if hasattr(chat_response, 'tool_calls') and chat_response.tool_calls:
        logger.info(f"Found tool_calls attribute with {len(chat_response.tool_calls)} tool call(s)")
        for i, tool_call in enumerate(chat_response.tool_calls):
            if hasattr(tool_call, 'function'):
                func = tool_call.function
                name = func.name if hasattr(func, 'name') else ""
                arguments = func.arguments if hasattr(func, 'arguments') else "{}"

                try:
                    input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                    logger.debug(f"Tool call {i}: {name} with {len(arguments)} chars of arguments")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse arguments for tool call {i} ({name}): {e}")
                    input_data = {}

                tool_calls.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": input_data
                })
            else:
                logger.warning(f"Tool call {i} missing 'function' attribute")

    elif hasattr(chat_response, 'message') and hasattr(chat_response.message, 'tool_calls'):
        logger.info(f"Found message.tool_calls attribute with {len(chat_response.message.tool_calls)} tool call(s)")
        for i, tool_call in enumerate(chat_response.message.tool_calls):
            if hasattr(tool_call, 'function'):
                func = tool_call.function
                name = func.name if hasattr(func, 'name') else ""
                arguments = func.arguments if hasattr(func, 'arguments') else "{}"

                try:
                    input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                    logger.debug(f"Tool call {i}: {name} with {len(arguments)} chars of arguments")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse arguments for tool call {i} ({name}): {e}")
                    input_data = {}

                tool_calls.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": input_data
                })
            else:
                logger.warning(f"Tool call {i} missing 'function' attribute")
    else:
        logger.debug("No tool calls found in OCI response structure")

    if tool_calls:
        logger.info(f"Extracted {len(tool_calls)} tool call(s) from OCI response")
    else:
        logger.debug("No tool calls extracted from OCI response")

    return tool_calls if tool_calls else None


def extract_tool_calls_from_cohere_response(cohere_response) -> Optional[List[dict]]:
    """
    Extract tool calls from OCI Cohere API response.

    Returns list of tool_use blocks in Anthropic format, or None if no tool calls.
    """
    tool_calls = []
    logger.debug(f"Starting Cohere response tool call extraction, response type: {type(cohere_response)}")

    if hasattr(cohere_response, 'tool_calls') and cohere_response.tool_calls:
        logger.info(f"Found Cohere tool_calls attribute with {len(cohere_response.tool_calls)} tool call(s)")
        for i, tool_call in enumerate(cohere_response.tool_calls):
            name = tool_call.name if hasattr(tool_call, 'name') else ""
            parameters = tool_call.parameters if hasattr(tool_call, 'parameters') else {}

            logger.debug(f"Cohere tool call {i}: {name} with {len(parameters)} parameters")

            tool_calls.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": name,
                "input": dict(parameters) if isinstance(parameters, dict) else {}
            })
    else:
        logger.debug("No tool calls found in Cohere response")

    if tool_calls:
        logger.info(f"Extracted {len(tool_calls)} tool call(s) from Cohere response")
    else:
        logger.debug("No tool calls extracted from Cohere response")

    return tool_calls if tool_calls else None


def convert_content_to_cohere_message(content: Union[str, List[dict]], role: str) -> Union[
    oci.generative_ai_inference.models.CohereUserMessage,
    oci.generative_ai_inference.models.CohereChatBotMessage,
    oci.generative_ai_inference.models.CohereToolMessage
]:
    """
    Convert Anthropic content to Cohere message format.

    Handles:
    - text content blocks -> CohereUserMessage.message
    - tool_use content blocks -> CohereChatBotMessage with tool_calls
    - tool_result content blocks -> CohereToolMessage with tool_results

    Args:
        content: Anthropic format content (string or list of blocks)
        role: Message role ("user" or "assistant")

    Returns:
        CohereUserMessage, CohereChatBotMessage, or CohereToolMessage
    """
    role_upper = role.upper()

    # Extract text from content
    text_parts = []
    tool_calls = []
    tool_results = []

    if isinstance(content, str):
        text_parts.append(content)
    else:
        for block in content:
            block_type = block.get("type", "")

            if block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type == "tool_use":
                # Convert tool_use to CohereToolCall
                name = block.get("name", "")
                input_data = block.get("input", {})
                tool_calls.append(oci.generative_ai_inference.models.CohereToolCall(
                    name=name,
                    parameters=input_data
                ))

            elif block_type == "tool_result":
                # Convert tool_result to CohereToolResult
                tool_use_id = block.get("tool_use_id", "")
                result = block.get("content", block.get("result", ""))

                # Find corresponding tool call name (might need context)
                # For now, use a placeholder name
                tool_call_ref = oci.generative_ai_inference.models.CohereToolCall(
                    name=block.get("name", "unknown"),
                    parameters={}
                )

                # Convert result to outputs format
                if isinstance(result, list):
                    outputs = [r.get("text", str(r)) if isinstance(r, dict) else str(r) for r in result]
                elif isinstance(result, dict):
                    outputs = [json.dumps(result)]
                else:
                    outputs = [str(result)]

                tool_results.append(oci.generative_ai_inference.models.CohereToolResult(
                    call=tool_call_ref,
                    outputs=outputs
                ))

    text_content = "".join(text_parts)

    # Create appropriate message type based on role and content
    if role_upper == "USER":
        return oci.generative_ai_inference.models.CohereUserMessage(message=text_content)

    elif role_upper == "ASSISTANT" or role_upper == "CHATBOT":
        if tool_calls:
            return oci.generative_ai_inference.models.CohereChatBotMessage(
                message=text_content,
                tool_calls=tool_calls
            )
        else:
            return oci.generative_ai_inference.models.CohereChatBotMessage(message=text_content)

    elif role_upper == "TOOL":
        return oci.generative_ai_inference.models.CohereToolMessage(tool_results=tool_results)

    # Default to user message
    return oci.generative_ai_inference.models.CohereUserMessage(message=text_content)


def convert_anthropic_messages_to_cohere(messages: List[dict], system: Optional[str] = None) -> tuple:
    """
    Convert Anthropic messages format to Cohere format.

    Cohere format:
    - message: The current user message (string)
    - chat_history: List of previous messages (CohereUserMessage, CohereChatBotMessage, etc.)

    Args:
        messages: List of Anthropic format messages
        system: Optional system prompt

    Returns:
        tuple: (current_message: str, chat_history: list, system_message: str or None)
    """
    chat_history = []
    current_message = ""
    system_message = system

    if not messages:
        return "", [], system_message

    # Process all messages except the last one as chat history
    for msg in messages[:-1]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        cohere_msg = convert_content_to_cohere_message(content, role)
        chat_history.append(cohere_msg)

    # The last message is the current message
    last_msg = messages[-1]
    last_role = last_msg.get("role", "user")
    last_content = last_msg.get("content", "")

    if isinstance(last_content, str):
        current_message = last_content
    else:
        # Extract text from content blocks
        text_parts = []
        for block in last_content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        current_message = "".join(text_parts)

    return current_message, chat_history, system_message


def detect_tool_call_in_text(text: str) -> Optional[dict]:
    """
    Try to detect if OCI returned a tool call in text format.

    Some OCI models may return tool calls as text rather than structured data.
    This attempts to parse such responses.
    """
    # Pattern for: [TOOL_CALL] function_name: {"key": "value"}
    #pattern = r'\[TOOL_CALL\]\s+(\w+):\s+(\{.*?\})$'。## GPT5 认为不闭合
    #pattern = r'\[TOOL_CALL\]\s+(\w+):\s+(\{.*\})'
    pattern = r'\[TOOL_CALL\]\s+(\w+):\s+(\{.*?\})'
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




def detect_tool_call_block(text: str) -> Optional[dict]:
    """
    Detect <tool_call>JSON</tool_call> block and parse it.

    Expected JSON format:
      <tool_call>
      {"name":"Read","input":{"file_path":"..."}}
      </tool_call>

    Returns a tool_use dict in Anthropic format, plus an internal "_span"
    (start_idx, end_idx) indicating the block span in the original text.

    Includes a compatibility fix:
      - If name == "Read" and input has "path" but not "file_path",
        it will rewrite "path" -> "file_path".
    """
    start_tag = "<TOOL_CALL>"
    end_tag = "</TOOL_CALL>"

    start = text.find(start_tag)
    if start == -1:
        return None

    end = text.find(end_tag, start + len(start_tag))
    if end == -1:
        return None  # not closed yet

    json_str = text[start + len(start_tag):end].strip()

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    name = payload.get("name")
    input_data = payload.get("input", {})

    if not isinstance(name, str) or not name:
        return None
    if not isinstance(input_data, dict):
        return None

    # Compatibility fix: Read expects file_path, but some outputs may use "path"
    if name == "Read" and "file_path" not in input_data and "path" in input_data:
        input_data["file_path"] = input_data.pop("path")

    return {
        "type": "tool_use",
        "id": f"toolu_{uuid.uuid4().hex[:24]}",
        "name": name,
        "input": input_data,
        "_span": (start, end + len(end_tag)),
    }


def _advance_search_position(end_match, start, start_tag_length):
    """
    Helper to advance search position after processing a tool call block.

    Args:
        end_match: Regex match object for end tag, or None
        start: Start position of the block
        start_tag_length: Length of the start tag

    Returns:
        Next search position
    """
    return end_match.end() if end_match else start + start_tag_length


def _quote_property_name(match):
    """Helper function for fixing unquoted JSON property names."""
    prop_name = match.group(1)
    return f'"{prop_name}":'


def _unquote_number(match):
    """Helper function for fixing quoted numbers in JSON."""
    return match.group(1)


def fix_json_issues(json_str: str) -> Optional[str]:
    """
    Attempt to fix common JSON formatting issues in malformed JSON strings.

    Handles:
    - Trailing commas in objects/arrays
    - Single quotes instead of double quotes
    - Unquoted property names
    - Missing quotes around string values
    - Extra commas
    - Escaped characters that shouldn't be escaped

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Fixed JSON string if fixable, None if not
    """
    if not json_str:
        return None

    original = json_str
    try:
        # Remove leading/trailing whitespace first
        json_str = json_str.strip()

        # Fix 1: Remove trailing commas (e.g., {"a": 1,} or {"a": 1, "b": 2,})
        # Be careful to only remove commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Fix 2: Replace single quotes with double quotes for strings
        # This is tricky - we need to be careful not to replace quotes inside strings
        # Simple approach: replace single-quoted property names and values
        # Match pattern: 'key': or 'value', but not inside double-quoted strings
        # Replace single quotes around property names (at start of object or after comma)
        json_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)

        # Fix 3: Ensure all property names are quoted
        # Match unquoted property names (word characters before colon)
        # Pattern: word followed by colon, but not preceded by quote or bracket
        # This is a simple heuristic - may not catch all cases
        json_str = re.sub(r'(\w+)(\s*:)', _quote_property_name, json_str)

        # Fix 4: Fix boolean/null values that might be quoted
        json_str = re.sub(r'"(true|false|null)"', r'\1', json_str)

        # Fix 5: Fix numbers that might be quoted (simple heuristic)
        # Match quoted numbers like "123" or "12.34"
        json_str = re.sub(r'"(\d+(?:\.\d+)?)"', _unquote_number, json_str)

        # Fix 6: Remove control characters that might break JSON
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)

        # Try to parse the fixed JSON
        json.loads(json_str)

        # If we got here, the JSON is now valid
        if json_str != original:
            logger.debug(f"Fixed JSON issues: {original[:100]}... -> {json_str[:100]}...")

        return json_str

    except (json.JSONDecodeError, re.error) as e:
        # If still invalid, return None
        logger.debug(f"Could not fix JSON: {e}")
        return None


def _extract_and_parse_json(json_str: str, start: int, end_match, start_tag_length: int, text_length: int) -> tuple:
    """
    Extract, clean, and parse JSON from a tool call block.

    Args:
        json_str: Raw JSON string extracted between tags
        start: Start position of the block
        end_match: Regex match object for end tag, or None
        start_tag_length: Length of the start tag
        text_length: Total length of the text being processed

    Returns:
        tuple: (payload, span_end, search_position) where:
            - payload: Parsed JSON dict, or None if parsing failed
            - span_end: End position of the span
            - search_position: Next search position (or None if should continue)
            If payload is None, search_position is the position to advance to.
    """
    # Clean and validate JSON string
    # 1. Remove leading/trailing whitespace and newlines (including \n, \r, \t, spaces)
    json_str = json_str.strip()

    # Log the raw JSON for debugging
    logger.debug(f"Raw JSON string after strip: {repr(json_str)[:200]}")

    # 2. If JSON is empty or too short, skip
    if not json_str or len(json_str) < MIN_JSON_LENGTH:
        logger.warning(f"JSON string too short or empty: {repr(json_str)}")
        return None, None, _advance_search_position(end_match, start, start_tag_length)

    # 3. Handle common JSON formatting issues
    # Some models output JSON with leading/trailing non-JSON characters
    # Find the first { and last } to extract the JSON object
    first_brace = json_str.find('{')
    last_brace = json_str.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        # If braces are not at the beginning/end, extract the JSON object
        if first_brace > 0 or last_brace < len(json_str) - 1:
            original_len = len(json_str)
            json_str = json_str[first_brace:last_brace + 1]
            logger.debug(f"Extracted JSON object from text (reduced {original_len} to {len(json_str)} chars): {json_str[:100]}...")

    # 4. Ensure the string starts with { and ends with }
    if not json_str.startswith('{') or not json_str.endswith('}'):
        logger.warning(f"JSON string doesn't start with {{ and end with }}: {repr(json_str)[:200]}")
        # Try to add missing braces
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        logger.debug(f"Fixed JSON string: {repr(json_str)[:200]}")

    # Try to parse the JSON
    try:
        payload = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON for tool call: {json_str[:100]}...")
        return payload, (end_match.end() if end_match else text_length), None
    except json.JSONDecodeError as e:
        # Try to recover: find the last complete JSON object
        # This handles cases where the JSON is incomplete or has syntax errors
        logger.warning(f"JSON decode error in tool call block: {e}, JSON (raw): {repr(json_str)[:200]}")

        # Enhanced recovery: try to fix common JSON issues
        cleaned_json = fix_json_issues(json_str)

        if cleaned_json:
            try:
                payload = json.loads(cleaned_json)
                logger.info(f"Recovered JSON after fixing issues: {cleaned_json[:100]}...")
                return payload, (end_match.end() if end_match else text_length), None
            except json.JSONDecodeError:
                logger.warning(f"Could not recover JSON even after fixing")
                return None, None, _advance_search_position(end_match, start, start_tag_length)
        else:
            # No valid JSON found
            return None, None, _advance_search_position(end_match, start, start_tag_length)


def _validate_tool_call_payload(payload) -> tuple:
    """
    Validate a parsed tool call payload.

    Args:
        payload: Parsed JSON object

    Returns:
        tuple: (is_valid, name, input_data) where:
            - is_valid: True if payload is valid
            - name: Tool name (or None if invalid)
            - input_data: Tool input parameters (or None if invalid)
    """
    if not isinstance(payload, dict):
        logger.warning(f"Tool call payload is not a dict: {type(payload)}")
        return False, None, None

    name = payload.get("name")
    input_data = payload.get("input", {})

    if not isinstance(name, str) or not name:
        logger.warning(f"Tool call missing or invalid 'name' field: {name}")
        return False, None, None

    if not isinstance(input_data, dict):
        logger.warning(f"Tool call 'input' field is not a dict: {type(input_data)}")
        return False, None, None

    return True, name, input_data


def detect_all_tool_call_blocks(text: str) -> List[dict]:
    """
    Detect ALL <tool_call>JSON</tool_call> blocks in text and parse them.

    Expected JSON format:
       <tool_call>
      {"name":"Read","input":{"file_path":"..."}}
       </tool_call>

    Returns a list of tool_use dicts in Anthropic format, each with an internal "_span"
    (start_idx, end_idx) indicating the block span in the original text.

    The returned list is sorted by position in the text (earliest first).

    Includes a compatibility fix:
      - If name == "Read" and input has "path" but not "file_path",
        it will rewrite "path" -> "file_path".
    """
    # Validate input type
    if not isinstance(text, str):
        logger.warning(f"detect_all_tool_call_blocks expected str, got {type(text).__name__}")
        return []

    # Use precompiled regex patterns for performance
    start_pattern = _TOOL_CALL_START_PATTERN
    end_pattern = _TOOL_CALL_END_PATTERN

    tool_calls = []
    search_start = 0
    text_length = len(text)

    logger.debug(f"Starting tool call detection on text of length {text_length}")

    while search_start < text_length:
        # Find next start tag
        start_match = start_pattern.search(text, search_start)
        if not start_match:
            logger.debug(f"No more start tags found after position {search_start}")
            break

        start = start_match.start()
        start_tag_length = start_match.end() - start_match.start()

        # Find corresponding end tag
        end_match = end_pattern.search(text, start + start_tag_length)

        # Extract JSON content (with or without closing tag)
        if end_match:
            # Complete block: extract between tags
            json_str = text[start + start_tag_length:end_match.start()]
            span_end = end_match.end()
            logger.debug(f"Found complete tool call block at positions {start}-{span_end}")
        else:
            # Incomplete block: try to parse from start_tag to end of text
            json_str = text[start + start_tag_length:]
            span_end = text_length  # Span goes to end of text
            logger.warning(f"Incomplete tool call block at position {start}, no closing tag found")

        # Extract and parse JSON using helper
        payload, actual_span_end, search_pos = _extract_and_parse_json(
            json_str, start, end_match, start_tag_length, text_length
        )

        # If parsing failed, advance search position and continue
        if payload is None:
            search_start = search_pos
            continue

        # Update span_end from helper
        span_end = actual_span_end

        # Validate payload using helper
        is_valid, name, input_data = _validate_tool_call_payload(payload)
        if not is_valid:
            search_start = _advance_search_position(end_match, start, start_tag_length)
            continue

        # Compatibility fix: Read expects file_path, but some outputs may use "path"
        if name == "Read" and "file_path" not in input_data and "path" in input_data:
            input_data = dict(input_data)  # Make a copy to avoid mutating original
            input_data["file_path"] = input_data.pop("path")
            logger.debug(f"Applied compatibility fix for Read tool: path -> file_path")

        tool_calls.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": name,
            "input": input_data,
            "_span": (start, span_end),
        })

        logger.info(f"Detected tool call: {name} with {len(input_data)} parameters")

        # Move search position
        if end_match:
            search_start = end_match.end()
        else:
            search_start = span_end

    logger.info(f"Total tool calls detected: {len(tool_calls)}")
    return tool_calls
# ----------------------- Core non-streaming generation function -----------------------

async def generate_oci_non_stream(oci_messages, params, message_id, model_conf, requested_model, cohere_messages=None):
    """
    Generate non-streaming response from OCI GenAI and format it as Anthropic-compatible JSON.

    Args:
        oci_messages: Messages in Generic format (Message objects)
        params: Request parameters
        message_id: Message ID
        model_conf: Model configuration
        requested_model: Requested model name
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
        logger.info(f"Using COHERE API format for model {requested_model}")
    else:
        # Use Generic format
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.messages = oci_messages
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        logger.info(f"Using GENERIC API format for model {requested_model}")

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
        if is_cohere:
            setattr(chat_request, "stop_sequences", params["stop_sequences"])
        else:
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
            if is_cohere:
                # For Cohere, prepend to current message
                chat_request.message = thinking_instruction + "\n\n" + chat_request.message
            else:
                # For Generic, prepend to messages
                oci_messages.insert(0, oci.generative_ai_inference.models.Message(
                    role="SYSTEM",
                    content=[oci.generative_ai_inference.models.TextContent(text=thinking_instruction)]
                ))

    chat_request.is_stream = False

    # --- Tool Calling Support ---
    if "tools" in params:
        if is_cohere:
            # Use native Cohere tools
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
                logger.info(f"Using simplified tool instruction for {len(tools_list)} tools (non-streaming, tool_choice={tool_choice})")
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
                logger.info(f"Injected tool use instruction for {len(params['tools'])} tools (Generic format)")

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

            # Check for structured tool calls based on API format
            if is_cohere:
                tool_calls = extract_tool_calls_from_cohere_response(chat_response)
            else:
                tool_calls = extract_tool_calls_from_oci_response(chat_response)

            if tool_calls:
                # Found tool calls in the response
                content_blocks = tool_calls
                stop_reason = "tool_use"
                logger.info(f"Detected {len(tool_calls)} tool call(s) in response")
            else:
                # Extract text content
                if is_cohere:
                    # Cohere format: response has 'text' field
                    accumulated_text = chat_response.text if hasattr(chat_response, 'text') else ""
                else:
                    # Generic format: check various response structures
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

                # Also check if text contains a tool call block (for models that return tool calls as text)
                if accumulated_text:
                    tool_calls_in_text = detect_all_tool_call_blocks(accumulated_text)
                    if tool_calls_in_text:
                        # Remove tool call blocks from the accumulated text
                        clean_text = accumulated_text
                        # Sort spans in reverse order to remove from end to start (preserving indices)
                        spans = sorted([tc.pop("_span") for tc in tool_calls_in_text if "_span" in tc], reverse=True)
                        for start, end in spans:
                            clean_text = clean_text[:start] + clean_text[end:]

                        # Build content blocks: text (if any) + all tool calls
                        content_blocks = []
                        if clean_text.strip():
                            content_blocks.append({"type": "text", "text": clean_text})
                        content_blocks.extend(tool_calls_in_text)

                        stop_reason = "tool_use"
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
        estimated_input_tokens = sum(len(str(m.content)) // 4 for m in oci_messages) if oci_messages else 50

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

async def generate_oci_stream(oci_messages, params, message_id, model_conf, requested_model, cohere_messages=None):
    """
    Generate streaming response from OCI GenAI and format it as Anthropic-compatible SSE.

    Args:
        oci_messages: Messages in Generic format (Message objects)
        params: Request parameters
        message_id: Message ID
        model_conf: Model configuration
        requested_model: Requested model name
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
        logger.info(f"Using COHERE API format for streaming (model: {requested_model})")
    else:
        # Use Generic format
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.messages = oci_messages
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        logger.info(f"Using GENERIC API format for streaming (model: {requested_model})")

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
        if is_cohere:
            setattr(chat_request, "stop_sequences", params["stop_sequences"])
        else:
            setattr(chat_request, "stop", params["stop_sequences"])
        logger.info(f"Set stop_sequences: {params['stop_sequences']}")

    # 6. Handle Thinking Mode (Extended Thinking)
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
                # For Cohere, prepend to current message
                chat_request.message = thinking_instruction + "\n\n" + chat_request.message
            else:
                # For Generic, prepend to messages
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
    chat_detail.compartment_id = COMPARTMENT_ID
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
        stop_reason = "end_turn"
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
                    logger.info("OCI_STREAM_EVENT %s", raw[:384])
                    sample_left -= 1

                data = json.loads(raw)

                # Handle both Generic and Cohere streaming formats
                if is_cohere:
                    # Cohere streaming format may have different structure
                    text_chunk = data.get("text", "")
                    if text_chunk:
                        accumulated_text += text_chunk
                        if not buffer_text_for_tool_detection:
                            yield f"event: content_block_delta\ndata: {json.dumps({
                                'type': 'content_block_delta',
                                'index': current_block_index,
                                'delta': {'type': 'text_delta', 'text': text_chunk}
                            }, ensure_ascii=False)}\n\n"

                    # Check for finish reason
                    finish_reason = data.get("finishReason")
                    if finish_reason:
                        if finish_reason == "COMPLETE":
                            stop_reason = "end_turn"
                        elif finish_reason == "MAX_TOKENS":
                            stop_reason = "max_tokens"
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
                                    continue

                    finish_reason = data.get("finishReason")
                    if finish_reason:
                        if finish_reason == "stop":
                            stop_reason = "end_turn"
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
            # If we buffered text for tool detection, process it now
            if buffer_text_for_tool_detection:
                tool_calls_in_text = detect_all_tool_call_blocks(accumulated_text)
                logger.info(f"Tool detection result: found {len(tool_calls_in_text)} tool calls")
                if tool_calls_in_text:
                    # Remove all tool call blocks from the accumulated text
                    clean_text = accumulated_text
                    # Sort spans in reverse order to remove from end to start (preserving indices)
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
                        # No clean text, close the initial text block
                        yield f"event: content_block_stop\ndata: {json.dumps({
                            'type': 'content_block_stop',
                            'index': current_block_index
                        })}\n\n"
                        next_index = current_block_index + 1

                    # Yield all tool call blocks
                    for tool_call in tool_calls_in_text:
                        tool_call.pop("_span", None)  # Remove internal span marker
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

                    stop_reason = "tool_use"
                    tool_call_detected = True
                    logger.info(f"Detected {len(tool_calls_in_text)} tool call block(s) in buffered text, sent clean text + tool calls")
                else:
                    # No tool calls found, send all accumulated text
                    yield f"event: content_block_delta\ndata: {json.dumps({
                        'type': 'content_block_delta',
                        'index': current_block_index,
                        'delta': {'type': 'text_delta', 'text': accumulated_text}
                    }, ensure_ascii=False)}\n\n"

                    yield f"event: content_block_stop\ndata: {json.dumps({
                        'type': 'content_block_stop',
                        'index': current_block_index
                    })}\n\n"
                    logger.info("No tool calls detected in buffered text, sent all accumulated text")
            else:
                # Not buffering for tool detection (Cohere format or no tools)
                # Just close the text block
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

        tools = body.get("tools") or []
        tool_names = [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
        logger.info("REQ stream=%s tools=%d tool_choice=%s", bool(body.get("stream")), len(tool_names),
                    body.get("tool_choice"))
        if tool_names:
            logger.info("REQ tool_names=%s", tool_names[:20])

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
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"):
                        logger.info("REQ msg[%d] has %s", idx, b.get("type"))
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

        # Determine API format and prepare messages accordingly
        api_format = selected_model_conf.get("api_format", "generic").lower()
        is_cohere = api_format == "cohere"

        cohere_messages = None
        if is_cohere:
            # Convert messages to Cohere format
            system_prompt = body.get("system")
            messages = body.get("messages", [])
            current_message, chat_history, system_message = convert_anthropic_messages_to_cohere(messages, system_prompt)
            cohere_messages = (current_message, chat_history, system_message)
            logger.info(f"Converted to Cohere format: current_message_len={len(current_message)}, chat_history_len={len(chat_history)}")

        if body.get("stream", False):
            return StreamingResponse(
                generate_oci_stream(oci_msgs, body, message_id, selected_model_conf, req_model, cohere_messages),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming logic
            return await generate_oci_non_stream(oci_msgs, body, message_id, selected_model_conf, req_model, cohere_messages)

    return {"detail": "Not Found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")