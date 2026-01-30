"""Content conversion utilities for OCI/Cohere format conversion."""

import json
import logging
import re
import uuid
from typing import Optional, Union, List
import oci

logger = logging.getLogger("oci-gateway")


def convert_content_to_oci(content: Union[str, List[dict]]) -> Union[str, List]:
    """
    Convert Anthropic content to OCI format.

    Handles:
    - Plain text strings
    - text content blocks
    - image content blocks (base64 or URL)
    - tool_result content blocks
    - tool_use content blocks

    Args:
        content: Anthropic format content (string or list of blocks)

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
            # Use a clear format that the model can understand
            tool_use_id = block.get("tool_use_id", "unknown")
            result = block.get("content", block.get("result", ""))
            is_error = block.get("is_error", False)
            
            # Extract the actual result content
            if isinstance(result, list):
                # Handle nested content blocks in tool_result
                result_text = "".join([r.get("text", "") if isinstance(r, dict) else str(r) for r in result])
            elif isinstance(result, dict):
                result_text = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                result_text = str(result)
            
            # Handle empty results - provide clear success message
            if not result_text or result_text.strip() == "":
                if is_error:
                    result_text = "Tool execution failed with no error details"
                else:
                    result_text = "Command executed successfully (no output)"
            
            # Format as a clear result block that models can understand
            if is_error:
                formatted_result = f"\n<TOOL_RESULT id='{tool_use_id}' status='error'>\n{result_text}\n</TOOL_RESULT>\n"
            else:
                formatted_result = f"\n<TOOL_RESULT id='{tool_use_id}' status='success'>\n{result_text}\n</TOOL_RESULT>\n"
            
            text_parts.append(formatted_result)
            logger.info(f"Converted tool_result for {tool_use_id}: {len(result_text)} chars, is_error={is_error}, preview: {result_text[:100]}")

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

    Args:
        chat_response: OCI chat response object

    Returns:
        List of tool_use blocks in Anthropic format, or None if no tool calls.
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

    Args:
        cohere_response: OCI Cohere chat response object

    Returns:
        List of tool_use blocks in Anthropic format, or None if no tool calls.
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


def convert_content_to_cohere_message(
    content: Union[str, List[dict]],
    role: str,
    tool_use_id_to_name: dict = None
) -> Union[
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
        tool_use_id_to_name: Optional dict mapping tool_use_id to tool_name for context

    Returns:
        CohereUserMessage, CohereChatBotMessage, or CohereToolMessage
    """
    role_upper = role.upper()

    # Initialize tool_use_id_to_name if not provided
    if tool_use_id_to_name is None:
        tool_use_id_to_name = {}

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
                tool_use_id = block.get("id", "")
                input_data = block.get("input", {})

                # Store the mapping for future tool_result references
                if tool_use_id and name:
                    tool_use_id_to_name[tool_use_id] = name

                tool_calls.append(oci.generative_ai_inference.models.CohereToolCall(
                    name=name,
                    parameters=input_data
                ))

            elif block_type == "tool_result":
                # Convert tool_result to CohereToolResult
                tool_use_id = block.get("tool_use_id", "")
                result = block.get("content", block.get("result", ""))

                # Find corresponding tool call name from context
                tool_name = tool_use_id_to_name.get(tool_use_id) if tool_use_id else None
                if not tool_name:
                    # Fallback to name field in block, or unknown
                    tool_name = block.get("name", "unknown")

                tool_call_ref = oci.generative_ai_inference.models.CohereToolCall(
                    name=tool_name,
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


def convert_anthropic_messages_to_cohere(
    messages: List[dict],
    system: Optional[str] = None
) -> tuple:
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

    # Maintain tool_use_id to tool_name mapping for context across messages
    tool_use_id_to_name = {}

    # Process all messages except the last one as chat history
    for msg in messages[:-1]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        cohere_msg = convert_content_to_cohere_message(content, role, tool_use_id_to_name)
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

    Args:
        text: Response text to parse

    Returns:
        Tool call dict in Anthropic format, or None if not found
    """
    # Pattern for: [TOOL_CALL] function_name: {"key": "value"}
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
