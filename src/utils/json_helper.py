"""JSON parsing and tool call detection utilities."""

import json
import logging
import re
import uuid
from typing import Optional, List

from .constants import MIN_JSON_LENGTH, _TOOL_CALL_START_PATTERN, _TOOL_CALL_END_PATTERN

logger = logging.getLogger("oci-gateway")

# Tool name normalization (compatibility with snake_case outputs from some models)
TOOL_NAME_ALIASES = {
    "web_search": "WebSearch",
    "websearch": "WebSearch",
    "read_file": "Read",
    "ask_user_question": "AskUserQuestion",
}

def normalize_tool_name(name: str) -> str:
    if not name:
        return name
    k = name.strip().replace("-", "_")
    return TOOL_NAME_ALIASES.get(k, TOOL_NAME_ALIASES.get(k.lower(), name))

def normalize_tool_input(tool_name: str, tool_input: dict) -> dict:
    # AskUserQuestion compatibility: accept {"question": "..."} and convert to {"questions": ["..."]}
    if tool_name == "AskUserQuestion" and isinstance(tool_input, dict):
        if "questions" not in tool_input and "question" in tool_input:
            return {"questions": [tool_input["question"]]}
    return tool_input

def _advance_search_position(end_match, start: int, start_tag_length: int) -> int:
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
    - Newlines and formatting issues
    - Incomplete JSON objects

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

        # Fix 1: Remove control characters and excessive newlines
        json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
        # Normalize newlines within JSON (but preserve structure)
        json_str = re.sub(r'\n\s*\n', '\n', json_str)

        # Fix 2: Remove trailing commas (e.g., {"a": 1,} or {"a": 1, "b": 2,})
        # Be careful to only remove commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Fix 3: Replace single quotes with double quotes for strings
        # This is tricky - we need to be careful not to replace quotes inside strings
        # Simple approach: replace single-quoted property names and values
        # Match pattern: 'key': or 'value', but not inside double-quoted strings
        # Replace single quotes around property names (at start of object or after comma)
        json_str = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', json_str)
        # Replace single quotes for string values (after colon or comma)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        json_str = re.sub(r",\s*'([^']*)'", r', "\1"', json_str)

        # Fix 4: Ensure all property names are quoted
        # Match unquoted property names (word characters before colon)
        # Pattern: word followed by colon, but not preceded by quote or bracket
        # This is a simple heuristic - may not catch all cases
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)

        # Fix 5: Fix boolean/null values that might be quoted
        json_str = re.sub(r'"(true|false|null)"', r'\1', json_str)

        # Fix 6: Fix numbers that might be quoted (simple heuristic)
        # Match quoted numbers like "123" or "12.34" but not in the middle of text
        json_str = re.sub(r':\s*"(\d+(?:\.\d+)?)"', r': \1', json_str)

        # Fix 7: Handle incomplete JSON - try to close it
        # Count braces and brackets
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        if open_braces > 0:
            json_str += '}' * open_braces
            logger.debug(f"Added {open_braces} closing braces")
        
        if open_brackets > 0:
            json_str += ']' * open_brackets
            logger.debug(f"Added {open_brackets} closing brackets")

        # Fix 8: Remove duplicate commas
        json_str = re.sub(r',\s*,', ',', json_str)

        # Fix 9: Fix missing commas between fields (common mistake)
        # Pattern: "value" followed by new field without comma
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
        json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)

        # Try to parse the fixed JSON
        json.loads(json_str)

        # If we got here, the JSON is now valid
        if json_str != original:
            logger.debug(f"Fixed JSON issues: {original[:100]}... -> {json_str[:100]}...")

        return json_str

    except (json.JSONDecodeError, re.error) as e:
        # If still invalid, try more aggressive fixes
        logger.debug(f"Could not fix JSON with standard fixes: {e}")
        
        # Last resort: try to extract just the first complete JSON object
        try:
            first_brace = json_str.find('{')
            if first_brace != -1:
                brace_count = 0
                for i, char in enumerate(json_str[first_brace:], start=first_brace):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete object
                            extracted = json_str[first_brace:i+1]
                            json.loads(extracted)  # Validate
                            logger.info(f"Extracted complete JSON object from position {first_brace} to {i+1}")
                            return extracted
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None


def _extract_and_parse_json(
    json_str: str,
    start: int,
    end_match,
    start_tag_length: int,
    text_length: int
) -> tuple:
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
    Detect ALL <TOOL_CALL>JSON blocks in text and parse them.

    Expected JSON format:
        <TOOL_CALL>
      {"name":"Read","input":{"file_path":"..."}}
      </TOOL_CALL>

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

        # Normalize tool name and input for compatibility
        name = normalize_tool_name(name)
        input_data = normalize_tool_input(name, input_data)

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


def detect_tool_call_block(text: str) -> Optional[dict]:
    """
    Detect a single <TOOL_CALL>JSON block and parse it.

    Expected JSON format:
        <TOOL_CALL>
      {"name":"Read","input":{"file_path":"..."}}
        </TOOL_CALL>

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

    name = normalize_tool_name(name)
    input_data = normalize_tool_input(name, input_data)

    return {
        "type": "tool_use",
        "id": f"toolu_{uuid.uuid4().hex[:24]}",
        "name": name,
        "input": input_data,
        "_span": (start, end + len(end_tag)),
    }


def detect_natural_language_tool_calls(text: str, available_tools: List[str] = None) -> List[dict]:
    """
    Fallback mechanism: Try to detect tool calls from natural language when the model
    doesn't follow the exact <TOOL_CALL> format.
    
    This looks for patterns like:
    - "I'll use the [tool_name] tool with [parameters]"
    - "Calling [tool_name] with [parameters]"
    - "Let me [action verb] using [tool_name]"
    - JSON objects that look like tool calls but without tags
    
    Args:
        text: Text to analyze
        available_tools: Optional list of available tool names to match against
        
    Returns:
        List of detected tool_use blocks in Anthropic format
    """
    tool_calls = []
    
    if not text or not isinstance(text, str):
        return tool_calls
    
    # Pattern 1: Look for standalone JSON objects that resemble tool calls
    # Match JSON objects with "name" and "input" fields that aren't already in TOOL_CALL tags
    json_pattern = r'(?<!<TOOL_CALL>)\s*\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"input"\s*:\s*\{[^}]*\}[^{}]*\}'
    
    for match in re.finditer(json_pattern, text, re.DOTALL | re.IGNORECASE):
        try:
            json_str = match.group(0).strip()
            payload = json.loads(json_str)
            
            if isinstance(payload, dict):
                name = payload.get("name")
                input_data = payload.get("input", {})

                name = normalize_tool_name(name)
                input_data = normalize_tool_input(name, input_data)

                if name and isinstance(input_data, dict):
                    # If available_tools is provided, only accept known tools
                    if available_tools is None or name in available_tools:
                        tool_calls.append({
                            "type": "tool_use",
                            "id": f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": name,
                            "input": input_data,
                            "_span": (match.start(), match.end()),
                        })
                        logger.info(f"Detected natural language tool call (JSON pattern): {name}")
        except (json.JSONDecodeError, ValueError):
            continue
    
    # Pattern 2: Look for explicit tool usage language patterns
    # Examples: "I'll use the search tool", "Calling the get_weather function"
    if available_tools:
        for tool_name in available_tools:
            # Create pattern for this specific tool
            patterns = [
                rf"(?:use|using|call|calling|invoke|invoking)\s+(?:the\s+)?{re.escape(tool_name)}",
                rf"{re.escape(tool_name)}\s+(?:function|tool|method)",
                rf"I['']ll\s+{re.escape(tool_name)}",
            ]
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Found mention of tool usage - try to extract parameters
                    # Look for JSON-like structures nearby
                    match_pos = re.search(pattern, text, re.IGNORECASE)
                    if match_pos:
                        # Check surrounding context for parameters
                        context_start = max(0, match_pos.start() - 200)
                        context_end = min(len(text), match_pos.end() + 200)
                        context = text[context_start:context_end]
                        
                        # Try to find parameter JSON in the context
                        param_pattern = r'\{[^{}]+\}'
                        for param_match in re.finditer(param_pattern, context):
                            try:
                                params = json.loads(param_match.group(0))
                                if isinstance(params, dict):
                                    tool_calls.append({
                                        "type": "tool_use",
                                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                                        "name": tool_name,
                                        "input": params,
                                        "_span": (match_pos.start(), match_pos.end()),
                                    })
                                    logger.info(f"Detected natural language tool call (NL pattern): {tool_name}")
                                    break  # Only take the first valid parameter match
                            except (json.JSONDecodeError, ValueError):
                                continue
    
    # Remove duplicates (based on name and input)
    seen = set()
    unique_calls = []
    for call in tool_calls:
        key = (call["name"], json.dumps(call["input"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique_calls.append(call)
    
    if unique_calls:
        logger.info(f"Total natural language tool calls detected: {len(unique_calls)}")
    
    return unique_calls
