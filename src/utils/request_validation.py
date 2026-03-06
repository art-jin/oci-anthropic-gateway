"""Request payload validation helpers."""

from __future__ import annotations

from typing import Any, Optional

from .constants import DEFAULT_MAX_TOKENS


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_count_tokens_payload(body: Any, *, max_messages: int = 200) -> Optional[str]:
    if not isinstance(body, dict):
        return "Request body must be a JSON object"

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return "'messages' must be an array"
    if len(messages) > max_messages:
        return f"'messages' exceeds limit ({max_messages})"
    return None


def validate_messages_payload(body: Any, *, max_messages: int = 200) -> Optional[str]:
    if not isinstance(body, dict):
        return "Request body must be a JSON object"

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return "'messages' must be an array"
    if len(messages) > max_messages:
        return f"'messages' exceeds limit ({max_messages})"

    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        if not isinstance(max_tokens, int):
            return "'max_tokens' must be an integer"
        if max_tokens < 1 or max_tokens > DEFAULT_MAX_TOKENS:
            return f"'max_tokens' must be between 1 and {DEFAULT_MAX_TOKENS}"

    temperature = body.get("temperature")
    if temperature is not None:
        if not _is_number(temperature):
            return "'temperature' must be a number"
        if float(temperature) < 0.0 or float(temperature) > 2.0:
            return "'temperature' must be between 0.0 and 2.0"

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            return f"'messages[{i}]' must be an object"
        if "role" not in message:
            return f"'messages[{i}].role' is required"
        if "content" not in message:
            return f"'messages[{i}].content' is required"
        content = message.get("content")
        if not isinstance(content, (str, list)):
            return f"'messages[{i}].content' must be a string or an array"

    return None
