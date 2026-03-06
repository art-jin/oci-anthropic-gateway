"""Request payload validation helpers."""

from __future__ import annotations

from typing import Any, Optional, Set, Iterable

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
        if isinstance(content, list):
            err = _validate_content_blocks(content, i)
            if err:
                return err

    return None


def _validate_content_blocks(blocks: list, message_index: int) -> Optional[str]:
    for bi, block in enumerate(blocks):
        if not isinstance(block, dict):
            return f"'messages[{message_index}].content[{bi}]' must be an object"
        block_type = str(block.get("type", "")).strip().lower()
        if not block_type:
            return f"'messages[{message_index}].content[{bi}].type' is required"

        if block_type == "text":
            if not isinstance(block.get("text", ""), str):
                return f"'messages[{message_index}].content[{bi}].text' must be a string"
            continue

        if block_type in {"image", "video", "audio"}:
            err = _validate_media_block(block, block_type, message_index, bi)
            if err:
                return err
            continue

        # Keep backward compatibility: don't reject unknown block types here.
    return None


def _validate_media_block(block: dict, block_type: str, message_index: int, block_index: int) -> Optional[str]:
    source = block.get("source")
    if not isinstance(source, dict):
        return f"'messages[{message_index}].content[{block_index}].source' must be an object"

    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in {"base64"}:
        return (
            f"'messages[{message_index}].content[{block_index}].source.type' "
            f"must be 'base64'"
        )

    if source_type == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data.strip():
            return (
                f"'messages[{message_index}].content[{block_index}].source.data' "
                f"is required for base64 {block_type}"
            )
        media_type = source.get("media_type")
        if not isinstance(media_type, str) or "/" not in media_type:
            return (
                f"'messages[{message_index}].content[{block_index}].source.media_type' "
                f"is required for base64 {block_type}"
            )
        media_prefix = media_type.split("/", 1)[0].lower()
        if block_type == "image" and media_prefix != "image":
            return (
                f"'messages[{message_index}].content[{block_index}]' media_type must start with 'image/'"
            )
        if block_type == "video" and media_prefix != "video":
            return (
                f"'messages[{message_index}].content[{block_index}]' media_type must start with 'video/'"
            )
        if block_type == "audio" and media_prefix != "audio":
            return (
                f"'messages[{message_index}].content[{block_index}]' media_type must start with 'audio/'"
            )
        return None

    return None


def collect_requested_modalities(messages: Any, system: Any = None) -> Set[str]:
    """Collect request modalities used by messages."""
    requested: Set[str] = set()
    if not isinstance(messages, list):
        return {"text"}

    for message in messages:
        if not isinstance(message, dict):
            requested.add("text")
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            requested.add("text")
            continue
        if not isinstance(content, list):
            requested.add("text")
            continue
        for block in content:
            if not isinstance(block, dict):
                requested.add("text")
                continue
            t = str(block.get("type", "")).strip().lower()
            if t == "image":
                requested.add("images")
            elif t == "video":
                requested.add("video")
            elif t == "audio":
                requested.add("audio")
            else:
                requested.add("text")

    if not requested:
        requested.add("text")

    # System prompt currently supports text blocks only, but include it for future-proof gating.
    if isinstance(system, str):
        requested.add("text")
    elif isinstance(system, list):
        for block in system:
            if not isinstance(block, dict):
                requested.add("text")
                continue
            t = str(block.get("type", "")).strip().lower()
            if t == "image":
                requested.add("images")
            elif t == "video":
                requested.add("video")
            elif t == "audio":
                requested.add("audio")
            else:
                requested.add("text")
    return requested


def validate_system_payload(system: Any) -> Optional[str]:
    """Validate system prompt payload shape."""
    if system is None or isinstance(system, str):
        return None
    if not isinstance(system, list):
        return "'system' must be a string or an array of content blocks"
    for i, block in enumerate(system):
        if not isinstance(block, dict):
            return f"'system[{i}]' must be an object"
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "text":
            if not isinstance(block.get("text", ""), str):
                return f"'system[{i}].text' must be a string"
            continue
        return f"'system[{i}].type' only supports 'text' currently"
    return None


def validate_model_modalities(
    requested_modalities: Iterable[str],
    model_types: Iterable[str],
    *,
    is_cohere: bool = False,
) -> Optional[str]:
    requested = {str(m).strip().lower() for m in requested_modalities if str(m).strip()}
    supported = {str(m).strip().lower() for m in model_types if str(m).strip()}
    if not requested:
        requested = {"text"}
    if not supported:
        supported = {"text"}

    non_text = requested - {"text"}
    if is_cohere and non_text:
        return "Current Cohere route supports text only; remove image/video/audio blocks or use a Generic multimodal model"

    missing = sorted(requested - supported)
    if missing:
        return (
            f"Requested modalities {missing} are not supported by this model; "
            f"model_types={sorted(supported)}"
        )

    # Gateway does not convert video/audio payloads yet.
    if requested.intersection({"video", "audio"}):
        return "Gateway video/audio conversion is not implemented yet; currently only text and image are supported"

    return None
