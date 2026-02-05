"""Token counting helper functions."""

import re
from typing import Optional, Union, List

from .constants import DEFAULT_MAX_TOKENS


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a heuristic approach since we don't have access to the exact tokenizer.
    - English: ~4 characters per token
    - Chinese: ~1.5 characters per token
    - Code: ~3-4 characters per token

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
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


def count_tokens_from_messages(
    messages: List[dict],
    system: Optional[Union[str, List[dict]]] = None
) -> int:
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
                    total_tokens += estimate_tokens(str(input_data))
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


def _extract_text_from_oci_message_content_item(item) -> str:
    # OCI SDK content items can be TextContent, dict-like, or other objects.
    # We keep this very defensive and heuristic.
    if item is None:
        return ""

    # Common: oci.generative_ai_inference.models.TextContent(text="...")
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text

    # Sometimes content may be dict-like: {"type": "TEXT", "text": "..."}
    if isinstance(item, dict):
        maybe_text = item.get("text")
        if isinstance(maybe_text, str):
            return maybe_text

    return ""


def estimate_tokens_from_oci_messages(oci_messages: list) -> int:
    """
    Estimate input tokens for OCI Generic ChatRequest messages.

    This is heuristic and intended for budget trimming to avoid OCI
    'context_length_exceeded' errors.

    Rules:
    - Text: estimate_tokens(text)
    - Images: fixed ~85 tokens (same heuristic as Anthropic message counting)
    - Unknown content: fall back to estimate_tokens(str(item))
    """
    if not oci_messages:
        return 0

    total = 0

    for msg in oci_messages:
        if msg is None:
            continue

        # Small constant for role/metadata overhead (heuristic)
        role = getattr(msg, "role", "") or ""
        total += estimate_tokens(str(role)) if role else 0

        content = getattr(msg, "content", None)

        # Generic format uses list content; but be defensive
        if content is None:
            continue

        if isinstance(content, list):
            for item in content:
                # Detect image content by class name (avoid importing SDK types here)
                cls_name = item.__class__.__name__ if item is not None else ""
                if cls_name == "ImageContent":
                    total += 85
                    continue

                extracted_text = _extract_text_from_oci_message_content_item(item)
                if extracted_text:
                    total += estimate_tokens(extracted_text)
                else:
                    total += estimate_tokens(str(item))
        else:
            # Unexpected single item
            total += estimate_tokens(str(content))

    return total
