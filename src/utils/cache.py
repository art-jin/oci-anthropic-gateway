"""Prompt caching helper functions."""

import logging
from typing import Optional, Union, List

logger = logging.getLogger("oci-gateway")


def extract_cache_control(blocks: Union[str, List[dict]]) -> dict:
    """
    Extract cache_control information from content blocks.

    Args:
        blocks: Content blocks (string or list of dicts)

    Returns:
        Dict with keys:
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


def log_cache_info(content_type: str, cache_info: dict, index: Optional[int] = None) -> None:
    """Log cache control information for debugging.

    Args:
        content_type: Type of content (e.g., "system", "message")
        cache_info: Cache info dict from extract_cache_control
        index: Optional index for array content
    """
    if cache_info["has_cache_control"]:
        location = f"{content_type}[{index}]" if index is not None else content_type
        logger.debug(f"Cache control detected in {location}: "
                   f"{cache_info['cached_blocks']} block(s), types: {cache_info['cache_types']}")
