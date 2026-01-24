"""Utility modules for OCI Anthropic Gateway."""

from . import constants
from .token import estimate_tokens, count_tokens_from_messages
from .tools import (
    anthropic_to_oci_tools,
    anthropic_to_cohere_tools,
    convert_tool_choice,
    _build_tool_use_instruction,
)
from .cache import extract_cache_control, log_cache_info
from .json_helper import (
    _advance_search_position,
    _quote_property_name,
    _unquote_number,
    fix_json_issues,
    detect_all_tool_call_blocks,
    detect_tool_call_block,
)
from .content_converter import (
    convert_content_to_oci,
    extract_tool_calls_from_oci_response,
    extract_tool_calls_from_cohere_response,
    convert_content_to_cohere_message,
    convert_anthropic_messages_to_cohere,
)

__all__ = [
    # Constants
    "constants",
    # Token counting
    "estimate_tokens",
    "count_tokens_from_messages",
    # Tools
    "anthropic_to_oci_tools",
    "anthropic_to_cohere_tools",
    "convert_tool_choice",
    "_build_tool_use_instruction",
    # Cache
    "extract_cache_control",
    "log_cache_info",
    # JSON helpers
    "_advance_search_position",
    "_quote_property_name",
    "_unquote_number",
    "fix_json_issues",
    "detect_all_tool_call_blocks",
    "detect_tool_call_block",
    # Content converter
    "convert_content_to_oci",
    "extract_tool_calls_from_oci_response",
    "extract_tool_calls_from_cohere_response",
    "convert_content_to_cohere_message",
    "convert_anthropic_messages_to_cohere",
]
