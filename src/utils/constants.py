"""Constants for OCI Anthropic Gateway."""

import re

# --- Token Limits ---
DEFAULT_MAX_TOKENS = 131070  # Maximum tokens for generation (65535 * 2)

# --- Tool Call Detection ---
MIN_JSON_LENGTH = 10  # Minimum valid JSON length for tool calls

# Precompiled regex patterns for tool call detection (performance optimization)
_TOOL_CALL_START_PATTERN = re.compile(r'<TOOL_CALL\s*>', re.IGNORECASE)
_TOOL_CALL_END_PATTERN = re.compile(r'</TOOL_CALL\s*>', re.IGNORECASE)

# --- API Response Types ---
STOP_REASON_END_TURN = "end_turn"
STOP_REASON_MAX_TOKENS = "max_tokens"
STOP_REASON_TOOL_USE = "tool_use"
STOP_REASON_STOP_SEQUENCE = "stop_sequence"

# --- Content Block Types ---
CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_IMAGE = "image"
CONTENT_TYPE_TOOL_USE = "tool_use"
CONTENT_TYPE_TOOL_RESULT = "tool_result"

# --- Message Roles ---
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_TOOL = "tool"

# --- Tool Choice Strategies ---
TOOL_CHOICE_AUTO = "auto"
TOOL_CHOICE_REQUIRED = "required"
TOOL_CHOICE_NONE = "none"
TOOL_CHOICE_ANY = "any"
