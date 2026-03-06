"""Shared tool name/input normalization helpers."""

from __future__ import annotations

from typing import Any, Dict

# Compatibility with snake_case outputs from some models.
TOOL_NAME_ALIASES = {
    "web_search": "WebSearch",
    "websearch": "WebSearch",
    "read_file": "Read",
    "ask_user_question": "AskUserQuestion",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "multiedit": "MultiEdit",
    "multi_edit": "MultiEdit",
    "ls": "LS",
    "glob": "Glob",
    "bash": "Bash",
    "todowrite": "TodoWrite",
}


def normalize_tool_name(name: str) -> str:
    if not name:
        return name
    key = name.strip().replace("-", "_")
    return TOOL_NAME_ALIASES.get(key, TOOL_NAME_ALIASES.get(key.lower(), name))


def normalize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    # AskUserQuestion compatibility: accept {"question": "..."} -> {"questions": ["..."]}.
    if tool_name == "AskUserQuestion" and isinstance(tool_input, dict):
        if "questions" not in tool_input and "question" in tool_input:
            return {"questions": [tool_input["question"]]}
    return tool_input
