"""Debug-only dump utilities.

This module is intentionally lightweight and safe:
- When debug is disabled, it becomes a no-op.
- Any dump write failure must not affect the request flow.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("oci-gateway")


@dataclass(frozen=True)
class DebugDumpConfig:
    enabled: bool
    dump_dir: str = "debug_dumps"
    max_bytes: int = 2_000_000


def _truncate_text(value: str, max_bytes: int) -> Dict[str, Any]:
    """Truncate a potentially large text to fit within max_bytes.

    Strategy: keep head/tail, and record original lengths.
    """
    if not isinstance(value, str):
        value = str(value)

    raw_bytes = value.encode("utf-8", errors="replace")
    if len(raw_bytes) <= max_bytes:
        return {
            "truncated": False,
            "original_length": len(value),
            "original_bytes": len(raw_bytes),
            "value": value,
        }

    # Keep 45% head and 45% tail with a marker in the middle.
    head_bytes_len = int(max_bytes * 0.45)
    tail_bytes_len = int(max_bytes * 0.45)

    head = raw_bytes[:head_bytes_len].decode("utf-8", errors="replace")
    tail = raw_bytes[-tail_bytes_len:].decode("utf-8", errors="replace")
    marker = "\n...<TRUNCATED>...\n"

    truncated_value = head + marker + tail
    truncated_bytes = truncated_value.encode("utf-8", errors="replace")
    if len(truncated_bytes) > max_bytes:
        # Safety fallback: hard cut
        truncated_value = raw_bytes[: max_bytes - 64].decode("utf-8", errors="replace") + "\n...<TRUNCATED>..."

    return {
        "truncated": True,
        "original_length": len(value),
        "original_bytes": len(raw_bytes),
        "value": truncated_value,
    }


def _truncate_payload(payload: Any, max_bytes: int) -> Any:
    """Truncate known large text fields in payload."""
    if not isinstance(payload, dict):
        return payload

    out = dict(payload)
    for key in ("raw_text", "accumulated_text", "response_content", "clean_text", "text_to_check"):
        if key in out and isinstance(out[key], str):
            out[key] = _truncate_text(out[key], max_bytes)
    return out


def write_debug_dump(
    config: DebugDumpConfig,
    message_id: str,
    kind: str,
    payload: Any,
) -> Optional[str]:
    """Write a debug dump JSON file.

    Returns the file path on success, otherwise None.
    """
    if not config or not config.enabled:
        return None

    try:
        dump_root = Path(config.dump_dir)
        dump_root.mkdir(parents=True, exist_ok=True)

        safe_payload = _truncate_payload(payload, config.max_bytes)
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}_{message_id}_{kind}.json"
        path = dump_root / fname

        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_payload, f, ensure_ascii=False, indent=2)

        return str(path)
    except Exception as e:
        logger.warning("Debug dump write failed: kind=%s message_id=%s err=%s", kind, message_id, e)
        return None
