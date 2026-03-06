"""SQLite repository for debug dump indexing and queries."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class DebugRepository:
    """Persist and query debug dump metadata."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_parent = Path(db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  tenant_id TEXT,
                  user_id TEXT,
                  first_seen TEXT NOT NULL,
                  last_seen TEXT NOT NULL,
                  message_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                  message_id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL,
                  request_id TEXT,
                  ts TEXT NOT NULL,
                  model TEXT,
                  stream INTEGER,
                  api_format TEXT,
                  stop_reason TEXT,
                  tools_count INTEGER,
                  has_tool_call INTEGER,
                  dump_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS dumps (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message_id TEXT NOT NULL,
                  session_id TEXT,
                  kind TEXT NOT NULL,
                  ts TEXT NOT NULL,
                  file_path TEXT NOT NULL UNIQUE,
                  payload_size INTEGER,
                  has_truncation INTEGER
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  message_id TEXT NOT NULL,
                  session_id TEXT,
                  ts TEXT NOT NULL,
                  tool_name TEXT NOT NULL,
                  source_kind TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session_id, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_dumps_kind_ts ON dumps(kind, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_dumps_session_ts ON dumps(session_id, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_ts ON tool_calls(tool_name, ts DESC);
                """
            )

    @staticmethod
    def _looks_like_envelope(obj: Dict[str, Any]) -> bool:
        return isinstance(obj, dict) and "kind" in obj and "payload" in obj

    @staticmethod
    def _has_truncation(value: Any) -> bool:
        if isinstance(value, dict):
            if value.get("truncated") is True:
                return True
            return any(DebugRepository._has_truncation(v) for v in value.values())
        if isinstance(value, list):
            return any(DebugRepository._has_truncation(v) for v in value)
        return False

    @staticmethod
    def _safe_text_ts(path: str) -> str:
        try:
            mtime = os.path.getmtime(path)
            return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()

    def ingest_dump_file(self, path: str) -> bool:
        """Ingest a single dump file. Returns True when inserted, False when skipped."""
        result = self.ingest_dump_file_with_event(path)
        return result is not None

    def ingest_dump_file_with_event(self, path: str) -> Optional[Dict[str, Any]]:
        """Ingest a single dump file and return event data for SSE broadcast.

        Returns None if skipped or failed.
        Returns dict with session_id and event if successful.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM dumps WHERE file_path = ?", (path,)).fetchone()
            if row:
                return None

            raw = Path(path).read_text(encoding="utf-8")
            parsed = json.loads(raw)
            envelope = parsed if self._looks_like_envelope(parsed) else {
                "schema_version": "1.0",
                "timestamp": self._safe_text_ts(path),
                "message_id": parsed.get("message_id") if isinstance(parsed, dict) else None,
                "request_id": None,
                "session_id": None,
                "user_id": None,
                "tenant_id": None,
                "kind": self._kind_from_filename(path),
                "payload": parsed,
            }

            message_id = envelope.get("message_id") or self._message_id_from_filename(path)
            if not message_id:
                return None

            ts = envelope.get("timestamp") or self._safe_text_ts(path)
            kind = envelope.get("kind") or self._kind_from_filename(path) or "unknown"
            payload = envelope.get("payload")
            session_id = envelope.get("session_id") or "unknown"
            request_id = envelope.get("request_id")
            user_id = envelope.get("user_id")
            tenant_id = envelope.get("tenant_id")
            payload_size = len(raw.encode("utf-8", errors="replace"))
            has_truncation = 1 if self._has_truncation(payload) else 0

            cursor = conn.execute(
                """
                INSERT INTO dumps(message_id, session_id, kind, ts, file_path, payload_size, has_truncation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, kind, ts, path, payload_size, has_truncation),
            )
            dump_id = cursor.lastrowid

            model, stream, api_format, stop_reason, tools_count, has_tool_call = self._extract_message_summary(kind, payload)

            conn.execute(
                """
                INSERT INTO messages(message_id, session_id, request_id, ts, model, stream, api_format, stop_reason, tools_count, has_tool_call, dump_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(message_id) DO UPDATE SET
                  session_id = COALESCE(NULLIF(excluded.session_id, 'unknown'), messages.session_id),
                  request_id = COALESCE(excluded.request_id, messages.request_id),
                  model = COALESCE(excluded.model, messages.model),
                  stream = COALESCE(excluded.stream, messages.stream),
                  api_format = COALESCE(excluded.api_format, messages.api_format),
                  stop_reason = COALESCE(excluded.stop_reason, messages.stop_reason),
                  tools_count = COALESCE(excluded.tools_count, messages.tools_count),
                  has_tool_call = COALESCE(excluded.has_tool_call, messages.has_tool_call),
                  dump_count = messages.dump_count + 1,
                  ts = CASE WHEN messages.ts < excluded.ts THEN excluded.ts ELSE messages.ts END
                """,
                (message_id, session_id, request_id, ts, model, stream, api_format, stop_reason, tools_count, has_tool_call),
            )

            conn.execute(
                """
                INSERT INTO sessions(session_id, tenant_id, user_id, first_seen, last_seen, message_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(session_id) DO UPDATE SET
                  tenant_id = COALESCE(excluded.tenant_id, sessions.tenant_id),
                  user_id = COALESCE(excluded.user_id, sessions.user_id),
                  first_seen = CASE WHEN sessions.first_seen > excluded.first_seen THEN excluded.first_seen ELSE sessions.first_seen END,
                  last_seen = CASE WHEN sessions.last_seen < excluded.last_seen THEN excluded.last_seen ELSE sessions.last_seen END,
                  message_count = (SELECT COUNT(*) FROM messages WHERE messages.session_id = sessions.session_id)
                """,
                (session_id, tenant_id, user_id, ts, ts),
            )

            if kind.endswith("tool_detection_primary") and isinstance(payload, dict):
                detected = payload.get("detected") or []
                if isinstance(detected, list):
                    for item in detected:
                        if not isinstance(item, dict):
                            continue
                        tool_name = item.get("name")
                        if not tool_name:
                            continue
                        conn.execute(
                            "INSERT INTO tool_calls(message_id, session_id, ts, tool_name, source_kind) VALUES (?, ?, ?, ?, ?)",
                            (message_id, session_id, ts, str(tool_name), kind),
                        )

            conn.commit()

        # Build event for SSE broadcast
        event = {
            "id": dump_id,
            "dump_id": dump_id,
            "message_id": message_id,
            "session_id": session_id,
            "lane": self._map_kind_to_lane(kind),
            "target_lane": self._map_kind_to_target_lane(kind),
            "kind": kind,
            "ts": ts,
            "label": self._make_event_label(kind, payload),
            "summary": self._make_event_summary(kind, payload),
            "payload_size": payload_size,
            "has_truncation": bool(has_truncation),
        }

        return {
            "session_id": session_id,
            "event": event,
        }

    @staticmethod
    def _message_id_from_filename(path: str) -> Optional[str]:
        name = Path(path).name
        parts = name.split("_")
        for p in parts:
            if p.startswith("msg_"):
                return p
        return None

    @staticmethod
    def _kind_from_filename(path: str) -> Optional[str]:
        name = Path(path).name
        if not name.endswith(".json"):
            return None
        parts = name[:-5].split("_")
        if len(parts) < 3:
            return None
        return "_".join(parts[2:])

    @staticmethod
    def _extract_message_summary(kind: str, payload: Any) -> tuple:
        model = None
        stream = None
        api_format = None
        stop_reason = None
        tools_count = None
        has_tool_call = None

        if isinstance(payload, dict):
            if kind == "request_summary":
                model = payload.get("requested_model")
                stream = 1 if payload.get("stream") else 0
                api_format = payload.get("api_format")
                tools_count = payload.get("tools_count")
            elif kind == "final_response_summary":
                stop_reason = payload.get("stop_reason")
                c_types = payload.get("content_types") or []
                if isinstance(c_types, list):
                    has_tool_call = 1 if "tool_use" in c_types else 0
            elif kind in ("tool_detection_primary", "stream_tool_detection_primary"):
                detected = payload.get("detected") or []
                if isinstance(detected, list):
                    has_tool_call = 1 if len(detected) > 0 else 0
            elif kind in ("raw_text", "stream_accumulated_text"):
                has_start = bool(payload.get("has_tool_call_start"))
                has_end = bool(payload.get("has_tool_call_end"))
                has_tool_call = 1 if has_start and has_end else 0

        return model, stream, api_format, stop_reason, tools_count, has_tool_call

    def list_sessions(self, page: int, size: int, q: str = "") -> Dict[str, Any]:
        offset = max(0, (page - 1) * size)
        query = "SELECT session_id, tenant_id, user_id, first_seen, last_seen, message_count FROM sessions"
        args: List[Any] = []
        if q:
            query += " WHERE session_id LIKE ? OR tenant_id LIKE ? OR user_id LIKE ?"
            pattern = f"%{q}%"
            args.extend([pattern, pattern, pattern])
        total = self._scalar_count(query.replace("SELECT session_id, tenant_id, user_id, first_seen, last_seen, message_count", "SELECT COUNT(*)"), args)
        query += " ORDER BY last_seen DESC LIMIT ? OFFSET ?"
        args.extend([size, offset])
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        items = [dict(r) for r in rows]
        return {"items": items, "page": page, "size": size, "total": total}

    def list_session_messages(self, session_id: str, page: int, size: int) -> Dict[str, Any]:
        offset = max(0, (page - 1) * size)
        total = self._scalar_count("SELECT COUNT(*) FROM messages WHERE session_id = ?", [session_id])
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, session_id, request_id, ts, model, stream, api_format, stop_reason, tools_count, has_tool_call, dump_count
                FROM messages
                WHERE session_id = ?
                ORDER BY ts DESC
                LIMIT ? OFFSET ?
                """,
                (session_id, size, offset),
            ).fetchall()
        return {"items": [dict(r) for r in rows], "page": page, "size": size, "total": total}

    def get_message_detail(self, message_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            msg = conn.execute(
                """
                SELECT message_id, session_id, request_id, ts, model, stream, api_format, stop_reason, tools_count, has_tool_call, dump_count
                FROM messages WHERE message_id = ?
                """,
                (message_id,),
            ).fetchone()
            if not msg:
                return None

            dumps = conn.execute(
                """
                SELECT id, kind, ts, file_path, payload_size, has_truncation
                FROM dumps WHERE message_id = ?
                ORDER BY ts ASC
                """,
                (message_id,),
            ).fetchall()

        detail = {
            "message": dict(msg),
            "dumps": [dict(d) for d in dumps],
            "kind_payloads": self._load_kind_payloads(dumps),
        }
        return detail

    def get_dump_raw(self, dump_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT id, file_path FROM dumps WHERE id = ?", (dump_id,)).fetchone()
        if not row:
            return None
        raw = Path(row["file_path"]).read_text(encoding="utf-8")
        return json.loads(raw)

    def _load_kind_payloads(self, dump_rows: List[sqlite3.Row]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in dump_rows:
            kind = r["kind"]
            path = r["file_path"]
            try:
                parsed = json.loads(Path(path).read_text(encoding="utf-8"))
                payload = parsed.get("payload") if self._looks_like_envelope(parsed) else parsed
                out[kind] = payload
            except Exception:
                out[kind] = {"_error": "failed_to_read_dump"}
        return out

    def _scalar_count(self, query: str, args: List[Any]) -> int:
        with self._connect() as conn:
            row = conn.execute(query, args).fetchone()
        if not row:
            return 0
        return int(row[0])

    # ==================== Timeline API for Swimlane ====================

    @staticmethod
    def _map_kind_to_lane(kind: str) -> str:
        """Map dump kind to swimlane (client/gateway/oci)."""
        if kind == "request_summary":
            return "client"
        elif kind.startswith("oci_request") or kind.startswith("final_response"):
            return "gateway"
        elif kind.startswith("oci_response") or kind.startswith("stream_accumulated"):
            return "oci"
        elif kind.startswith("tool_detection") or kind == "raw_text":
            return "gateway"
        else:
            return "gateway"

    @staticmethod
    def _map_kind_to_target_lane(kind: str) -> Optional[str]:
        """Map dump kind to target lane for connector lines."""
        if kind == "request_summary":
            return "gateway"
        elif kind.startswith("oci_request"):
            return "oci"
        elif kind.startswith("oci_response") or kind.startswith("stream_accumulated"):
            return "gateway"
        elif kind.startswith("final_response"):
            return "client"
        return None

    @staticmethod
    def _make_event_label(kind: str, payload: Any) -> str:
        """Create short label for event node."""
        labels = {
            "request_summary": "Request",
            "oci_request": "OCI Req",
            "oci_response": "OCI Resp",
            "oci_response_error": "OCI Error",
            "stream_accumulated_text": "Stream",
            "tool_detection_primary": "Tool Detect",
            "stream_tool_detection_primary": "Tool Detect",
            "final_response_summary": "Response",
            "raw_text": "Raw Text",
        }
        if kind in labels:
            return labels[kind]
        # Fallback: use kind without underscores
        return kind.replace("_", " ").title()[:15]

    @staticmethod
    def _make_event_summary(kind: str, payload: Any) -> str:
        """Create one-line summary for tooltip/detail."""
        if not isinstance(payload, dict):
            return kind

        if kind == "request_summary":
            model = payload.get("requested_model", "?")
            tools = payload.get("tools_count", 0)
            stream = "stream" if payload.get("stream") else "non-stream"
            return f"model={model}, tools={tools}, {stream}"
        elif kind in ("tool_detection_primary", "stream_tool_detection_primary"):
            detected = payload.get("detected") or []
            count = len(detected) if isinstance(detected, list) else 0
            return f"detected {count} tool calls"
        elif kind == "stream_accumulated_text":
            text = payload.get("accumulated_text", "")
            length = len(text) if text else 0
            has_tool = payload.get("has_tool_call_start", False)
            return f"text_len={length}, has_tool_start={has_tool}"
        elif kind == "final_response_summary":
            stop = payload.get("stop_reason", "?")
            return f"stop_reason={stop}"
        elif kind == "oci_response_error":
            et = payload.get("error_type", "error")
            return f"oci_error={et}"
        return kind

    def get_session_timeline(self, session_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get timeline events for swimlane visualization."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, message_id, session_id, kind, ts, file_path, payload_size, has_truncation
                FROM dumps
                WHERE session_id = ?
                ORDER BY ts ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        events = []
        for row in rows:
            kind = row["kind"]
            file_path = row["file_path"]

            # Load payload for label/summary
            try:
                raw = Path(file_path).read_text(encoding="utf-8")
                parsed = json.loads(raw)
                payload = parsed.get("payload") if self._looks_like_envelope(parsed) else parsed
            except Exception:
                payload = {}

            event = {
                "id": row["id"],
                "dump_id": row["id"],
                "message_id": row["message_id"],
                "session_id": row["session_id"],
                "lane": self._map_kind_to_lane(kind),
                "target_lane": self._map_kind_to_target_lane(kind),
                "kind": kind,
                "ts": row["ts"],
                "label": self._make_event_label(kind, payload),
                "summary": self._make_event_summary(kind, payload),
                "payload_size": row["payload_size"],
                "has_truncation": bool(row["has_truncation"]),
            }
            events.append(event)

        return {
            "session_id": session_id,
            "lanes": ["client", "gateway", "oci"],
            "events": events,
            "total": len(events),
        }

    def get_all_timeline(self, limit: int = 500) -> Dict[str, Any]:
        """Get timeline events for ALL sessions (for global swimlane view)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, message_id, session_id, kind, ts, file_path, payload_size, has_truncation
                FROM dumps
                ORDER BY ts ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        events = []
        for row in rows:
            kind = row["kind"]
            file_path = row["file_path"]

            # Load payload for label/summary
            try:
                raw = Path(file_path).read_text(encoding="utf-8")
                parsed = json.loads(raw)
                payload = parsed.get("payload") if self._looks_like_envelope(parsed) else parsed
            except Exception:
                payload = {}

            event = {
                "id": row["id"],
                "dump_id": row["id"],
                "message_id": row["message_id"],
                "session_id": row["session_id"],
                "lane": self._map_kind_to_lane(kind),
                "target_lane": self._map_kind_to_target_lane(kind),
                "kind": kind,
                "ts": row["ts"],
                "label": self._make_event_label(kind, payload),
                "summary": self._make_event_summary(kind, payload),
                "payload_size": row["payload_size"],
                "has_truncation": bool(row["has_truncation"]),
            }
            events.append(event)

        return {
            "lanes": ["client", "gateway", "oci"],
            "events": events,
            "total": len(events),
        }
