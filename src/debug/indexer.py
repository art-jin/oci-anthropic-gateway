"""Background dump indexer for debug UI."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .repository import DebugRepository

logger = logging.getLogger("oci-gateway")


class DebugDumpIndexer:
    """Periodically scans debug dump directory and ingests new files."""

    def __init__(self, dump_dir: str, db_path: str, scan_interval_sec: int = 3):
        self.dump_dir = Path(dump_dir)
        self.repo = DebugRepository(db_path)
        self.scan_interval_sec = max(1, int(scan_interval_sec))
        self._event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def set_event_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Set callback for new events.

        Callback signature: callback(session_id: str, event: dict)
        The event dict contains timeline event data for SSE broadcast.
        """
        self._event_callback = callback

    def scan_once(self) -> int:
        if not self.dump_dir.exists():
            return 0

        inserted = 0
        for path in sorted(self.dump_dir.glob("*.json")):
            try:
                event = self.repo.ingest_dump_file_with_event(str(path))
                if event:
                    inserted += 1
                    if self._event_callback:
                        try:
                            self._event_callback(event["session_id"], event["event"])
                        except Exception as e:
                            logger.warning("Event callback failed: %s", e)
            except Exception as e:
                logger.warning("Debug index ingest failed: file=%s err=%s", path, e)
        return inserted

    async def run_forever(self) -> None:
        logger.info("Debug indexer started: dir=%s db=%s", self.dump_dir, self.repo.db_path)
        while True:
            try:
                inserted = await asyncio.to_thread(self.scan_once)
                if inserted > 0:
                    logger.debug("Debug indexer ingested %d new dump files", inserted)
            except asyncio.CancelledError:
                logger.info("Debug indexer stopped")
                raise
            except Exception as e:
                logger.warning("Debug indexer loop error: %s", e)

            await asyncio.sleep(self.scan_interval_sec)
