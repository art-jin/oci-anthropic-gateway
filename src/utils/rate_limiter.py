"""Simple in-memory sliding-window rate limiter."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_sec: int):
        self.limit = int(limit)
        self.window_sec = int(window_sec)
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            bucket = self._hits[key]
            cutoff = now - self.window_sec
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = max(1, int(bucket[0] + self.window_sec - now))
                return False, retry_after

            bucket.append(now)
            return True, 0
