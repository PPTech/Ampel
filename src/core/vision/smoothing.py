"""
Version: 0.9.9
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque


class TemporalBuffer:
    """Anti-flicker state filter using a sliding vote window."""

    def __init__(self, window_size: int = 5, persist_threshold: int = 4) -> None:
        self.window_size = max(3, window_size)
        self.persist_threshold = max(2, persist_threshold)
        self._window: Deque[str] = deque(maxlen=self.window_size)
        self._stable_state = "UNKNOWN"

    def push(self, state: str) -> str:
        normalized = (state or "UNKNOWN").upper()
        self._window.append(normalized)
        counts = Counter(self._window)
        candidate, votes = counts.most_common(1)[0]

        if candidate != self._stable_state and votes >= self.persist_threshold:
            self._stable_state = candidate
        elif self._stable_state == "UNKNOWN" and votes >= self.persist_threshold:
            self._stable_state = candidate

        return self._stable_state

    @property
    def stable_state(self) -> str:
        return self._stable_state
