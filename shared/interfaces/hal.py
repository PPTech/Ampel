"""
Version: 0.9.7
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Detection:
    state: str
    confidence: float
    bbox: Optional[tuple[float, float, float, float]] = None
    lane_id: Optional[int] = None


class ICameraProvider(ABC):
    @abstractmethod
    def start_stream(self) -> None: ...

    @abstractmethod
    def get_frame(self) -> Any: ...


class ILocationProvider(ABC):
    @abstractmethod
    def get_anonymized_location(self) -> str: ...


class IInferenceEngine(ABC):
    @abstractmethod
    def detect(self, frame: Any) -> list[Detection]: ...


class IAlertSink(ABC):
    @abstractmethod
    def play_audio(self, alert_type: str) -> None: ...

    @abstractmethod
    def show_visual(self, alert_type: str) -> None: ...


class ISecureStorage(ABC):
    @abstractmethod
    def save_encrypted(self, key: str, value: str) -> None: ...

    @abstractmethod
    def load_encrypted(self, key: str) -> Optional[str]: ...
