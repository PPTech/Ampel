#!/usr/bin/env python3
"""
Version: 0.9.3
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


class FrameProvider(Protocol):
    def frames(self) -> Iterator[object]: ...


@dataclass(frozen=True)
class DebugConfig:
    mock_mode: bool
    video_path: Path


class CameraProvider:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index

    def frames(self) -> Iterator[object]:
        if cv2 is None:
            return iter(())
        cap = cv2.VideoCapture(self.camera_index)
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


class VideoFileProvider:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path

    def frames(self) -> Iterator[object]:
        if cv2 is None:
            return iter(())
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop for repeatable QA
                    continue
                yield frame
        finally:
            cap.release()


class CameraSourceInjector:
    """Swap camera provider for debug playback (MOCK_MODE=True)."""

    def __init__(self, config: DebugConfig) -> None:
        self.config = config

    def provider(self) -> FrameProvider:
        if self.config.mock_mode:
            return VideoFileProvider(self.config.video_path)
        return CameraProvider()


def synced_gpx_point(gpx_points: list[tuple[float, float]], frame_idx: int) -> Optional[tuple[float, float]]:
    if not gpx_points:
        return None
    return gpx_points[frame_idx % len(gpx_points)]
