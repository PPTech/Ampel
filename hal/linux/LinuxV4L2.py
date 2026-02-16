"""Version: 0.9.5
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import cv2


class LinuxV4L2:
    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.device_index)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def next_frame(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None
