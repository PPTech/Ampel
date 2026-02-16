"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from typing import Any


def anonymize_frame(frame: Any) -> Any:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        return frame
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    if frame is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for x, y, w, h in list(face.detectMultiScale(gray, 1.1, 4)) + list(
        plate.detectMultiScale(gray, 1.1, 4)
    ):
        roi = frame[y : y + h, x : x + w]
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (31, 31), 0)
    return frame
