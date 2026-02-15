#!/usr/bin/env python3
"""
Version: 0.9.3
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    class _NP:
        ndarray = object
    np = _NP()


@dataclass(frozen=True)
class DetectionResult:
    label: str
    score: float
    bbox_xywh: Tuple[float, float, float, float]


@dataclass(frozen=True)
class TemporalAlertResult:
    message: str
    status: str
    detection: Optional[DetectionResult]
    buffer_count: int


class StateBuffer:
    """ISO-26262 style temporal state filter to suppress single-frame false alarms."""

    def __init__(self, size: int = 3) -> None:
        self.size = size
        self._states: Deque[str] = deque(maxlen=size)

    def push(self, state: str) -> int:
        self._states.append(state)
        return self.consecutive_count(state)

    def consecutive_count(self, state: str) -> int:
        count = 0
        for item in reversed(self._states):
            if item == state:
                count += 1
            else:
                break
        return count


class TrafficLightDetector:
    """Apache-2.0 friendly detector wrapper using TensorFlow Lite interpreter.

    Temporal consistency policy:
    - red light with confidence > 0.85 must persist for 3 consecutive frames.
    - same rule for green status validation.
    - if no light is detected for >5s while vehicle is near an intersection,
      return a fail-safe warning for the driver.
    """

    def __init__(self, model_path: str, threshold: float = 0.35) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.interpreter = self._load_interpreter(model_path)
        self.state_buffer = StateBuffer(size=3)
        self.last_light_seen_ts = time.monotonic()

    @staticmethod
    def _load_interpreter(model_path: str):
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ModuleNotFoundError:
            from tensorflow.lite import Interpreter  # type: ignore
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def detect(self, image: "np.ndarray") -> List[DetectionResult]:
        if image is None:
            return []
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_index = input_details[0]["index"]
        in_h = int(input_details[0]["shape"][1])
        in_w = int(input_details[0]["shape"][2])

        if hasattr(np, "array"):
            resized = np.array(image)
            if resized.shape[0] != in_h or resized.shape[1] != in_w:
                import cv2
                resized = cv2.resize(resized, (in_w, in_h))
            tensor = resized.astype("float32") / 255.0
            tensor = tensor.reshape((1, in_h, in_w, -1))
            self.interpreter.set_tensor(input_index, tensor)
            self.interpreter.invoke()

            boxes = self.interpreter.get_tensor(output_details[0]["index"])
            classes = self.interpreter.get_tensor(output_details[1]["index"])
            scores = self.interpreter.get_tensor(output_details[2]["index"])

            results: List[DetectionResult] = []
            for i, score in enumerate(scores[0]):
                s = float(score)
                if s < self.threshold:
                    continue
                y1, x1, y2, x2 = [float(v) for v in boxes[0][i]]
                label = f"class_{int(classes[0][i])}"
                results.append(DetectionResult(label=label, score=s, bbox_xywh=(x1, y1, x2 - x1, y2 - y1)))
            return results
        return []

    def evaluate_frame(self, detections: List[DetectionResult], at_intersection: bool) -> TemporalAlertResult:
        now = time.monotonic()
        winning = self._select_traffic_light(detections)
        if winning is None:
            if at_intersection and now - self.last_light_seen_ts > 5.0:
                return TemporalAlertResult(
                    message="Check Traffic Light",
                    status="failsafe_warning",
                    detection=None,
                    buffer_count=0,
                )
            return TemporalAlertResult("Scanning...", "scanning", None, 0)

        self.last_light_seen_ts = now
        state = winning.label.lower()
        if state not in {"red_light", "green_light"}:
            return TemporalAlertResult("Scanning...", "scanning", winning, 0)

        persisted = self.state_buffer.push(state)
        if persisted >= 3:
            return TemporalAlertResult(
                message=f"Valid Alert: {state}",
                status="valid_alert",
                detection=winning,
                buffer_count=persisted,
            )
        return TemporalAlertResult("Scanning...", "buffering", winning, persisted)

    @staticmethod
    def _select_traffic_light(detections: List[DetectionResult]) -> Optional[DetectionResult]:
        candidates = [
            d for d in detections
            if d.label.lower() in {"red_light", "green_light"} and d.score > 0.85
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda d: d.score, reverse=True)[0]
