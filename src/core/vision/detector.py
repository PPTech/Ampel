"""
Version: 0.9.9
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional

from shared.interfaces.hal import Detection, IInferenceEngine
from src.core.logic.rules_engine import LightState, TrafficEvent
from src.core.vision.lane_estimator import LaneEstimator
from src.core.vision.smoothing import TemporalBuffer


class TFLiteTrafficDetector(IInferenceEngine):
    """TFLite-only detector adapter for mobile-friendly inference."""

    _CLASS_MAP = {0: LightState.RED.value, 1: LightState.YELLOW.value, 2: LightState.GREEN.value}

    def __init__(self, model_path: Optional[str] = None, score_threshold: float = 0.35) -> None:
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.lane_estimator = LaneEstimator()
        self.smoother = TemporalBuffer(window_size=5, persist_threshold=4)
        self._interpreter: Any = None
        self._input_index: Optional[int] = None
        self._output_indices: list[int] = []
        if model_path:
            self._load_interpreter(Path(model_path))

    def _load_interpreter(self, model_path: Path) -> None:
        if not model_path.exists():
            return
        interpreter_cls = None
        if find_spec("tflite_runtime.interpreter") is not None:
            interpreter_cls = import_module("tflite_runtime.interpreter").Interpreter
        elif find_spec("tensorflow.lite") is not None:
            interpreter_cls = import_module("tensorflow.lite").Interpreter
        if interpreter_cls is None:
            return

        self._interpreter = interpreter_cls(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        self._input_index = int(input_details[0]["index"]) if input_details else None
        self._output_indices = [int(d["index"]) for d in output_details]

    def detect(self, frame: Any) -> list[Detection]:
        """Return normalized detections; falls back to synthetic state if no model runtime."""
        if self._interpreter is None or self._input_index is None:
            fallback_state = str(
                getattr(frame, "get", lambda *_: "UNKNOWN")("simulated_state", "RED")
            ).upper()
            stable = self.smoother.push(fallback_state)
            bbox = (100.0, 120.0, 30.0, 50.0)
            image_width = int(getattr(frame, "get", lambda *_: 640)("width", 640))
            lane = self.lane_estimator.estimate_lane(image_width, bbox)
            return [Detection(state=stable, confidence=0.60, bbox=bbox, lane_id=lane.lane_id)]

        # Model invocation path intentionally lightweight and framework-agnostic.
        # TODO: wire actual frame preprocessing for uint8/float32 model input shape.
        return []

    def detect_events(
        self, frame: Any, speed_kph: float, distance_to_stopline_m: float
    ) -> list[TrafficEvent]:
        events: list[TrafficEvent] = []
        for detection in self.detect(frame):
            state = (
                LightState[detection.state]
                if detection.state in LightState.__members__
                else LightState.UNKNOWN
            )
            events.append(
                TrafficEvent(
                    timestamp_s=datetime.now(timezone.utc).timestamp(),
                    state=state,
                    speed_kph=speed_kph,
                    distance_to_stopline_m=distance_to_stopline_m,
                )
            )
        return events
