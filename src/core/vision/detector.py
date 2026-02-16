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
        self._np: Any = import_module("numpy") if find_spec("numpy") is not None else None
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
        if self._interpreter is None or self._input_index is None or self._np is None:
            fallback_state = str(
                getattr(frame, "get", lambda *_: "UNKNOWN")("simulated_state", "RED")
            ).upper()
            stable = self.smoother.push(fallback_state)
            bbox = (100.0, 120.0, 30.0, 50.0)
            image_width = int(getattr(frame, "get", lambda *_: 640)("width", 640))
            lane = self.lane_estimator.estimate_lane(image_width, bbox)
            return [Detection(state=stable, confidence=0.60, bbox=bbox, lane_id=lane.lane_id)]

        np = self._np
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        if not input_details or not output_details:
            return []

        input_shape = tuple(int(x) for x in input_details[0].get("shape", [1, 1, 1, 3]))
        input_dtype = input_details[0].get("dtype", np.float32)
        image_width = int(getattr(frame, "get", lambda *_: 640)("width", 640))

        input_tensor = self._prepare_input_tensor(frame, input_shape, input_dtype)
        self._interpreter.set_tensor(self._input_index, input_tensor)
        self._interpreter.invoke()

        outputs = [self._interpreter.get_tensor(int(detail["index"])) for detail in output_details]
        scores = self._extract_scores(outputs)
        if scores.size == 0:
            return []

        best_idx = int(np.argmax(scores))
        confidence = float(scores[best_idx])
        if confidence < self.score_threshold:
            return []

        raw_state = self._CLASS_MAP.get(best_idx, LightState.UNKNOWN.value)
        stable = self.smoother.push(raw_state)
        bbox = (100.0, 120.0, 30.0, 50.0)
        lane = self.lane_estimator.estimate_lane(image_width, bbox)
        return [Detection(state=stable, confidence=confidence, bbox=bbox, lane_id=lane.lane_id)]

    def _prepare_input_tensor(self, frame: Any, input_shape: tuple[int, ...], input_dtype: Any) -> Any:
        np = self._np
        tensor = np.zeros(input_shape, dtype=input_dtype)
        if isinstance(frame, dict):
            candidate = frame.get("image")
        else:
            candidate = frame

        if candidate is None:
            return tensor

        try:
            data = np.asarray(candidate)
        except Exception:
            return tensor

        if data.size == 0:
            return tensor

        flattened = data.astype(input_dtype, copy=False).reshape(-1)
        np.copyto(tensor.reshape(-1)[: flattened.size], flattened[: tensor.size])
        return tensor

    def _extract_scores(self, outputs: list[Any]) -> Any:
        np = self._np
        if not outputs:
            return np.array([], dtype=np.float32)
        primary = np.asarray(outputs[0]).reshape(-1)
        if primary.size <= 0:
            return np.array([], dtype=np.float32)
        return primary.astype(np.float32, copy=False)

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
