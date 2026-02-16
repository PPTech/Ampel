"""
Version: 0.9.19
License: MIT
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.core.vision.detector import TFLiteTrafficDetector


class _FakeInterpreter:
    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores
        self._input: np.ndarray | None = None

    def get_input_details(self) -> list[dict[str, object]]:
        return [{"index": 0, "shape": np.array([1, 2, 2, 3]), "dtype": np.float32}]

    def get_output_details(self) -> list[dict[str, object]]:
        return [{"index": 1}]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        assert index == 0
        self._input = value

    def invoke(self) -> None:
        assert self._input is not None

    def get_tensor(self, index: int) -> np.ndarray:
        assert index == 1
        return self._scores


def test_detect_runs_model_path_and_returns_detection() -> None:
    detector = TFLiteTrafficDetector(model_path=None)
    detector._interpreter = _FakeInterpreter(np.array([[0.1, 0.2, 0.8]], dtype=np.float32))
    detector._input_index = 0

    detections = detector.detect({"width": 640, "image": np.ones((2, 2, 3), dtype=np.float32)})

    assert len(detections) == 1
    assert detections[0].state == "GREEN"
    assert detections[0].confidence == 0.8


def test_detect_respects_score_threshold() -> None:
    detector = TFLiteTrafficDetector(model_path=None, score_threshold=0.5)
    detector._interpreter = _FakeInterpreter(np.array([[0.2, 0.3, 0.4]], dtype=np.float32))
    detector._input_index = 0

    assert detector.detect({"width": 640, "image": np.zeros((2, 2, 3), dtype=np.float32)}) == []
