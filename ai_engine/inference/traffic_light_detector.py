#!/usr/bin/env python3
"""
Version: 0.9.2
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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


class TrafficLightDetector:
    """Apache-2.0 friendly detector wrapper using TensorFlow Lite interpreter."""

    def __init__(self, model_path: str, threshold: float = 0.35) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.interpreter = self._load_interpreter(model_path)

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
        """Run inference and return filtered detections.

        Note: This project keeps post-processing minimal for portability.
        Convert this function to platform-optimized kernels for mobile release.
        """
        if image is None:
            return []
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_index = input_details[0]["index"]
        in_h = int(input_details[0]["shape"][1])
        in_w = int(input_details[0]["shape"][2])

        # Preprocess
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
