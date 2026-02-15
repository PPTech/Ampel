"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ampel_app.core.models import TrafficLightState

LABEL_MAP = {
    "red_light": TrafficLightState.RED,
    "green_light": TrafficLightState.GREEN,
    "yellow_light": TrafficLightState.YELLOW,
    "traffic light red": TrafficLightState.RED,
    "traffic light yellow": TrafficLightState.YELLOW,
    "traffic light green": TrafficLightState.GREEN,
}


@dataclass(frozen=True)
class MappedDetection:
    state: TrafficLightState
    score: float
    model_score: float
    color_score: float


def load_label_map(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {k: v.value for k, v in LABEL_MAP.items()}
    import json

    return json.loads(p.read_text(encoding="utf-8"))


def classify_color(crop: Any) -> tuple[TrafficLightState, float]:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        return TrafficLightState.UNKNOWN, 0.0
    if crop is None or getattr(crop, "size", 0) == 0:
        return TrafficLightState.UNKNOWN, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        TrafficLightState.RED: cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        + cv2.inRange(hsv, (160, 80, 80), (180, 255, 255)),
        TrafficLightState.YELLOW: cv2.inRange(hsv, (15, 80, 80), (40, 255, 255)),
        TrafficLightState.GREEN: cv2.inRange(hsv, (40, 60, 60), (95, 255, 255)),
    }
    total = float(crop.shape[0] * crop.shape[1])
    scores = {state: float(mask.sum() / 255.0) / total for state, mask in masks.items()}
    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]
    if best_score < 0.05:
        return TrafficLightState.UNKNOWN, 0.0
    return best_state, min(1.0, best_score * 3.0)


def fuse_state(
    model_state: TrafficLightState,
    model_conf: float,
    color_state: TrafficLightState,
    color_conf: float,
) -> MappedDetection:
    if model_state == color_state and model_state != TrafficLightState.UNKNOWN:
        final = model_state
    elif model_conf >= 0.85:
        final = model_state
    elif color_conf >= 0.75:
        final = color_state
    else:
        final = TrafficLightState.UNKNOWN
    score = (model_conf * 0.7) + (color_conf * 0.3)
    return MappedDetection(state=final, score=score, model_score=model_conf, color_score=color_conf)
