"""
Version: 0.9.9
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LaneEstimate:
    lane_id: int
    lane_label: str


class LaneEstimator:
    """MVP lane assignment using normalized bbox center heuristic."""

    def __init__(self, left_threshold: float = 0.30, right_threshold: float = 0.70) -> None:
        self.left_threshold = left_threshold
        self.right_threshold = right_threshold

    def estimate_lane(
        self, image_width: int, bbox: Optional[Tuple[float, float, float, float]]
    ) -> LaneEstimate:
        if image_width <= 0 or not bbox:
            return LaneEstimate(lane_id=0, lane_label="CENTER")

        x, _, w, _ = bbox
        center_x = x + (w / 2.0)
        norm_x = center_x / float(image_width)

        if norm_x < self.left_threshold:
            return LaneEstimate(lane_id=-1, lane_label="LEFT")
        if norm_x > self.right_threshold:
            return LaneEstimate(lane_id=1, lane_label="RIGHT")
        return LaneEstimate(lane_id=0, lane_label="CENTER")
