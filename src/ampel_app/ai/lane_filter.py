"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from collections.abc import Iterable

from ampel_app.core.models import TrafficLightCandidate


def lane_context_filter(
    detected_lights: Iterable[TrafficLightCandidate],
    gps_heading: float,
    imu_lateral_g: float,
) -> list[TrafficLightCandidate]:
    lane_hint = "left" if imu_lateral_g < -0.05 else "right" if imu_lateral_g > 0.05 else "center"
    filtered = []
    for light in detected_lights:
        if not light.lane_ids:
            continue
        if lane_hint == "center" and light.confidence > 0.4:
            filtered.append(light)
            continue
        if lane_hint == "left" and any("left" in lane for lane in light.lane_ids):
            filtered.append(light)
            continue
        if lane_hint == "right" and any("right" in lane for lane in light.lane_ids):
            filtered.append(light)
            continue
        if abs(gps_heading) <= 360 and light.confidence > 0.8:
            filtered.append(light)
    return filtered
