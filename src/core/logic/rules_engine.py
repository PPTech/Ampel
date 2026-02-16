"""
Version: 0.9.8
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Iterable


class LightState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


class AlertType(str, Enum):
    NONE = "NONE"
    WARN_ALERT = "WARN_ALERT"
    INFO_ALERT = "INFO_ALERT"
    CRITICAL_ALERT = "CRITICAL_ALERT"


@dataclass(frozen=True)
class TrafficEvent:
    timestamp_s: float
    state: LightState
    speed_kph: float
    distance_to_stopline_m: float


class TrafficRulesEngine:
    """Deterministic rule-only engine (no AI guessing)."""

    def __init__(self, green_idle_threshold_s: float = 3.0) -> None:
        self.green_idle_threshold_s = green_idle_threshold_s

    @staticmethod
    def time_to_collision_s(speed_kph: float, distance_m: float) -> float:
        if distance_m <= 0.0:
            return 0.0
        speed_mps = max(0.0, speed_kph) / 3.6
        if speed_mps <= 0.0:
            return float("inf")
        return distance_m / speed_mps

    def evaluate(self, event: TrafficEvent, history: Iterable[TrafficEvent]) -> AlertType:
        if event.state == LightState.RED:
            ttc = self.time_to_collision_s(event.speed_kph, event.distance_to_stopline_m)
            if event.speed_kph > 30.0 and event.distance_to_stopline_m < 20.0 and ttc <= 2.5:
                return AlertType.CRITICAL_ALERT
            return AlertType.WARN_ALERT

        if event.state == LightState.YELLOW:
            return AlertType.WARN_ALERT

        if event.state == LightState.GREEN:
            window: Deque[TrafficEvent] = deque(history, maxlen=200)
            window.append(event)
            green_idle = [x for x in window if x.state == LightState.GREEN and x.speed_kph <= 0.0]
            if len(green_idle) >= 2:
                idle_duration = green_idle[-1].timestamp_s - green_idle[0].timestamp_s
            elif len(green_idle) == 1:
                idle_duration = 0.0
            else:
                idle_duration = -1.0
            if idle_duration > self.green_idle_threshold_s:
                return AlertType.INFO_ALERT
            return AlertType.NONE

        return AlertType.NONE
