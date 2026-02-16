"""
Version: 0.9.8
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from collections import deque

from src.core.logic.rules_engine import AlertType, LightState, TrafficEvent, TrafficRulesEngine


def test_red_violation_risk_critical() -> None:
    engine = TrafficRulesEngine()
    e = TrafficEvent(
        timestamp_s=10.0, state=LightState.RED, speed_kph=45.0, distance_to_stopline_m=10.0
    )
    assert engine.evaluate(e, history=[]) == AlertType.CRITICAL_ALERT


def test_green_idle_info_alert() -> None:
    engine = TrafficRulesEngine(green_idle_threshold_s=3.0)
    history = deque(
        [
            TrafficEvent(
                timestamp_s=0.0, state=LightState.GREEN, speed_kph=0.0, distance_to_stopline_m=12.0
            ),
            TrafficEvent(
                timestamp_s=4.2, state=LightState.GREEN, speed_kph=0.0, distance_to_stopline_m=11.0
            ),
        ],
        maxlen=20,
    )
    event = TrafficEvent(
        timestamp_s=4.5, state=LightState.GREEN, speed_kph=0.0, distance_to_stopline_m=10.0
    )
    assert engine.evaluate(event, history=history) == AlertType.INFO_ALERT


def test_yellow_always_warn() -> None:
    engine = TrafficRulesEngine()
    e = TrafficEvent(
        timestamp_s=1.0, state=LightState.YELLOW, speed_kph=20.0, distance_to_stopline_m=15.0
    )
    assert engine.evaluate(e, history=[]) == AlertType.WARN_ALERT


def test_ttc_speed_zero_and_distance_zero_edges() -> None:
    engine = TrafficRulesEngine()
    assert engine.time_to_collision_s(speed_kph=0.0, distance_m=10.0) == float("inf")
    assert engine.time_to_collision_s(speed_kph=30.0, distance_m=0.0) == 0.0


def test_engine_stateless_history_input() -> None:
    engine = TrafficRulesEngine()
    history = deque(maxlen=5)
    event = TrafficEvent(
        timestamp_s=2.0, state=LightState.GREEN, speed_kph=5.0, distance_to_stopline_m=20.0
    )
    before = len(history)
    out = engine.evaluate(event, history=history)
    assert out in {AlertType.NONE, AlertType.INFO_ALERT}
    assert len(history) == before
