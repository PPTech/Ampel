"""
Version: 0.9.8
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_bdd")
from pytest_bdd import given, scenario, then, when

from src.core.logic.rules_engine import AlertType, LightState, TrafficEvent, TrafficRulesEngine


@scenario("../../features/traffic_rules.feature", "Red Light Violation Risk")
def test_red_violation_risk() -> None:
    pass


@scenario("../../features/traffic_rules.feature", "Green Light Idle (Stop & Go)")
def test_green_idle() -> None:
    pass


@scenario("../../features/traffic_rules.feature", "Yellow Light Caution")
def test_yellow_caution() -> None:
    pass


@given("state is RED", target_fixture="event")
def given_red() -> TrafficEvent:
    return TrafficEvent(
        timestamp_s=2.0, state=LightState.RED, speed_kph=40.0, distance_to_stopline_m=10.0
    )


@given("speed is > 30 km/h")
def given_speed() -> None:
    return None


@given("distance_to_stopline is < 20m")
def given_distance() -> None:
    return None


@given("state is GREEN", target_fixture="event")
def given_green() -> TrafficEvent:
    return TrafficEvent(
        timestamp_s=6.0, state=LightState.GREEN, speed_kph=0.0, distance_to_stopline_m=8.0
    )


@given("speed is 0 km/h for > 3 seconds", target_fixture="history")
def given_idle_history() -> list[TrafficEvent]:
    return [
        TrafficEvent(
            timestamp_s=0.0, state=LightState.GREEN, speed_kph=0.0, distance_to_stopline_m=12.0
        ),
        TrafficEvent(
            timestamp_s=4.5, state=LightState.GREEN, speed_kph=0.0, distance_to_stopline_m=10.0
        ),
    ]


@given("state is YELLOW", target_fixture="event")
def given_yellow() -> TrafficEvent:
    return TrafficEvent(
        timestamp_s=3.0, state=LightState.YELLOW, speed_kph=15.0, distance_to_stopline_m=14.0
    )


@when("the deterministic rules engine evaluates the event", target_fixture="alert")
def when_evaluate(event: TrafficEvent, history: list[TrafficEvent] | None = None) -> AlertType:
    engine = TrafficRulesEngine()
    return engine.evaluate(event, history or [])


@then("output CRITICAL_ALERT")
def then_critical(alert: AlertType) -> None:
    assert alert == AlertType.CRITICAL_ALERT


@then('output INFO_ALERT ("Please go")')
def then_info(alert: AlertType) -> None:
    assert alert == AlertType.INFO_ALERT


@then("output WARN_ALERT")
def then_warn(alert: AlertType) -> None:
    assert alert == AlertType.WARN_ALERT
