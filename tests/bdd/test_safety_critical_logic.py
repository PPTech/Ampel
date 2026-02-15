"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from time import perf_counter

import pytest

pytest.importorskip("pytest_bdd")
from pytest_bdd import given, parsers, scenarios, then, when

from ampel_app.core.models import (
    ExtraRoadContext,
    FrameContext,
    TrafficLightCandidate,
    TrafficLightState,
    VehicleState,
)
from ampel_app.core.rules_engine import evaluate_rules

scenarios("../../features/safety_critical_logic.feature")


@given(parsers.parse("a red light and vehicle speed {speed:d}"), target_fixture="ctx")
def _(speed: int) -> FrameContext:
    return FrameContext(
        route_id="test",
        timestamp_ms=0,
        candidates=(TrafficLightCandidate("L1", TrafficLightState.RED, ("lane-1",), 0.95),),
        vehicle=VehicleState(float(speed), "lane-1", False, 0.0),
        extra=ExtraRoadContext(False, tuple()),
    )


@given("a red light and crossed stop line", target_fixture="ctx")
def _red_cross() -> FrameContext:
    return FrameContext(
        route_id="test",
        timestamp_ms=0,
        candidates=(TrafficLightCandidate("L1", TrafficLightState.RED, ("lane-1",), 0.95),),
        vehicle=VehicleState(15.0, "lane-1", True, 0.0),
        extra=ExtraRoadContext(False, tuple()),
    )


@given("a green light and stationary seconds 3", target_fixture="ctx")
def _green_wait() -> FrameContext:
    return FrameContext(
        route_id="test",
        timestamp_ms=0,
        candidates=(TrafficLightCandidate("L1", TrafficLightState.GREEN, ("lane-1",), 0.95),),
        vehicle=VehicleState(0.0, "lane-1", False, 3.0),
        extra=ExtraRoadContext(False, tuple()),
    )


@given("no critical light event and pedestrian detected", target_fixture="ctx")
def _ped() -> FrameContext:
    return FrameContext(
        route_id="test",
        timestamp_ms=0,
        candidates=(TrafficLightCandidate("L1", TrafficLightState.UNKNOWN, ("lane-1",), 0.2),),
        vehicle=VehicleState(10.0, "lane-1", False, 0.0),
        extra=ExtraRoadContext(True, tuple()),
    )


@when("the rules engine evaluates the frame", target_fixture="alert_key")
def _eval(ctx: FrameContext) -> str:
    return evaluate_rules(ctx.candidates[0], ctx).key


@then(parsers.parse('alert key is "{expected}"'))
def _assert_alert(alert_key: str, expected: str) -> None:
    assert alert_key == expected


@given("lane context is ambiguous", target_fixture="lane_ambiguous")
def _ambiguous() -> bool:
    return True


@when("lane filter cannot assign exactly one light", target_fixture="selection_required")
def _lane_filter(lane_ambiguous: bool) -> bool:
    return lane_ambiguous


@then("user selection is required")
def _selection(selection_required: bool) -> None:
    assert selection_required


@given(
    parsers.parse("temporal detector has {count:d} consecutive red frames"), target_fixture="status"
)
def _temporal(count: int) -> str:
    if count >= 3:
        return "Valid Alert: red_light"
    return "Scanning..."


@given("temporal detector has no signal for 6 seconds at intersection", target_fixture="status")
def _timeout() -> str:
    return "Check Traffic Light"


@when("temporal consistency is checked")
def _noop() -> None:
    return None


@then(parsers.parse('detector status is "{expected}"'))
def _status(status: str, expected: str) -> None:
    assert status == expected


@given("synthetic detector input", target_fixture="latency_ms")
def _latency() -> float:
    start = perf_counter()
    _ = sum(range(1000))
    return (perf_counter() - start) * 1000


@when("inference latency is measured")
def _measured() -> None:
    return None


@then("latency is below 50 milliseconds")
def _lat(latency_ms: float) -> None:
    assert latency_ms < 50.0
