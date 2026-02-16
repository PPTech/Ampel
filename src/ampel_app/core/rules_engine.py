"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import AlertChannel, FrameContext, TrafficLightCandidate, TrafficLightState


@dataclass(frozen=True)
class Alert:
    key: str
    channel: AlertChannel


def evaluate_rules(light: TrafficLightCandidate, ctx: FrameContext) -> Alert:
    if light.state == TrafficLightState.RED and ctx.vehicle.speed_kph > 45:
        return Alert("overspeed_red", AlertChannel.AUDIO)
    if light.state == TrafficLightState.RED and ctx.vehicle.crossed_stop_line:
        return Alert("red_crossed", AlertChannel.SIREN)
    if light.state == TrafficLightState.GREEN and ctx.vehicle.stationary_seconds >= 2.0:
        return Alert("green_wait", AlertChannel.AUDIO)
    if ctx.extra.pedestrian_detected:
        return Alert("pedestrian", AlertChannel.VISUAL)
    return Alert("ok", AlertChannel.NONE)
