"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from ampel_app.ai.detector import classify_color
from ampel_app.core.models import TrafficLightState


def test_classify_red_crop() -> None:
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    crop[:, :] = (0, 0, 255)
    state, _ = classify_color(crop)
    assert state == TrafficLightState.RED


def test_classify_yellow_crop() -> None:
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    crop[:, :] = (0, 255, 255)
    state, _ = classify_color(crop)
    assert state == TrafficLightState.YELLOW


def test_classify_green_crop() -> None:
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    crop[:, :] = (0, 255, 0)
    state, _ = classify_color(crop)
    assert state == TrafficLightState.GREEN


def test_classify_unknown_crop() -> None:
    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    state, _ = classify_color(crop)
    assert state in {TrafficLightState.UNKNOWN, TrafficLightState.RED}
