"""
Version: 0.9.9
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from src.core.vision.smoothing import TemporalBuffer


def test_temporal_buffer_rejects_single_frame_glitch() -> None:
    buf = TemporalBuffer(window_size=5, persist_threshold=4)
    sequence = ["RED", "RED", "RED", "RED", "GREEN", "RED", "RED"]
    outputs = [buf.push(s) for s in sequence]
    assert outputs[-1] == "RED"
    assert "GREEN" not in outputs
