"""
Version: 0.9.17
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from pathlib import Path

from src.ampel_app.cli import DB, _feedback_state_bias, _record_feedback


def test_feedback_bias_updates_preferred_state(tmp_path: Path) -> None:
    db = DB(tmp_path / "feedback.sqlite3")
    for _ in range(3):
        _record_feedback(
            db,
            source="image",
            predicted_state="red",
            correct=False,
            corrected_state="green",
        )
    assert _feedback_state_bias(db, "image", "red") == "green"
