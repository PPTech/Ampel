"""
Version: 0.9.21
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from src.ampel_app.cli import analyze_uploaded_video, analyze_youtube_link, dataset_name_matches


def test_dataset_name_matching_accepts_lisa_alias() -> None:
    assert dataset_name_matches("LISA traffic-light reference", "LISA Traffic Light Dataset")


def test_video_timeline_contains_state_progression() -> None:
    out = analyze_uploaded_video("city_yellow_cycle.mp4")
    timeline = out.get("timeline", [])
    states = [frame.get("traffic_light_state") for frame in timeline]
    assert "yellow" in states
    assert states[-1] == "green"


def test_youtube_payload_avoids_stale_boxes() -> None:
    out = analyze_youtube_link("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert out.get("objects") == []
    assert all(
        "bbox" not in obj for frame in out.get("timeline", []) for obj in frame.get("objects", [])
    )
