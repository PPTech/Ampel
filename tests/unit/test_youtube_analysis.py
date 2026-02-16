"""
Version: 0.9.14
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from src.ampel_app.cli import analyze_youtube_link


def test_youtube_analysis_accepts_valid_url() -> None:
    out = analyze_youtube_link("https://www.youtube.com/watch?v=abc123")
    assert out.get("source") == "youtube"
    assert isinstance(out.get("timeline"), list)


def test_youtube_analysis_rejects_invalid_url() -> None:
    out = analyze_youtube_link("https://example.com/video")
    assert out.get("error") == "invalid_youtube_url"
