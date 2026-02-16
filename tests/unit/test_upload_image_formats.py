"""
Version: 0.9.12
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

import base64

from src.ampel_app.cli import analyze_uploaded_image


def _to_data_url(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def test_analyze_uploaded_image_accepts_png() -> None:
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    out = analyze_uploaded_image(_to_data_url(png_header, "image/png"))
    assert out["format"] == "png"
    assert out["traffic_light_state"] in {"red", "yellow", "green"}


def test_analyze_uploaded_image_accepts_webp() -> None:
    webp_header = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 8
    out = analyze_uploaded_image(_to_data_url(webp_header, "image/webp"))
    assert out["format"] == "webp"


def test_analyze_uploaded_image_rejects_bad_payload() -> None:
    out = analyze_uploaded_image("data:image/png;base64,%%%%")
    assert "error" in out
