"""
Version: 0.9.13
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from src.ampel_app.cli import analyze_uploaded_image, analyze_uploaded_video


def test_image_payload_contains_sign_and_light_boxes() -> None:
    # Tiny valid PNG header payload
    payload = "data:image/png;base64,iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAA"
    out = analyze_uploaded_image(payload)
    classes = {o.get("class") for o in out.get("objects", [])}
    assert "traffic_light" in classes
    assert "traffic_sign" in classes
    assert all("bbox" in o for o in out.get("objects", []))


def test_video_payload_contains_timeline_messages_and_boxes() -> None:
    out = analyze_uploaded_video("sample.mp4")
    assert isinstance(out.get("timeline"), list)
    assert len(out["timeline"]) >= 1
    first = out["timeline"][0]
    assert "message" in first
    assert any("bbox" in obj for obj in first.get("objects", []))
