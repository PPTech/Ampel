"""
Version: 0.9.18
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


def test_video_payload_is_temporally_stable_by_default() -> None:
    out = analyze_uploaded_video("sample.mp4")
    timeline = out.get("timeline", [])
    states = {frame.get("traffic_light_state") for frame in timeline}
    assert len(states) == 1


def test_image_payload_boxes_are_normalized() -> None:
    payload = "data:image/png;base64,iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAA"
    out = analyze_uploaded_image(payload)
    for obj in out.get("objects", []):
        box = obj.get("bbox", [])
        assert len(box) == 4
        assert all(0.0 <= float(v) <= 1.0 for v in box)


def test_image_payload_confidence_varies_between_payloads() -> None:
    payload_a = "data:image/png;base64,iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAA"
    payload_b = "data:image/png;base64,iVBORw0KGgo/////////////////////"
    out_a = analyze_uploaded_image(payload_a)
    out_b = analyze_uploaded_image(payload_b)
    conf_a = [o.get("confidence") for o in out_a.get("objects", [])]
    conf_b = [o.get("confidence") for o in out_b.get("objects", [])]
    assert conf_a != conf_b
