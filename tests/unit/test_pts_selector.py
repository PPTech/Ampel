"""
Version: 0.9.15
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from scripts.pts_selector import choose_targets


def test_shared_specs_change_runs_all() -> None:
    out = choose_targets(["shared/specs/traffic_event.schema.json"])
    assert out.run_android is True
    assert out.run_gadget is True
    assert out.run_brain is True


def test_mobile_change_runs_android_only() -> None:
    out = choose_targets(["mobile/android/app/src/main/AndroidManifest.xml"])
    assert out.run_android is True
    assert out.run_gadget is False


def test_gadget_change_runs_gadget_only() -> None:
    out = choose_targets(["gadget/linux/driver.py"])
    assert out.run_android is False
    assert out.run_gadget is True


def test_core_python_change_runs_brain_only() -> None:
    out = choose_targets(["src/ampel_app/cli.py"])
    assert out.run_brain is True
    assert out.run_android is False
    assert out.run_gadget is False
