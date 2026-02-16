"""
Version: 0.9.19
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from src.ampel_app.cli import DatasetRegistry


def test_license_compatibility_accepts_open_terms() -> None:
    assert DatasetRegistry.is_license_compatible("Apache-2.0")
    assert DatasetRegistry.is_license_compatible("CC-BY-4.0")


def test_license_compatibility_rejects_blocked_terms() -> None:
    assert not DatasetRegistry.is_license_compatible("all rights reserved")
    assert not DatasetRegistry.is_license_compatible("unknown")
