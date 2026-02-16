"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from ampel_app.ai.agent import hash_intersection
from ampel_app.privacy.redaction import safe_json_dumps


def test_hash_intersection_stable() -> None:
    h1 = hash_intersection(52.520008, 13.404954)
    h2 = hash_intersection(52.520011, 13.404951)
    assert h1 == h2


def test_redaction_hides_geo_and_plate() -> None:
    payload = {"msg": "latlon 52.5200,13.4050 B-AB 123"}
    rendered = safe_json_dumps(payload)
    assert "[GEO-REDACTED]" in rendered
    assert "[PLATE-REDACTED]" in rendered
