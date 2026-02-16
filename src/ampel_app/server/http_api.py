"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

from .html_templates import in_car_guard_notice


def health_extensions() -> dict[str, str]:
    return {"in_car_ui_policy": in_car_guard_notice()}
