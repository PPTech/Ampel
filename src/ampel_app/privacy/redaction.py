"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import json
import re
from typing import Any

GEO_COORD_PATTERN = re.compile(r"(?<!\d)([-+]?\d{1,3}\.\d{4,})\s*,\s*([-+]?\d{1,3}\.\d{4,})(?!\d)")
PLATE_PATTERN = re.compile(r"\b([A-Z]{1,3}-[A-Z]{1,2}\s?\d{1,4})\b", re.IGNORECASE)


def redact_text(value: str) -> str:
    value = GEO_COORD_PATTERN.sub("[GEO-REDACTED]", value)
    return PLATE_PATTERN.sub("[PLATE-REDACTED]", value)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, v in value.items():
            lk = str(key).lower()
            if any(x in lk for x in ("image", "bitmap", "frame", "uiimage")):
                out[key] = "[BINARY-REDACTED]"
            else:
                out[key] = sanitize(v)
        return out
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[BINARY-REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    return value


def safe_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(sanitize(payload), ensure_ascii=False)
