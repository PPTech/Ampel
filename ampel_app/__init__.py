"""Compatibility shim for src-layout execution without installation.
Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import pkgutil
from pathlib import Path

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_src_pkg = Path(__file__).resolve().parent.parent / "src" / "ampel_app"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))
