#!/usr/bin/env python3
"""
Version: 0.9.8
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "proto" / "python" / "traffic_ai_assist.py"
    runpy.run_path(str(target), run_name="__main__")
