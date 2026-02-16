#!/usr/bin/env python3
"""
Version: 0.9.12
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path


SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(apikey|api_key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?<!\d)([-+]?\d{1,3}\.\d{4,})\s*,\s*([-+]?\d{1,3}\.\d{4,})(?!\d)"),
]


def redact(text: str) -> str:
    out = text
    for pattern in SENSITIVE_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def extract_failed_scenarios(pytest_output: str) -> list[str]:
    scenarios: list[str] = []
    for line in pytest_output.splitlines():
        if "Scenario:" in line or "FAILED" in line:
            scenarios.append(line.strip())
    return scenarios


def run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create self-healing failure diagnostics artifact")
    parser.add_argument(
        "--pytest-log", default="pytest_output.log", help="Path to pytest output log"
    )
    parser.add_argument("--system-log", default="system_output.log", help="Path to system/app log")
    parser.add_argument("--output", default="failure_report.zip", help="Zip artifact output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pytest_text = (
        Path(args.pytest_log).read_text(encoding="utf-8") if Path(args.pytest_log).exists() else ""
    )
    sys_text = (
        Path(args.system_log).read_text(encoding="utf-8") if Path(args.system_log).exists() else ""
    )

    scenarios = extract_failed_scenarios(pytest_text)
    diff_text = run_command(["git", "diff", "--", "."])

    report = {
        "failed_scenarios": scenarios,
        "notes": "Auto-generated diagnostics with redacted sensitive tokens.",
    }

    staging = Path(".failure_report")
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "failed_scenarios.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (staging / "system_logs.redacted.log").write_text(redact(sys_text), encoding="utf-8")
    (staging / "pytest_output.redacted.log").write_text(redact(pytest_text), encoding="utf-8")
    (staging / "git_diff.patch").write_text(redact(diff_text), encoding="utf-8")

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in staging.iterdir():
            zf.write(file, arcname=file.name)

    for file in staging.iterdir():
        file.unlink(missing_ok=True)
    os.rmdir(staging)
    print(f"created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
