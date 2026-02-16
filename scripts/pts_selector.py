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
from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    run_android: bool
    run_gadget: bool
    run_brain: bool
    reason: str


def choose_targets(changed_files: list[str]) -> Selection:
    if any(path.startswith("shared/specs/") for path in changed_files):
        return Selection(
            run_android=True, run_gadget=True, run_brain=True, reason="shared_specs_changed"
        )

    mobile_changed = any(path.startswith("mobile/") for path in changed_files)
    gadget_changed = any(path.startswith("gadget/") for path in changed_files)

    if mobile_changed and gadget_changed:
        return Selection(
            run_android=True, run_gadget=True, run_brain=True, reason="mobile_and_gadget_changed"
        )
    if mobile_changed:
        return Selection(
            run_android=True, run_gadget=False, run_brain=True, reason="mobile_changed"
        )
    if gadget_changed:
        return Selection(
            run_android=False, run_gadget=True, run_brain=True, reason="gadget_changed"
        )

    return Selection(run_android=True, run_gadget=True, run_brain=True, reason="default_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictive test selector for Ampel monorepo")
    parser.add_argument(
        "--changed-files",
        nargs="*",
        default=[],
        help="Space-separated changed file paths (e.g. from git diff --name-only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = choose_targets(args.changed_files)
    print(
        json.dumps(
            {
                "run_android": result.run_android,
                "run_gadget": result.run_gadget,
                "run_brain": result.run_brain,
                "reason": result.reason,
                "changed_files": args.changed_files,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
