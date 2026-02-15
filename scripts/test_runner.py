#!/usr/bin/env python3
"""
Version: 0.9.2
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FEATURES_DIR = REPO_ROOT / "features"

from ai_engine.inference.edge_detection import SensorSnapshot, LightDetection, LaneContextFilter


def run_bdd_features() -> Dict[str, object]:
    feature_files = sorted(str(p) for p in FEATURES_DIR.glob("*.feature"))
    if not feature_files:
        return {"executed": False, "reason": "no feature files"}
    cmd = ["python3", "-m", "pytest", "-q"] + feature_files
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "executed": True,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-1000:],
        "stderr": proc.stderr[-1000:],
    }


def run_inference_benchmark(iterations: int = 300) -> Dict[str, object]:
    detections = [
        LightDetection("L1", lane_hint=0, heading_deg=90.0, confidence=0.93, bbox_xyxy=(0, 0, 10, 20)),
        LightDetection("L2", lane_hint=2, heading_deg=150.0, confidence=0.41, bbox_xyxy=(2, 1, 11, 19)),
    ]
    imu = SensorSnapshot(gyro_yaw_rate=0.2, accel_x=0.1, accel_y=0.0, accel_z=9.8)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = LaneContextFilter(detected_lights=detections, gps_heading=92.0, imu=imu)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_ms = elapsed_ms / iterations
    return {"iterations": iterations, "avg_ms": round(avg_ms, 4), "pass_lt_50ms": avg_ms < 50.0}


def run_privacy_audit() -> Dict[str, object]:
    risky_patterns = [
        r"print\(.*gps",
        r"logger\..*gps",
        r"print\(.*latitude",
        r"print\(.*longitude",
        r"logger\..*license_plate",
    ]
    hits: List[Dict[str, object]] = []
    for py_file in REPO_ROOT.rglob("*.py"):
        if "/.venv/" in str(py_file):
            continue
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for idx, line in enumerate(text.splitlines(), start=1):
            for pat in risky_patterns:
                if re.search(pat, line, flags=re.IGNORECASE):
                    hits.append({"file": str(py_file.relative_to(REPO_ROOT)), "line": idx, "text": line.strip()[:200]})
    return {"hits": hits, "pass": len(hits) == 0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ampel test runner")
    parser.add_argument("--skip-bdd", action="store_true")
    args = parser.parse_args()

    result: Dict[str, object] = {}
    if not args.skip_bdd:
        result["bdd"] = run_bdd_features()
    result["benchmark"] = run_inference_benchmark()
    result["privacy"] = run_privacy_audit()

    bdd_ok = True
    if not args.skip_bdd:
        bdd = result["bdd"]
        bdd_ok = bool((not bdd.get("executed")) or bdd.get("returncode") == 0)

    ok = bdd_ok and bool(result["benchmark"]["pass_lt_50ms"]) and bool(result["privacy"]["pass"])
    result["overall_pass"] = ok
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
