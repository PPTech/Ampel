#!/usr/bin/env python3
"""
Version: 0.9.3
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path


def build_trace(points: int = 120) -> str:
    start_lat, start_lon = 52.520008, 13.404954
    t0 = datetime.now(timezone.utc)
    chunks = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<gpx version=\"1.1\" creator=\"AmpelAI\" xmlns=\"http://www.topografix.com/GPX/1/1\">",
        "  <trk><name>AmpelAI Mock Berlin City Center</name><trkseg>",
    ]
    for i in range(points):
        lat = start_lat + (i * 0.00002)
        lon = start_lon + (i * 0.00003)
        ts = (t0 + timedelta(milliseconds=500 * i)).isoformat()
        chunks.append(f"    <trkpt lat=\"{lat:.6f}\" lon=\"{lon:.6f}\"><time>{ts}</time></trkpt>")
    chunks.append("  </trkseg></trk></gpx>")
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sample GPX trace for camera mock playback.")
    parser.add_argument("--output", default="data/mock/sample_trace.gpx")
    parser.add_argument("--points", type=int, default=120)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_trace(points=max(30, args.points)), encoding="utf-8")
    print(f"generated: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
