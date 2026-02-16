"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time


def hash_intersection(lat: float, lon: float) -> str:
    salt = os.environ.get("AMPEL_HASH_SALT", "ampel-default-salt")
    lat_r = round(lat, 3)
    lon_r = round(lon, 3)
    raw = f"{lat_r:.3f}:{lon_r:.3f}:{salt}".encode()
    return hashlib.sha256(raw).hexdigest()


class AdaptiveAgent:
    def __init__(self, db_path: str = "traffic_ai.sqlite3") -> None:
        self.conn = sqlite3.connect(db_path)

    def predict_time_to_green(
        self, intersection_hash: str, local_time_unix: int | None = None
    ) -> float:
        ts = int(local_time_unix or time.time())
        hour_factor = 1.12 if 7 <= time.localtime(ts).tm_hour <= 9 else 1.0
        cur = self.conn.cursor()
        cur.execute(
            "SELECT avg_wait_time, local_weight FROM route_memory WHERE intersection_hash=?",
            (intersection_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return 30.0 * hour_factor
        return max(3.0, min(120.0, float(row[0]) * float(row[1]) * hour_factor))
