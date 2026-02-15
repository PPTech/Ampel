#!/usr/bin/env python3
"""
Version: 0.9.2
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

ROUTE_MEMORY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS route_memory (
  intersection_hash TEXT PRIMARY KEY,
  avg_wait_time REAL NOT NULL DEFAULT 30.0,
  light_cycle_pattern TEXT NOT NULL DEFAULT 'unknown',
  local_weight REAL NOT NULL DEFAULT 1.0,
  samples INTEGER NOT NULL DEFAULT 0,
  updated_at INTEGER NOT NULL DEFAULT 0
);
"""

PRIVACY_RULE = (
    "All learning is on-device."
    " No raw GPS history and no intersection coordinates are uploaded to cloud services."
)


@dataclass(frozen=True)
class RouteMemory:
    intersection_hash: str
    avg_wait_time: float
    light_cycle_pattern: str
    local_weight: float
    samples: int


class AdaptiveAgent:
    """Federated-style local agent (privacy-preserving, on-device only)."""

    def __init__(self, db_path: str = "traffic_ai.sqlite3") -> None:
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(ROUTE_MEMORY_SCHEMA_SQL)
        self.conn.commit()

    def predict_time_to_green(self, intersection_hash: str, local_time_unix: Optional[int] = None) -> float:
        """Input: location hash + time. Output: predicted seconds to green."""
        ts = int(local_time_unix or time.time())
        hour_factor = 1.12 if 7 <= time.localtime(ts).tm_hour <= 9 else 1.0

        cur = self.conn.cursor()
        cur.execute(
            "SELECT avg_wait_time, local_weight FROM route_memory WHERE intersection_hash=?",
            (intersection_hash,),
        )
        row = cur.fetchone()
        if not row:
            return 30.0 * hour_factor
        avg_wait_time, local_weight = float(row[0]), float(row[1])
        return max(3.0, min(120.0, avg_wait_time * local_weight * hour_factor))

    def update_observation(self, intersection_hash: str, measured_wait: float, cycle_pattern: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT avg_wait_time, samples, local_weight FROM route_memory WHERE intersection_hash=?",
            (intersection_hash,),
        )
        row = cur.fetchone()
        now = int(time.time())
        if row:
            prev_wait, samples, weight = float(row[0]), int(row[1]), float(row[2])
            new_samples = samples + 1
            new_wait = ((prev_wait * samples) + measured_wait) / new_samples
            cur.execute(
                "UPDATE route_memory SET avg_wait_time=?, light_cycle_pattern=?, samples=?, local_weight=?, updated_at=? WHERE intersection_hash=?",
                (new_wait, cycle_pattern, new_samples, weight, now, intersection_hash),
            )
        else:
            cur.execute(
                "INSERT INTO route_memory(intersection_hash, avg_wait_time, light_cycle_pattern, local_weight, samples, updated_at) VALUES(?,?,?,?,?,?)",
                (intersection_hash, measured_wait, cycle_pattern, 1.0, 1, now),
            )
        self.conn.commit()

    def apply_feedback_false_positive_brake(self, intersection_hash: str, hard_brake_at_green: bool) -> None:
        """Feedback loop: if user brakes hard at green, reduce confidence weight."""
        if not hard_brake_at_green:
            return
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE route_memory SET local_weight = MAX(0.55, local_weight * 0.92), updated_at=? WHERE intersection_hash=?",
            (int(time.time()), intersection_hash),
        )
        self.conn.commit()
