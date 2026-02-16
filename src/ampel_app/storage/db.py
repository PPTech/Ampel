"""Version: 0.9.4
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


@dataclass
class DB:
    conn: sqlite3.Connection


def connect(db_path: str) -> DB:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS route_memory(
            intersection_hash TEXT PRIMARY KEY,
            avg_wait_time REAL NOT NULL DEFAULT 30.0,
            light_cycle_pattern TEXT NOT NULL DEFAULT 'unknown',
            local_weight REAL NOT NULL DEFAULT 1.0,
            samples INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return DB(conn=conn)


def retention_cleanup(db: DB, retention_days: int) -> int:
    cutoff = int(time.time()) - (retention_days * 86400)
    cur = db.conn.cursor()
    try:
        cur.execute("DELETE FROM route_memory WHERE updated_at > 0 AND updated_at < ?", (cutoff,))
        affected = cur.rowcount
        db.conn.commit()
        return max(affected, 0)
    except sqlite3.Error:
        return 0
