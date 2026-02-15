#!/usr/bin/env python3
"""
Version: 0.9.0
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MANIFEST = Path("data/external_datasets_manifest.json")
EXTRA_MANIFEST = Path("data/manual_extra_datasets.json")
COMPILED_MANIFEST = Path("data/compiled_datasets_manifest.json")
DB_PATH = Path("traffic_ai.sqlite3")


def load_manifest() -> Dict[str, object]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_manual_extra() -> List[Dict[str, str]]:
    if not EXTRA_MANIFEST.exists():
        return []
    obj = json.loads(EXTRA_MANIFEST.read_text(encoding="utf-8"))
    return list(obj.get("datasets", []))


def compile_manifest(output: Path = COMPILED_MANIFEST) -> Dict[str, object]:
    base = list(load_manifest().get("datasets", []))
    extra = load_manual_extra()
    all_rows = base + extra
    dedup: Dict[str, Dict[str, str]] = {}
    for row in all_rows:
        key = str(row.get("name", "unknown")).strip().lower()
        if key not in dedup:
            dedup[key] = {
                "name": str(row.get("name", "unknown")),
                "scope": str(row.get("scope", "unknown")),
                "license": str(row.get("license", "unknown")),
                "url": str(row.get("url", "")),
                "usage": str(row.get("usage", "")),
                "status": str(row.get("status", "unknown")),
            }
    compiled = {
        "version": "0.9.0",
        "license": "MIT",
        "generated_at": int(time.time()),
        "datasets": list(dedup.values()),
        "totals": {
            "count": len(dedup),
            "download_required": sum(1 for x in dedup.values() if x.get("status") == "download_required"),
            "online_reference": sum(1 for x in dedup.values() if x.get("status") == "online_reference"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compiled, ensure_ascii=False, indent=2), encoding="utf-8")
    return compiled


def init_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS external_dataset_catalog(
          dataset_name TEXT PRIMARY KEY,
          scope TEXT,
          license TEXT,
          source_url TEXT,
          synced_at INTEGER
        )
        """
    )
    cur.execute("PRAGMA table_info(external_dataset_catalog)")
    cols = {r[1] for r in cur.fetchall()}
    if "usage_reason" not in cols:
        cur.execute("ALTER TABLE external_dataset_catalog ADD COLUMN usage_reason TEXT NOT NULL DEFAULT ''")
    if "status" not in cols:
        cur.execute("ALTER TABLE external_dataset_catalog ADD COLUMN status TEXT NOT NULL DEFAULT 'unknown'")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_health(
          dataset_name TEXT,
          checked_at INTEGER,
          online_ok INTEGER,
          notes TEXT
        )
        """
    )
    con.commit()
    return con


def sync_datasets() -> int:
    compiled = compile_manifest(COMPILED_MANIFEST)
    rows = list(compiled.get("datasets", []))
    con = init_db()
    cur = con.cursor()
    now = int(time.time())
    for ds in rows:
        cur.execute(
            """
            INSERT INTO external_dataset_catalog(dataset_name,scope,license,source_url,synced_at,usage_reason,status)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(dataset_name)
            DO UPDATE SET scope=excluded.scope, license=excluded.license, source_url=excluded.source_url, synced_at=excluded.synced_at, usage_reason=excluded.usage_reason, status=excluded.status
            """,
            (
                str(ds.get("name", "unknown")),
                str(ds.get("scope", "unknown")),
                str(ds.get("license", "unknown")),
                str(ds.get("url", "")),
                now,
                str(ds.get("usage", "")),
                str(ds.get("status", "unknown")),
            ),
        )
    con.commit()
    con.close()
    print(json.dumps({"synced": len(rows), "db": str(DB_PATH), "compiled_manifest": str(COMPILED_MANIFEST)}, ensure_ascii=False))
    return 0


def check_online() -> int:
    data = compile_manifest(COMPILED_MANIFEST)
    con = init_db()
    cur = con.cursor()
    checked = []
    now = int(time.time())
    for ds in data.get("datasets", []):
        name = str(ds.get("name", "unknown"))
        url = str(ds.get("url", ""))
        ok = 0
        notes = ""
        try:
            req = Request(url, headers={"User-Agent": "traffic-ai-assist-dataset-manager/0.9.0"}, method="GET")
            with urlopen(req, timeout=5) as resp:
                ok = 1 if resp.status < 400 else 0
                notes = f"http_status={resp.status}"
        except (URLError, HTTPError, TimeoutError) as exc:
            notes = str(exc)
        cur.execute("INSERT INTO dataset_health(dataset_name,checked_at,online_ok,notes) VALUES(?,?,?,?)", (name, now, ok, notes))
        checked.append({"name": name, "online_ok": ok, "notes": notes})
    con.commit()
    con.close()
    print(json.dumps({"checked": checked}, ensure_ascii=False, indent=2))
    return 0


def register_manual(name: str, scope: str, license_name: str, url: str, usage: str) -> int:
    payload = {"datasets": load_manual_extra()}
    payload["datasets"].append(
        {
            "name": name,
            "scope": scope,
            "license": license_name,
            "url": url,
            "usage": usage,
            "status": "manual_local_or_private",
        }
    )
    EXTRA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EXTRA_MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(EXTRA_MANIFEST))
    return 0


def build_training_plan(output: Path) -> int:
    rows = list(compile_manifest(COMPILED_MANIFEST).get("datasets", []))
    plan = {
        "version": "0.9.0",
        "strategy": [
            "1) Legally download datasets according to upstream terms",
            "2) Normalize labels and image sizes to unified training schema",
            "3) Build random balanced train/val/test splits",
            "4) Train detector baseline (FasterRCNN/RT-DETR) then refine",
            "5) Run policy A/B tests and safety/security checks",
            "6) Export edge-ready model for mobile/embedded runtime",
        ],
        "datasets": rows,
    }
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Dataset manager for Traffic AI Assist")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--online-health", action="store_true")
    p.add_argument("--compile-manifest", action="store_true")
    p.add_argument("--build-training-plan", type=Path)
    p.add_argument("--register-manual", action="store_true")
    p.add_argument("--name", type=str)
    p.add_argument("--scope", type=str)
    p.add_argument("--license", dest="license_name", type=str)
    p.add_argument("--url", type=str)
    p.add_argument("--usage", type=str)
    args = p.parse_args()

    if args.sync:
        return sync_datasets()
    if args.online_health:
        return check_online()
    if args.compile_manifest:
        compiled = compile_manifest(COMPILED_MANIFEST)
        print(json.dumps({"compiled": len(compiled.get("datasets", [])), "path": str(COMPILED_MANIFEST)}, ensure_ascii=False))
        return 0
    if args.build_training_plan:
        return build_training_plan(args.build_training_plan)
    if args.register_manual:
        required = [args.name, args.scope, args.license_name, args.url, args.usage]
        if any(x is None for x in required):
            raise SystemExit("--register-manual requires --name --scope --license --url --usage")
        return register_manual(args.name, args.scope, args.license_name, args.url, args.usage)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
