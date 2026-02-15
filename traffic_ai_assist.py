#!/usr/bin/env python3
"""
Traffic AI Assist - Real Agent Core
Version: 0.9.0
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

APP_NAME = "traffic-ai-assist"
SEMVER = "0.9.0"


class TrafficLightState(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class AlertChannel(str, Enum):
    NONE = "none"
    VISUAL = "visual"
    AUDIO = "audio"
    SIREN = "siren"


class PrivacyMode(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"


class InferenceBackend(str, Enum):
    INPUT = "input"
    NVIDIA_TRITON = "nvidia_triton"


@dataclass(frozen=True)
class TrafficLightCandidate:
    light_id: str
    state: TrafficLightState
    lane_ids: Tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class VehicleState:
    speed_kph: float
    lane_id: str
    crossed_stop_line: bool
    stationary_seconds: float


@dataclass(frozen=True)
class ExtraRoadContext:
    pedestrian_detected: bool
    road_signs: Tuple[str, ...]


@dataclass(frozen=True)
class FrameContext:
    route_id: str
    timestamp_ms: int
    candidates: Tuple[TrafficLightCandidate, ...]
    vehicle: VehicleState
    extra: ExtraRoadContext


@dataclass(frozen=True)
class Alert:
    key: str
    channel: AlertChannel


class Localization:
    MESSAGES: Dict[str, Dict[str, str]] = {
        "en": {
            "overspeed_red": "Warning: Red light ahead. Reduce speed now.",
            "red_crossed": "SIREN: Red light violation detected.",
            "green_wait": "Green light active. Move if safe.",
            "select_light": "Lane-light mismatch. Please select the correct signal.",
            "pedestrian": "Pedestrian detected. Drive with caution.",
            "ok": "No active warning.",
            "demo_started": "Demo mode started using free sample frames.",
        },
        "fa": {
            "overspeed_red": "هشدار: چراغ قرمز جلو است. سرعت را کم کنید.",
            "red_crossed": "آژیر: عبور از چراغ قرمز تشخیص داده شد.",
            "green_wait": "چراغ سبز است. در صورت ایمن بودن حرکت کنید.",
            "select_light": "تشخیص چراغ مرتبط با باند نامشخص است. لطفاً چراغ صحیح را انتخاب کنید.",
            "pedestrian": "عابر پیاده تشخیص داده شد. با احتیاط حرکت کنید.",
            "ok": "هشدار فعالی وجود ندارد.",
            "demo_started": "حالت دمو با داده نمونه رایگان شروع شد.",
        },
    }

    @classmethod
    def t(cls, lang: str, key: str) -> str:
        return cls.MESSAGES.get(lang, cls.MESSAGES["en"]).get(key, key)


class DatasetRegistry:
    MANIFEST_PATH = Path("data/external_datasets_manifest.json")
    DEFAULT_DATASETS: Tuple[Dict[str, str], ...] = (
        {
            "name": "BDD100K",
            "scope": "traffic lights, lanes, drivable area",
            "license": "Berkeley DeepDrive dataset terms",
            "url": "https://bdd-data.berkeley.edu/",
            "usage": "Demo taxonomy for lane + route behaviors.",
        },
        {
            "name": "Bosch Small Traffic Lights",
            "scope": "traffic light detection",
            "license": "Bosch dataset terms",
            "url": "https://hci.iwr.uni-heidelberg.de/node/6132",
            "usage": "Traffic-light candidate schema.",
        },
        {
            "name": "LISA Traffic Light Dataset",
            "scope": "traffic lights",
            "license": "Academic usage terms",
            "url": "https://datasetninja.com/lisa-traffic-light",
            "usage": "Signal state change examples.",
        },
    )

    @classmethod
    def datasets(cls) -> Tuple[Dict[str, str], ...]:
        if cls.MANIFEST_PATH.exists():
            try:
                obj = json.loads(cls.MANIFEST_PATH.read_text(encoding="utf-8"))
                rows = obj.get("datasets", [])
                cleaned: List[Dict[str, str]] = []
                for row in rows:
                    cleaned.append(
                        {
                            "name": str(row.get("name", "unknown")),
                            "scope": str(row.get("scope", "unknown")),
                            "license": str(row.get("license", "unknown")),
                            "url": str(row.get("url", "")),
                            "usage": str(row.get("usage", "")),
                        }
                    )
                return tuple(cleaned) if cleaned else cls.DEFAULT_DATASETS
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                return cls.DEFAULT_DATASETS
        return cls.DEFAULT_DATASETS


def free_demo_samples() -> Tuple[Dict[str, object], ...]:
    now = int(time.time() * 1000)
    return (
        {
            "sample_id": "demo-001",
            "dataset_name": "BDD100K",
            "dataset_license": "Berkeley DeepDrive dataset terms",
            "note": "red+overspeed",
            "frame": {
                "route_id": "berlin-city-center",
                "timestamp_ms": now,
                "candidates": [{"light_id": "TL-001", "state": "red", "lane_ids": ["lane-1"], "confidence": 0.94}],
                "vehicle": {"speed_kph": 52, "lane_id": "lane-1", "crossed_stop_line": False, "stationary_seconds": 0},
                "extra": {"pedestrian_detected": False, "road_signs": ["speed_limit_50"]},
            },
        },
        {
            "sample_id": "demo-002",
            "dataset_name": "Bosch Small Traffic Lights",
            "dataset_license": "Bosch dataset terms",
            "note": "red-line crossing",
            "frame": {
                "route_id": "berlin-city-center",
                "timestamp_ms": now + 1000,
                "candidates": [{"light_id": "TL-001", "state": "red", "lane_ids": ["lane-1"], "confidence": 0.95}],
                "vehicle": {"speed_kph": 22, "lane_id": "lane-1", "crossed_stop_line": True, "stationary_seconds": 0},
                "extra": {"pedestrian_detected": False, "road_signs": ["stop"]},
            },
        },
        {
            "sample_id": "demo-003",
            "dataset_name": "LISA Traffic Light Dataset",
            "dataset_license": "Academic usage terms",
            "note": "green-wait",
            "frame": {
                "route_id": "berlin-city-center",
                "timestamp_ms": now + 2000,
                "candidates": [{"light_id": "TL-002", "state": "green", "lane_ids": ["lane-1"], "confidence": 0.91}],
                "vehicle": {"speed_kph": 0, "lane_id": "lane-1", "crossed_stop_line": False, "stationary_seconds": 7},
                "extra": {"pedestrian_detected": False, "road_signs": ["go_straight"]},
            },
        },
        {
            "sample_id": "demo-004",
            "dataset_name": "Mapillary Traffic Sign Dataset",
            "dataset_license": "Mapillary Vistas terms",
            "note": "pedestrian-caution",
            "frame": {
                "route_id": "berlin-city-center",
                "timestamp_ms": now + 3000,
                "candidates": [{"light_id": "TL-003", "state": "green", "lane_ids": ["lane-2"], "confidence": 0.86}],
                "vehicle": {"speed_kph": 24, "lane_id": "lane-2", "crossed_stop_line": False, "stationary_seconds": 0},
                "extra": {"pedestrian_detected": True, "road_signs": ["pedestrian_crossing"]},
            },
        },
    )


class DB:
    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()
        self._migrate_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS lane_light_memory(route_id TEXT, lane_id TEXT, light_id TEXT, seen_count INTEGER, updated_at INTEGER, PRIMARY KEY(route_id,lane_id,light_id))")
        cur.execute("CREATE TABLE IF NOT EXISTS audit_log(ts INTEGER, event_type TEXT, payload TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS external_dataset_catalog(dataset_name TEXT PRIMARY KEY, scope TEXT, license TEXT, source_url TEXT, synced_at INTEGER)")
        cur.execute("CREATE TABLE IF NOT EXISTS demo_sample_frames(sample_id TEXT PRIMARY KEY, dataset_name TEXT, dataset_license TEXT, note TEXT, frame_payload TEXT, inserted_at INTEGER)")
        cur.execute("CREATE TABLE IF NOT EXISTS model_profile(model_key TEXT PRIMARY KEY, value TEXT, updated_at INTEGER)")
        self.conn.commit()

    def _migrate_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(external_dataset_catalog)")
        columns = {x[1] for x in cur.fetchall()}
        if "usage_reason" not in columns:
            cur.execute("ALTER TABLE external_dataset_catalog ADD COLUMN usage_reason TEXT NOT NULL DEFAULT ''")
        self.conn.commit()


class DatasetBootstrapper:
    def __init__(self, db: DB, logger: logging.Logger) -> None:
        self.db = db
        self.logger = logger

    def sync_catalog(self) -> None:
        cur = self.db.conn.cursor()
        now = int(time.time())
        for d in DatasetRegistry.datasets():
            cur.execute(
                """
                INSERT INTO external_dataset_catalog(dataset_name,scope,license,source_url,synced_at,usage_reason)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(dataset_name)
                DO UPDATE SET scope=excluded.scope, license=excluded.license, source_url=excluded.source_url, synced_at=excluded.synced_at, usage_reason=excluded.usage_reason
                """,
                (d["name"], d["scope"], d["license"], d["url"], now, d["usage"]),
            )
        self.db.conn.commit()

    def seed_demo_samples(self) -> int:
        cur = self.db.conn.cursor()
        count = 0
        now = int(time.time())
        for sample in free_demo_samples():
            cur.execute(
                """
                INSERT INTO demo_sample_frames(sample_id,dataset_name,dataset_license,note,frame_payload,inserted_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(sample_id)
                DO UPDATE SET dataset_name=excluded.dataset_name, dataset_license=excluded.dataset_license, note=excluded.note, frame_payload=excluded.frame_payload, inserted_at=excluded.inserted_at
                """,
                (sample["sample_id"], sample["dataset_name"], sample["dataset_license"], sample["note"], json.dumps(sample["frame"], ensure_ascii=False), now),
            )
            count += 1
        self.db.conn.commit()
        return count

    def iter_demo_frames(self) -> Iterator[str]:
        cur = self.db.conn.cursor()
        cur.execute("SELECT frame_payload FROM demo_sample_frames ORDER BY sample_id")
        for row in cur.fetchall():
            yield str(row[0])

    def fetch_remote_metadata(self, output_path: Path) -> None:
        out: List[Dict[str, str]] = []
        for d in DatasetRegistry.datasets():
            try:
                req = Request(d["url"], headers={"User-Agent": f"{APP_NAME}/{SEMVER}"}, method="GET")
                with urlopen(req, timeout=6) as resp:
                    snippet = resp.read(256).decode("utf-8", errors="replace")
                out.append({"name": d["name"], "url": d["url"], "sample": snippet})
            except (URLError, HTTPError, TimeoutError) as exc:
                out.append({"name": d["name"], "url": d["url"], "error": str(exc)})
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Dataset metadata exported to %s", output_path)


class LearningAgent:
    def __init__(self, db: DB, privacy_mode: PrivacyMode) -> None:
        self.db = db
        self.privacy_mode = privacy_mode

    def remember(self, route_id: str, lane_id: str, light_id: str) -> None:
        cur = self.db.conn.cursor()
        cur.execute(
            "INSERT INTO lane_light_memory(route_id,lane_id,light_id,seen_count,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(route_id,lane_id,light_id) DO UPDATE SET seen_count=seen_count+1, updated_at=excluded.updated_at",
            (route_id, lane_id, light_id, 1, int(time.time())),
        )
        self.db.conn.commit()

    def suggest(self, route_id: str, lane_id: str) -> Optional[str]:
        cur = self.db.conn.cursor()
        cur.execute("SELECT light_id FROM lane_light_memory WHERE route_id=? AND lane_id=? ORDER BY seen_count DESC,updated_at DESC LIMIT 1", (route_id, lane_id))
        row = cur.fetchone()
        return row[0] if row else None

    def log_event(self, event_type: str, payload: Dict[str, object]) -> None:
        safe = {"event_type": event_type, "mode": "strict"} if self.privacy_mode == PrivacyMode.STRICT else payload
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO audit_log(ts,event_type,payload) VALUES(?,?,?)", (int(time.time()), event_type, json.dumps(safe, ensure_ascii=False)))
        self.db.conn.commit()

    def set_model_value(self, key: str, value: str) -> None:
        cur = self.db.conn.cursor()
        cur.execute(
            "INSERT INTO model_profile(model_key,value,updated_at) VALUES(?,?,?) ON CONFLICT(model_key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, int(time.time())),
        )
        self.db.conn.commit()

    def get_model_value(self, key: str, default: str) -> str:
        cur = self.db.conn.cursor()
        cur.execute("SELECT value FROM model_profile WHERE model_key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default


class NvidiaTritonClient:
    def __init__(self, endpoint: str, model_name: str, logger: logging.Logger) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.logger = logger

    def infer(self, features: Dict[str, object]) -> Optional[Tuple[TrafficLightCandidate, ...]]:
        req = Request(f"{self.endpoint}/v2/models/{self.model_name}/infer", data=json.dumps(features).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=2) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warning("Triton unavailable; fallback to input candidates: %s", exc)
            return None
        out: List[TrafficLightCandidate] = []
        for cand in raw.get("candidates", []):
            try:
                out.append(TrafficLightCandidate(str(cand["light_id"]), TrafficLightState(str(cand.get("state", "unknown"))), tuple(str(x) for x in cand.get("lane_ids", [])), float(cand.get("confidence", 0.0))))
            except (KeyError, ValueError):
                continue
        return tuple(out) if out else None


class LaneAwareResolver:
    def __init__(self, agent: LearningAgent) -> None:
        self.agent = agent

    def resolve(self, ctx: FrameContext) -> Tuple[Optional[TrafficLightCandidate], bool]:
        lane_matches = [c for c in ctx.candidates if ctx.vehicle.lane_id in c.lane_ids]
        if lane_matches:
            lane_matches.sort(key=lambda x: x.confidence, reverse=True)
            pick = lane_matches[0]
            self.agent.remember(ctx.route_id, ctx.vehicle.lane_id, pick.light_id)
            return pick, False
        suggestion = self.agent.suggest(ctx.route_id, ctx.vehicle.lane_id)
        if suggestion:
            for c in ctx.candidates:
                if c.light_id == suggestion:
                    return c, False
        return None, True


class AlertEngine:
    def __init__(self, red_speed_threshold_kph: float = 25.0, green_idle_threshold_s: float = 4.0) -> None:
        self.red_speed_threshold_kph = red_speed_threshold_kph
        self.green_idle_threshold_s = green_idle_threshold_s

    def evaluate(self, light: Optional[TrafficLightCandidate], ctx: FrameContext) -> Alert:
        if ctx.extra.pedestrian_detected:
            return Alert("pedestrian", AlertChannel.VISUAL)
        if light is None:
            return Alert("select_light", AlertChannel.VISUAL)
        if light.state == TrafficLightState.RED and ctx.vehicle.crossed_stop_line:
            return Alert("red_crossed", AlertChannel.SIREN)
        if light.state == TrafficLightState.RED and ctx.vehicle.speed_kph > self.red_speed_threshold_kph:
            return Alert("overspeed_red", AlertChannel.AUDIO)
        if light.state == TrafficLightState.GREEN and ctx.vehicle.stationary_seconds > self.green_idle_threshold_s:
            return Alert("green_wait", AlertChannel.VISUAL)
        return Alert("ok", AlertChannel.NONE)


def parse_frame(raw: str) -> FrameContext:
    obj = json.loads(raw)
    cands = tuple(TrafficLightCandidate(str(c["light_id"]), TrafficLightState(str(c.get("state", "unknown"))), tuple(str(x) for x in c.get("lane_ids", [])), float(c.get("confidence", 0.0))) for c in obj.get("candidates", []))
    v = obj.get("vehicle", {})
    ex = obj.get("extra", {})
    return FrameContext(
        route_id=str(obj.get("route_id", "default")),
        timestamp_ms=int(obj.get("timestamp_ms", int(time.time() * 1000))),
        candidates=cands,
        vehicle=VehicleState(float(v.get("speed_kph", 0.0)), str(v.get("lane_id", "unknown")), bool(v.get("crossed_stop_line", False)), float(v.get("stationary_seconds", 0.0))),
        extra=ExtraRoadContext(bool(ex.get("pedestrian_detected", False)), tuple(str(x) for x in ex.get("road_signs", []))),
    )


def event_payload(alert: Alert, ctx: FrameContext, light: Optional[TrafficLightCandidate], ask: bool, lang: str, backend: InferenceBackend, triton_online: bool) -> Dict[str, object]:
    return {
        "app": APP_NAME,
        "version": SEMVER,
        "backend": backend.value,
        "triton_online": triton_online,
        "route_id": ctx.route_id,
        "light": {"id": light.light_id if light else None, "state": light.state.value if light else None},
        "lane_id": ctx.vehicle.lane_id,
        "alert": alert.key,
        "channel": alert.channel.value,
        "message": Localization.t(lang, alert.key),
        "ask_user_selection": ask,
        "map_overlay": {"show_camera_preview": True, "show_signal_state": True, "show_alert": alert.channel != AlertChannel.NONE},
        "extra": {"pedestrian_detected": ctx.extra.pedestrian_detected, "road_signs": list(ctx.extra.road_signs)},
    }


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if row:
                yield row


def process_stream(lines: Iterable[str], db: DB, lang: str, privacy_mode: PrivacyMode, backend: InferenceBackend, triton: Optional[NvidiaTritonClient], interactive: bool, logger: logging.Logger) -> int:
    agent = LearningAgent(db, privacy_mode)
    resolver = LaneAwareResolver(agent)
    engine = learned_alert_engine(db)
    for line in lines:
        try:
            ctx = parse_frame(line)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("invalid frame: %s", exc)
            continue
        triton_online = False
        if backend == InferenceBackend.NVIDIA_TRITON and triton is not None:
            inferred = triton.infer({"vehicle": {"lane_id": ctx.vehicle.lane_id, "speed_kph": ctx.vehicle.speed_kph}})
            if inferred:
                triton_online = True
                ctx = FrameContext(ctx.route_id, ctx.timestamp_ms, inferred, ctx.vehicle, ctx.extra)
        light, ask = resolver.resolve(ctx)
        if ask and interactive and ctx.candidates:
            print("Unable to resolve lane-specific light. Select index:")
            for i, c in enumerate(ctx.candidates):
                print(f"[{i}] id={c.light_id} state={c.state.value}")
            raw = sys.stdin.readline().strip()
            if raw.isdigit() and 0 <= int(raw) < len(ctx.candidates):
                light = ctx.candidates[int(raw)]
                ask = False
                agent.remember(ctx.route_id, ctx.vehicle.lane_id, light.light_id)
        alert = engine.evaluate(light, ctx)
        event = event_payload(alert, ctx, light, ask, lang, backend, triton_online)
        agent.log_event("alert", event)
        print(json.dumps(event, ensure_ascii=False))
    return 0


def run_ab_test(db: DB) -> Dict[str, object]:
    samples = [parse_frame(json.dumps(x["frame"], ensure_ascii=False)) for x in free_demo_samples()]
    resolver = LaneAwareResolver(LearningAgent(db, PrivacyMode.STRICT))
    a_engine = AlertEngine(red_speed_threshold_kph=25.0)
    b_engine = AlertEngine(red_speed_threshold_kph=35.0)
    out_a: List[str] = []
    out_b: List[str] = []
    for sample in samples:
        light, _ = resolver.resolve(sample)
        out_a.append(a_engine.evaluate(light, sample).key)
        out_b.append(b_engine.evaluate(light, sample).key)
    return {"version": SEMVER, "variant_a": out_a, "variant_b": out_b, "delta_alerts": sum(1 for a, b in zip(out_a, out_b) if a != b)}


def security_check() -> Dict[str, object]:
    import ast

    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            hits.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "pickle" and node.func.attr == "loads":
                hits.append("pickle.loads")
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess" and node.func.attr == "Popen":
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        hits.append("subprocess.Popen(shell=True)")
    return {
        "version": SEMVER,
        "passes": len(hits) == 0,
        "blocked_token_hits": sorted(set(hits)),
        "references": [
            "CISA secure-by-design principles",
            "NSA/CISA guidance for hardening",
            "BSI baseline controls",
            "Cyber Volunteer Resource Center secure coding references",
        ],
    }


def export_demo_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in free_demo_samples():
            f.write(json.dumps(sample["frame"], ensure_ascii=False) + "\n")


def menu_html() -> str:
    html = """<!doctype html>
<html><head><meta charset='utf-8'><title>__APP__ menu v__VER__</title>
<style>
body{font-family:Arial;background:#0f1117;color:#ebebeb;margin:0;padding:0}
header{padding:14px;background:#1a2236}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;padding:14px}
.card{background:#161d2e;border:1px solid #334;border-radius:10px;padding:12px}
a{color:#83b8ff;text-decoration:none;font-weight:bold}
small{color:#b9c6dd}
</style></head><body>
<header><strong>__APP__ Menu v__VER__</strong></header>
<div class='grid'>
  <div class='card'><a href='/dashboard'>Dashboard</a><br><small>Visual demo app with map + camera + lamp simulation.</small></div>
  <div class='card'><a href='/developer'>Developer Mode</a><br><small>Live camera + object guess labels.</small></div>
  <div class='card'><a href='/datasets'>Datasets</a><br><small>Visual card list with legal and count metadata.</small></div>
  <div class='card'><a href='/architecture'>Architecture</a><br><small>Mobile integration path for Android/iOS/CarPlay/AAOS.</small></div>
  <div class='card'><a href='/health'>Health</a><br><small>Service status endpoint.</small></div>
</div>
</body></html>"""
    return html.replace('__APP__', APP_NAME).replace('__VER__', SEMVER)


def dataset_stats() -> Dict[str, object]:
    rows = list(DatasetRegistry.datasets())
    known = [int(r.get("sample_count", 0)) for r in rows if str(r.get("sample_count", "")).isdigit()]
    return {
        "version": SEMVER,
        "dataset_count": len(rows),
        "known_sample_total": sum(known),
        "datasets": rows,
    }


def datasets_html() -> str:
    cards = []
    for row in DatasetRegistry.datasets():
        cards.append(
            f"<div class='card'><h3>{row['name']}</h3><p><b>Scope:</b> {row.get('scope','')}</p><p><b>License:</b> {row.get('license','')}</p><p><b>Usage:</b> {row.get('usage','')}</p><p><b>Sample count:</b> {row.get('sample_count','unknown')}</p><p><a href='{row.get('url','')}' target='_blank'>source</a></p></div>"
        )
    joined = "".join(cards)
    summary = dataset_stats()
    html = """<!doctype html><html><head><meta charset='utf-8'><title>__APP__ datasets</title>
<style>body{font-family:Arial;background:#0f1117;color:#eee;margin:0}header{background:#1a2236;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;padding:12px}.card{background:#161d2e;border:1px solid #334;padding:10px;border-radius:8px}a{color:#8ec1ff}</style>
</head><body><header><a href='/menu' style='color:#8ec1ff'>Menu</a> · Datasets · total datasets: __TOTAL__ · known samples: __KNOWN__</header><div class='grid'>__CARDS__</div></body></html>"""
    return html.replace('__APP__', APP_NAME).replace('__TOTAL__', str(summary['dataset_count'])).replace('__KNOWN__', str(summary['known_sample_total'])).replace('__CARDS__', joined)


def developer_html() -> str:
    html = """<!doctype html><html><head><meta charset='utf-8'><title>__APP__ developer v__VER__</title>
<style>body{font-family:Arial;background:#111;color:#eee;margin:0;padding:12px}video,canvas{width:100%;max-width:900px;background:#000;border:1px solid #333}pre{background:#0f1624;padding:8px}</style>
</head><body>
<h2>Developer Mode v__VER__</h2>
<p>License: MIT — Code generated with support from CODEX and CODEX CLI — Owner: Dr. Babak Sorkhpour</p>
<video id='cam' autoplay playsinline muted></video>
<p><button id='startCam'>Start camera</button> <button id='stopCam'>Stop camera</button></p>
<canvas id='cv' width='900' height='500'></canvas>
<pre id='det'>waiting...</pre>
<script>
const video=document.getElementById('cam');
let activeStream=null;
const canvas=document.getElementById('cv');
const ctx=canvas.getContext('2d');
const det=document.getElementById('det');
function detectObjects(){
  if(!video.srcObject){return;}
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  const img=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  let red=0,green=0,yellow=0,bright=0;
  for(let i=0;i<img.length;i+=16){
    const r=img[i],g=img[i+1],b=img[i+2];
    if(r>160 && r>g+25 && r>b+25) red++;
    if(g>140 && g>r+20 && g>b+20) green++;
    if(r>140 && g>120 && b<110) yellow++;
    if((r+g+b)>560) bright++;
  }
  const objects=[];
  if(red>600) objects.push('traffic_light_red');
  if(green>600) objects.push('traffic_light_green');
  if(yellow>400) objects.push('traffic_light_yellow');
  if(bright>1300) objects.push('vehicle_or_reflection');
  if(objects.length===0) objects.push('unknown_scene');
  det.textContent=JSON.stringify({objects,red,green,yellow,bright},null,2);
}
(async()=>{
  async function startCam(){
    try{
      activeStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});
      video.srcObject=activeStream;
      det.textContent='camera started';
    }catch(e){det.textContent='camera unavailable: '+e;}
  }
  function stopCam(){
    if(activeStream){activeStream.getTracks().forEach(t=>t.stop()); activeStream=null;}
    video.srcObject=null; det.textContent='camera stopped';
  }
  document.getElementById('startCam').onclick=startCam;
  document.getElementById('stopCam').onclick=stopCam;
  await startCam();
  setInterval(detectObjects,700);
})();
</script>
</body></html>"""
    return html.replace('__APP__', APP_NAME).replace('__VER__', SEMVER)


def dashboard_html() -> str:
    options = ''.join([f"<option value=\"{r.get('name','')}\">{r.get('name','')}</option>" for r in DatasetRegistry.datasets()])
    html = """<!doctype html>
<html><head><meta charset='utf-8'><title>__APP__ dashboard v__VER__</title>
<style>
body{font-family:Arial;background:#111;color:#eee;margin:0}
header{padding:12px;background:#1b2233}
nav a{color:#8ec1ff;margin-right:12px}
.wrap{padding:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{background:#1a1f2e;border:1px solid #334;padding:10px;border-radius:8px}
video{width:100%;height:240px;border:0;background:#000}
#mapWrap{position:relative;height:320px}
#gm{width:100%;height:320px;border:0}
#lamp{position:absolute;width:24px;height:24px;border-radius:50%;border:2px solid #fff;left:75%;top:35%;background:#777;box-shadow:0 0 18px #000}
pre{white-space:pre-wrap;word-break:break-word;background:#0d1119;padding:8px;border-radius:6px}
button{padding:8px 12px}
</style>
</head><body>
<header><strong>__APP__ v__VER__</strong> · Owner: <a style='color:#8ec1ff' href='https://x.com/Drbabakskr'>Dr. Babak Sorkhpour</a>
<nav><a href='/menu'>Menu</a><a href='/datasets'>Datasets</a><a href='/developer'>Developer</a><a href='/architecture'>Architecture</a><a href='/health'>Health</a></nav></header>
<div class='wrap'><div class='grid'>
<div class='card'><h3>Camera (browser)</h3><video id='cam' autoplay playsinline muted></video><p><button id='startCam'>Start</button> <button id='stopCam'>Stop</button></p><small id='camstate'>Requesting camera…</small></div>
<div class='card'><h3>Google Map + Traffic Lamp</h3><div id='mapWrap'><iframe id='gm' src='https://maps.google.com/maps?q=Berlin&z=13&output=embed'></iframe><div id='lamp'></div></div><pre id='vision'>vision: pending</pre></div>
<div class='card'><h3>Random dataset demo</h3><label>Dataset:</label><select id='ds'><option value=''>Any dataset</option>__OPTIONS__</select> <button id='runDemo'>Run random sample</button><pre id='sampleInfo'>No sample selected</pre></div>
<div class='card'><h3>Agent output</h3><pre id='demoEvents'>No event yet</pre></div>
</div></div>
<script>
function lampColor(state){if(state==='red')return '#e53935'; if(state==='green')return '#43a047'; if(state==='yellow')return '#fdd835'; return '#777';}
function detectLampVisual(){
  const color=getComputedStyle(document.getElementById('lamp')).backgroundColor;
  if(color.includes('229, 57, 53')) return 'red';
  if(color.includes('67, 160, 71')) return 'green';
  if(color.includes('253, 216, 53')) return 'yellow';
  return 'unknown';
}
(async()=>{
  let stream=null;
  async function startCam(){
    try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});document.getElementById('cam').srcObject=stream;document.getElementById('camstate').textContent='Camera connected';}
    catch(e){document.getElementById('camstate').textContent='Camera unavailable in this browser/device: '+e;}
  }
  function stopCam(){ if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;} document.getElementById('cam').srcObject=null; document.getElementById('camstate').textContent='Camera stopped'; }
  document.getElementById('startCam').onclick=startCam;
  document.getElementById('stopCam').onclick=stopCam;
  await startCam();
  document.getElementById('runDemo').onclick=async()=>{
    const dataset=document.getElementById('ds').value;
    const r=await fetch('/demo/random',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_name:dataset})}); const d=await r.json();
    document.getElementById('sampleInfo').textContent=JSON.stringify(d.sample,null,2);
    document.getElementById('demoEvents').textContent=JSON.stringify(d.event,null,2);
    const st=(d.event.light && d.event.light.state) ? d.event.light.state : 'unknown';
    document.getElementById('lamp').style.background=lampColor(st);
    document.getElementById('vision').textContent='visual detection: '+detectLampVisual();
    const route=(d.sample && d.sample.frame && d.sample.frame.route_id) ? d.sample.frame.route_id : 'Berlin';
    document.getElementById('gm').src='https://maps.google.com/maps?q='+encodeURIComponent(route)+'&z=13&output=embed';
  };
})();
</script></body></html>"""
    return html.replace('__APP__', APP_NAME).replace('__VER__', SEMVER).replace('__OPTIONS__', options)


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class APIServer:
    def __init__(self, db: DB, host: str, port: int, logger: logging.Logger) -> None:
        self.db = db
        self.host = host
        self.port = port
        self.logger = logger

    def serve(self) -> int:
        db = self.db
        logger = self.logger

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, payload: object, ctype: str = "application/json") -> None:
                body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload, ensure_ascii=False)
                raw = body if isinstance(body, bytes) else body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", ctype + "; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _json_body(self) -> Dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}

            def do_GET(self) -> None:  # noqa
                if self.path in ("/", "/dashboard"):
                    self._send(HTTPStatus.OK, dashboard_html(), "text/html")
                    return
                if self.path == "/menu":
                    self._send(HTTPStatus.OK, menu_html(), "text/html")
                    return
                if self.path == "/developer":
                    self._send(HTTPStatus.OK, developer_html(), "text/html")
                    return
                if self.path == "/datasets":
                    self._send(HTTPStatus.OK, datasets_html(), "text/html")
                    return
                if self.path == "/dataset-stats":
                    self._send(HTTPStatus.OK, dataset_stats())
                    return
                if self.path == "/health":
                    self._send(HTTPStatus.OK, {"app": APP_NAME, "version": SEMVER, "status": "ok"})
                    return
                if self.path == "/architecture":
                    self._send(HTTPStatus.OK, {"mobile_plan": ["iOS/Android native camera feed", "on-device model inference (CoreML/TFLite/ONNX)", "AAOS/CarPlay projection adapter"], "current_demo": "Visual map + traffic lamp overlay + rule-based AI agent + developer camera mode"})
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found", "path": self.path})

            def do_POST(self) -> None:  # noqa
                if self.path == "/demo/random":
                    params = self._json_body()
                    selected_dataset = str(params.get("dataset_name", "")).strip()
                    bs = DatasetBootstrapper(db, logger)
                    bs.sync_catalog()
                    bs.seed_demo_samples()
                    cur = db.conn.cursor()
                    if selected_dataset:
                        cur.execute("SELECT dataset_name, frame_payload FROM demo_sample_frames WHERE dataset_name=?", (selected_dataset,))
                        fetched = cur.fetchall()
                        if not fetched:
                            cur.execute("SELECT dataset_name, frame_payload FROM demo_sample_frames")
                            fetched = cur.fetchall()
                    else:
                        cur.execute("SELECT dataset_name, frame_payload FROM demo_sample_frames")
                        fetched = cur.fetchall()
                    if not fetched:
                        self._send(HTTPStatus.BAD_REQUEST, {"error": "no demo frames"})
                        return
                    dataset_name, frame_raw = random.choice(fetched)
                    frame_obj = json.loads(str(frame_raw))
                    sample = {"frame": frame_obj, "source": "demo_sample_frames", "selected_dataset": selected_dataset or "any", "dataset_name": str(dataset_name)}
                    ctx = parse_frame(json.dumps(frame_obj, ensure_ascii=False))
                    agent = LearningAgent(db, PrivacyMode.STRICT)
                    resolver = LaneAwareResolver(agent)
                    engine = learned_alert_engine(db)
                    light, ask = resolver.resolve(ctx)
                    event = event_payload(engine.evaluate(light, ctx), ctx, light, ask, "en", InferenceBackend.INPUT, False)
                    agent.log_event("alert", event)
                    self._send(HTTPStatus.OK, {"sample": sample, "event": event})
                    return
                if self.path == "/demo/run":
                    bs = DatasetBootstrapper(db, logger)
                    bs.sync_catalog()
                    seeded = bs.seed_demo_samples()
                    events: List[Dict[str, object]] = []
                    agent = LearningAgent(db, PrivacyMode.STRICT)
                    resolver = LaneAwareResolver(agent)
                    engine = learned_alert_engine(db)
                    for row in bs.iter_demo_frames():
                        ctx = parse_frame(row)
                        light, ask = resolver.resolve(ctx)
                        event = event_payload(engine.evaluate(light, ctx), ctx, light, ask, "en", InferenceBackend.INPUT, False)
                        agent.log_event("alert", event)
                        events.append(event)
                    self._send(HTTPStatus.OK, {"seeded": seeded, "events": events})
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found", "path": self.path})

        try:
            httpd = ReusableHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            self.logger.error("Server failed on %s:%s (%s). Use another port with --port.", self.host, self.port, exc)
            return 98
        self.logger.info("Dashboard/API running on http://%s:%s", self.host, self.port)
        httpd.serve_forever()
        return 0


def generate_gherkin_from_runtime(path: Path) -> None:
    feature = f"""# Version: {SEMVER}
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

Feature: Traffic AI Assist generated BDD
  Scenario: Agent training refinement
    Given seeded demo samples and stored telemetry
    When train-agent command runs
    Then learned model profile is updated for alert thresholds

  Scenario: Dataset schema migration
    Given existing sqlite database may miss usage_reason column
    When app initializes DB
    Then missing columns are migrated automatically

  Scenario: Random dataset demo reaction
    Given free demo samples are available
    When user requests random demo with optional dataset filter
    Then app visualizes lamp and emits warning event based on rules

  Scenario: Visual dashboard and menus
    Given serve mode is active
    When user opens menu and dashboard
    Then visual cards, map, camera block, and demo controls are visible

  Scenario: Developer mode live detection
    Given developer mode is opened
    When browser camera stream is active
    Then client-side object guesses are shown continuously
"""
    path.write_text(feature, encoding="utf-8")


def train_agent_model(db: DB, epochs: int = 1) -> Dict[str, object]:
    agent = LearningAgent(db, PrivacyMode.STRICT)
    cur = db.conn.cursor()
    cur.execute("SELECT frame_payload FROM demo_sample_frames")
    rows = [r[0] for r in cur.fetchall()]
    red_speeds: List[float] = []
    for raw in rows:
        try:
            obj = json.loads(str(raw))
            cands = obj.get("candidates", [])
            vehicle = obj.get("vehicle", {})
            if cands and str(cands[0].get("state", "unknown")) == "red":
                red_speeds.append(float(vehicle.get("speed_kph", 0.0)))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    if red_speeds:
        avg = sum(red_speeds) / len(red_speeds)
        threshold = max(20.0, min(45.0, avg * 0.72))
    else:
        threshold = 25.0
    for _ in range(max(1, int(epochs))):
        agent.set_model_value("red_speed_threshold_kph", f"{threshold:.2f}")
    return {"version": SEMVER, "epochs": epochs, "trained_rows": len(rows), "red_speed_threshold_kph": threshold}


def learned_alert_engine(db: DB) -> AlertEngine:
    agent = LearningAgent(db, PrivacyMode.STRICT)
    val = agent.get_model_value("red_speed_threshold_kph", "25.0")
    try:
        threshold = float(val)
    except ValueError:
        threshold = 25.0
    return AlertEngine(red_speed_threshold_kph=threshold)


def run_demo_mode(db: DB, lang: str, privacy_mode: PrivacyMode, logger: logging.Logger) -> int:
    bs = DatasetBootstrapper(db, logger)
    bs.sync_catalog()
    seeded = bs.seed_demo_samples()
    print(json.dumps({"app": APP_NAME, "version": SEMVER, "demo": Localization.t(lang, "demo_started"), "seeded_frames": seeded}, ensure_ascii=False))
    return process_stream(bs.iter_demo_frames(), db, lang, privacy_mode, InferenceBackend.INPUT, None, False, logger)


def run_self_test() -> int:
    db_path = Path(".traffic_ai_test.sqlite3")
    if db_path.exists():
        db_path.unlink()
    db = DB(db_path)
    out = run_ab_test(db)
    sec = security_check()
    ok = out["delta_alerts"] >= 0 and sec["passes"]
    print("Self-test:", "PASS" if ok else "FAIL")
    db_path.unlink(missing_ok=True)
    return 0 if ok else 1


def configure_logger(debug: bool, log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream_h)
    if log_file:
        file_h = logging.FileHandler(log_file, encoding="utf-8")
        file_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_h)
    return logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traffic AI Assist v0.9.0")
    p.add_argument("--version", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset-manifest", action="store_true")
    p.add_argument("--sync-datasets", action="store_true")
    p.add_argument("--fetch-dataset-metadata", type=Path)
    p.add_argument("--export-demo-sample", type=Path)
    p.add_argument("--demo-mode", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--generate-bdd", type=Path)
    p.add_argument("--ab-test", action="store_true")
    p.add_argument("--security-check", action="store_true")
    p.add_argument("--train-agent", action="store_true")
    p.add_argument("--epochs", type=int, default=1)

    p.add_argument("--input", type=Path)
    p.add_argument("--db", type=Path, default=Path("./traffic_ai.sqlite3"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--lang", default="en", choices=["en", "fa"])
    p.add_argument("--privacy", default="strict", choices=["strict", "balanced"])
    p.add_argument("--inference-backend", default="input", choices=["input", "nvidia_triton"])
    p.add_argument("--nvidia-endpoint", default="http://127.0.0.1:8000")
    p.add_argument("--nvidia-model", default="traffic_light_detector")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log-file", type=Path)
    return p


def main() -> int:
    args = build_parser().parse_args()
    logger = configure_logger(args.debug, args.log_file)

    if args.version:
        print(f"{APP_NAME} {SEMVER}")
        return 0
    if args.self_test:
        return run_self_test()

    db = DB(args.db)
    bootstrapper = DatasetBootstrapper(db, logger)

    if args.dataset_manifest:
        print(json.dumps({"version": SEMVER, "datasets": list(DatasetRegistry.datasets())}, ensure_ascii=False, indent=2))
        return 0
    if args.sync_datasets:
        bootstrapper.sync_catalog()
        print("dataset catalog synced")
        return 0
    if args.fetch_dataset_metadata:
        bootstrapper.fetch_remote_metadata(args.fetch_dataset_metadata)
        print(str(args.fetch_dataset_metadata))
        return 0
    if args.export_demo_sample:
        export_demo_file(args.export_demo_sample)
        print(str(args.export_demo_sample))
        return 0
    if args.generate_bdd:
        generate_gherkin_from_runtime(args.generate_bdd)
        print(str(args.generate_bdd))
        return 0
    if args.ab_test:
        print(json.dumps(run_ab_test(db), ensure_ascii=False, indent=2))
        return 0
    if args.security_check:
        result = security_check()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passes"] else 1
    if args.train_agent:
        bs = DatasetBootstrapper(db, logger)
        bs.seed_demo_samples()
        print(json.dumps(train_agent_model(db, args.epochs), ensure_ascii=False, indent=2))
        return 0
    if args.demo_mode:
        return run_demo_mode(db, args.lang, PrivacyMode(args.privacy), logger)
    if args.serve:
        return APIServer(db, args.host, args.port, logger).serve()

    if args.input is None:
        print("Error: --input is required unless --demo-mode/--serve is used.", file=sys.stderr)
        return 2
    backend = InferenceBackend(args.inference_backend)
    triton = NvidiaTritonClient(args.nvidia_endpoint, args.nvidia_model, logger) if backend == InferenceBackend.NVIDIA_TRITON else None
    return process_stream(iter_lines(args.input), db, args.lang, PrivacyMode(args.privacy), backend, triton, args.interactive, logger)


if __name__ == "__main__":
    raise SystemExit(main())
