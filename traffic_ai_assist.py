#!/usr/bin/env python3
"""
Traffic AI Assist - Real Agent Core
Version: 0.4.1
License: MIT

Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

import argparse
import json
import logging
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
SEMVER = "0.4.1"


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
    DATASETS: Tuple[Dict[str, str], ...] = (
        {"name": "BDD100K", "scope": "traffic lights, lanes, drivable area", "license": "Berkeley DeepDrive dataset terms", "url": "https://bdd-data.berkeley.edu/", "usage": "Demo taxonomy and route/lane examples."},
        {"name": "Bosch Small Traffic Lights", "scope": "traffic light detection", "license": "Bosch dataset terms", "url": "https://hci.iwr.uni-heidelberg.de/node/6132", "usage": "Traffic-light candidate structure examples."},
        {"name": "LISA Traffic Light Dataset", "scope": "traffic lights", "license": "Academic usage terms", "url": "https://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html", "usage": "State-transition scenario examples."},
        {"name": "Mapillary Traffic Sign Dataset", "scope": "road signs", "license": "Mapillary Vistas terms", "url": "https://www.mapillary.com/dataset/vistas", "usage": "Road-sign context labels in extra metadata."},
    )

    @classmethod
    def to_json(cls) -> str:
        return json.dumps({"version": SEMVER, "datasets": list(cls.DATASETS)}, ensure_ascii=False, indent=2)


def free_demo_samples() -> Tuple[Dict[str, object], ...]:
    now = int(time.time() * 1000)
    return (
        {"sample_id": "demo-001", "dataset_name": "BDD100K", "dataset_license": "Berkeley DeepDrive dataset terms", "note": "Red + overspeed", "frame": {"route_id": "berlin-city-center", "timestamp_ms": now, "candidates": [{"light_id": "TL-001", "state": "red", "lane_ids": ["lane-1"], "confidence": 0.94}], "vehicle": {"speed_kph": 52, "lane_id": "lane-1", "crossed_stop_line": False, "stationary_seconds": 0}, "extra": {"pedestrian_detected": False, "road_signs": ["speed_limit_50"]}}},
        {"sample_id": "demo-002", "dataset_name": "Bosch Small Traffic Lights", "dataset_license": "Bosch dataset terms", "note": "Red crossing", "frame": {"route_id": "berlin-city-center", "timestamp_ms": now + 1000, "candidates": [{"light_id": "TL-001", "state": "red", "lane_ids": ["lane-1"], "confidence": 0.95}], "vehicle": {"speed_kph": 20, "lane_id": "lane-1", "crossed_stop_line": True, "stationary_seconds": 0}, "extra": {"pedestrian_detected": False, "road_signs": ["stop"]}}},
        {"sample_id": "demo-003", "dataset_name": "LISA Traffic Light Dataset", "dataset_license": "Academic usage terms", "note": "Green wait", "frame": {"route_id": "berlin-city-center", "timestamp_ms": now + 2000, "candidates": [{"light_id": "TL-002", "state": "green", "lane_ids": ["lane-1"], "confidence": 0.92}], "vehicle": {"speed_kph": 0, "lane_id": "lane-1", "crossed_stop_line": False, "stationary_seconds": 6}, "extra": {"pedestrian_detected": False, "road_signs": ["go_straight"]}}},
        {"sample_id": "demo-004", "dataset_name": "Mapillary Traffic Sign Dataset", "dataset_license": "Mapillary Vistas terms", "note": "Pedestrian caution", "frame": {"route_id": "berlin-city-center", "timestamp_ms": now + 3000, "candidates": [{"light_id": "TL-003", "state": "green", "lane_ids": ["lane-2"], "confidence": 0.86}], "vehicle": {"speed_kph": 24, "lane_id": "lane-2", "crossed_stop_line": False, "stationary_seconds": 0}, "extra": {"pedestrian_detected": True, "road_signs": ["pedestrian_crossing"]}}},
    )


class DB:
    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()
        self._migrate_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS lane_light_memory(route_id TEXT, lane_id TEXT, light_id TEXT, seen_count INTEGER DEFAULT 1, updated_at INTEGER, PRIMARY KEY(route_id,lane_id,light_id))")
        cur.execute("CREATE TABLE IF NOT EXISTS audit_log(ts INTEGER, event_type TEXT, payload TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS external_dataset_catalog(dataset_name TEXT PRIMARY KEY, scope TEXT NOT NULL, license TEXT NOT NULL, source_url TEXT NOT NULL, synced_at INTEGER NOT NULL)")
        cur.execute("CREATE TABLE IF NOT EXISTS demo_sample_frames(sample_id TEXT PRIMARY KEY, dataset_name TEXT NOT NULL, dataset_license TEXT NOT NULL, note TEXT NOT NULL, frame_payload TEXT NOT NULL, inserted_at INTEGER NOT NULL)")
        self.conn.commit()

    def _migrate_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(external_dataset_catalog)")
        cols = {row[1] for row in cur.fetchall()}
        if "usage_reason" not in cols:
            cur.execute("ALTER TABLE external_dataset_catalog ADD COLUMN usage_reason TEXT NOT NULL DEFAULT ''")
        self.conn.commit()


class DatasetBootstrapper:
    def __init__(self, db: DB, logger: logging.Logger) -> None:
        self.db = db
        self.logger = logger

    def sync_catalog(self) -> None:
        cur = self.db.conn.cursor()
        now = int(time.time())
        for d in DatasetRegistry.DATASETS:
            cur.execute(
                """
                INSERT INTO external_dataset_catalog(dataset_name, scope, license, source_url, synced_at, usage_reason)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(dataset_name)
                DO UPDATE SET scope=excluded.scope, license=excluded.license, source_url=excluded.source_url, synced_at=excluded.synced_at, usage_reason=excluded.usage_reason
                """,
                (d["name"], d["scope"], d["license"], d["url"], now, d["usage"]),
            )
        self.db.conn.commit()

    def fetch_remote_metadata(self, output_path: Path) -> None:
        out: List[Dict[str, str]] = []
        for d in DatasetRegistry.DATASETS:
            try:
                req = Request(d["url"], headers={"User-Agent": f"{APP_NAME}/{SEMVER}"}, method="GET")
                with urlopen(req, timeout=8) as resp:
                    snippet = resp.read(256).decode("utf-8", errors="replace")
                out.append({"name": d["name"], "url": d["url"], "sample": snippet})
            except (URLError, HTTPError, TimeoutError) as exc:
                out.append({"name": d["name"], "url": d["url"], "error": str(exc)})
        output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Dataset metadata exported to %s", output_path)

    def seed_demo_samples(self) -> int:
        cur = self.db.conn.cursor()
        now = int(time.time())
        count = 0
        for sample in free_demo_samples():
            cur.execute(
                """
                INSERT INTO demo_sample_frames(sample_id, dataset_name, dataset_license, note, frame_payload, inserted_at)
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
        cur.execute("SELECT light_id FROM lane_light_memory WHERE route_id=? AND lane_id=? ORDER BY seen_count DESC, updated_at DESC LIMIT 1", (route_id, lane_id))
        row = cur.fetchone()
        return row[0] if row else None

    def log_event(self, event_type: str, payload: Dict[str, object]) -> None:
        safe = {"mode": "strict", "event_type": event_type} if self.privacy_mode == PrivacyMode.STRICT else payload
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO audit_log(ts,event_type,payload) VALUES(?,?,?)", (int(time.time()), event_type, json.dumps(safe, ensure_ascii=False)))
        self.db.conn.commit()


class NvidiaTritonClient:
    def __init__(self, endpoint: str, model_name: str, logger: logging.Logger) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.logger = logger

    def infer(self, inference_features: Dict[str, object]) -> Optional[Tuple[TrafficLightCandidate, ...]]:
        req = Request(f"{self.endpoint}/v2/models/{self.model_name}/infer", data=json.dumps(inference_features).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=2) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warning("Triton infer failed; falling back to input candidates: %s", exc)
            return None
        out: List[TrafficLightCandidate] = []
        for c in raw.get("candidates", []):
            try:
                out.append(TrafficLightCandidate(str(c["light_id"]), TrafficLightState(str(c.get("state", "unknown"))), tuple(str(x) for x in c.get("lane_ids", [])), float(c.get("confidence", 0.0))))
            except (KeyError, ValueError):
                continue
        return tuple(out) if out else None


class LaneAwareResolver:
    def __init__(self, agent: LearningAgent) -> None:
        self.agent = agent

    def resolve(self, ctx: FrameContext) -> Tuple[Optional[TrafficLightCandidate], bool]:
        matches = [c for c in ctx.candidates if ctx.vehicle.lane_id in c.lane_ids]
        if matches:
            matches.sort(key=lambda x: x.confidence, reverse=True)
            selected = matches[0]
            self.agent.remember(ctx.route_id, ctx.vehicle.lane_id, selected.light_id)
            return selected, False
        suggested = self.agent.suggest(ctx.route_id, ctx.vehicle.lane_id)
        if suggested:
            for c in ctx.candidates:
                if c.light_id == suggested:
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
    candidates = tuple(TrafficLightCandidate(str(c["light_id"]), TrafficLightState(str(c.get("state", "unknown"))), tuple(str(x) for x in c.get("lane_ids", [])), float(c.get("confidence", 0.0))) for c in obj.get("candidates", []))
    vehicle = obj.get("vehicle", {})
    extra = obj.get("extra", {})
    return FrameContext(
        route_id=str(obj.get("route_id", "default")),
        timestamp_ms=int(obj.get("timestamp_ms", int(time.time() * 1000))),
        candidates=candidates,
        vehicle=VehicleState(float(vehicle.get("speed_kph", 0.0)), str(vehicle.get("lane_id", "unknown")), bool(vehicle.get("crossed_stop_line", False)), float(vehicle.get("stationary_seconds", 0.0))),
        extra=ExtraRoadContext(bool(extra.get("pedestrian_detected", False)), tuple(str(x) for x in extra.get("road_signs", []))),
    )


def to_event(alert: Alert, ctx: FrameContext, light: Optional[TrafficLightCandidate], ask: bool, lang: str, backend: InferenceBackend, triton_online: bool) -> Dict[str, object]:
    return {
        "app": APP_NAME,
        "version": SEMVER,
        "backend": backend.value,
        "triton_online": triton_online,
        "route_id": ctx.route_id,
        "lane_id": ctx.vehicle.lane_id,
        "light": {"id": light.light_id if light else None, "state": light.state.value if light else None},
        "alert": alert.key,
        "channel": alert.channel.value,
        "message": Localization.t(lang, alert.key),
        "ask_user_selection": ask,
        "map_overlay": {"show_camera_preview": True, "show_signal_state": True, "show_alert": alert.channel != AlertChannel.NONE},
        "extra": {"pedestrian_detected": ctx.extra.pedestrian_detected, "road_signs": list(ctx.extra.road_signs)},
    }


def process_stream(lines: Iterable[str], db: DB, lang: str, privacy_mode: PrivacyMode, backend: InferenceBackend, triton: Optional[NvidiaTritonClient], interactive: bool, logger: logging.Logger) -> int:
    agent = LearningAgent(db, privacy_mode)
    resolver = LaneAwareResolver(agent)
    engine = AlertEngine()

    for line in lines:
        try:
            ctx = parse_frame(line)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Invalid frame: %s", exc)
            continue

        triton_online = False
        if backend == InferenceBackend.NVIDIA_TRITON and triton is not None:
            inferred = triton.infer({"vehicle": {"lane_id": ctx.vehicle.lane_id, "speed_kph": ctx.vehicle.speed_kph}, "meta": {"route_id": ctx.route_id, "timestamp_ms": ctx.timestamp_ms}})
            if inferred:
                ctx = FrameContext(ctx.route_id, ctx.timestamp_ms, inferred, ctx.vehicle, ctx.extra)
                triton_online = True

        light, ask = resolver.resolve(ctx)
        if ask and interactive and ctx.candidates:
            print("Unable to resolve lane-specific light. Select index:")
            for i, c in enumerate(ctx.candidates):
                print(f"[{i}] id={c.light_id} state={c.state.value} lanes={','.join(c.lane_ids)}")
            selected_raw = sys.stdin.readline().strip()
            if selected_raw.isdigit() and 0 <= int(selected_raw) < len(ctx.candidates):
                light = ctx.candidates[int(selected_raw)]
                ask = False
                agent.remember(ctx.route_id, ctx.vehicle.lane_id, light.light_id)

        alert = engine.evaluate(light, ctx)
        event = to_event(alert, ctx, light, ask, lang, backend, triton_online)
        agent.log_event("alert", event)
        print(json.dumps(event, ensure_ascii=False))
    return 0


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if row:
                yield row


def export_demo_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in free_demo_samples():
            handle.write(json.dumps(sample["frame"], ensure_ascii=False) + "\n")


def dashboard_html() -> str:
    return """<!doctype html>
<html><head><meta charset='utf-8'><title>traffic-ai-assist dashboard</title>
<style>
body{font-family:Arial;background:#0f1117;color:#e7e7e7;margin:0}
header{background:#161c2b;padding:12px}
nav a{color:#7cb1ff;margin-right:12px}
.wrap{padding:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{background:#151b29;border:1px solid #2f3d5c;border-radius:8px;padding:10px}
iframe{width:100%;height:320px;border:0;background:#000}
pre{white-space:pre-wrap}
button{padding:8px 12px}
</style>
</head><body>
<header><strong>traffic-ai-assist Dashboard v0.4.1</strong> · Owner: <a style='color:#7cb1ff' href='https://x.com/Drbabakskr'>Dr. Babak Sorkhpour</a>
<nav><a href='/'>Dashboard</a><a href='/health'>Health</a><a href='/datasets'>Datasets</a><a href='/menu'>Menu</a></nav></header>
<div class='wrap'>
<div class='grid'>
<div class='card'><h3>Maps</h3><iframe src='https://maps.google.com/maps?q=Berlin&z=12&output=embed'></iframe></div>
<div class='card'><h3>Quick Actions</h3>
<p>Run demo from API:</p>
<pre>curl -X POST http://127.0.0.1:8080/demo/run</pre>
<p>Check menus:</p>
<pre>curl http://127.0.0.1:8080/menu</pre>
</div>
</div>
</div>
</body></html>"""


class APIServer:
    def __init__(self, db: DB, host: str, port: int, logger: logging.Logger) -> None:
        self.db = db
        self.host = host
        self.port = port
        self.logger = logger

    def serve(self) -> None:
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

            def do_GET(self) -> None:  # noqa
                if self.path == "/" or self.path == "/dashboard":
                    self._send(HTTPStatus.OK, dashboard_html(), "text/html")
                    return
                if self.path == "/health":
                    self._send(HTTPStatus.OK, {"app": APP_NAME, "version": SEMVER, "status": "ok"})
                    return
                if self.path == "/datasets":
                    self._send(HTTPStatus.OK, json.loads(DatasetRegistry.to_json()))
                    return
                if self.path == "/menu":
                    self._send(HTTPStatus.OK, {"menus": ["Dashboard", "Health", "Datasets", "Run Demo"]})
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found", "path": self.path})

            def do_POST(self) -> None:  # noqa
                if self.path == "/demo/run":
                    bootstrap = DatasetBootstrapper(db, logger)
                    bootstrap.sync_catalog()
                    seeded = bootstrap.seed_demo_samples()
                    events: List[Dict[str, object]] = []
                    agent = LearningAgent(db, PrivacyMode.STRICT)
                    resolver = LaneAwareResolver(agent)
                    engine = AlertEngine()
                    for line in bootstrap.iter_demo_frames():
                        ctx = parse_frame(line)
                        light, ask = resolver.resolve(ctx)
                        alert = engine.evaluate(light, ctx)
                        ev = to_event(alert, ctx, light, ask, "en", InferenceBackend.INPUT, False)
                        agent.log_event("alert", ev)
                        events.append(ev)
                    self._send(HTTPStatus.OK, {"seeded": seeded, "events": events})
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found", "path": self.path})

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        logger.info("Dashboard/API running on http://%s:%s", self.host, self.port)
        httpd.serve_forever()


def run_demo_mode(db: DB, lang: str, privacy_mode: PrivacyMode, logger: logging.Logger) -> int:
    bootstrap = DatasetBootstrapper(db, logger)
    bootstrap.sync_catalog()
    seeded = bootstrap.seed_demo_samples()
    print(json.dumps({"app": APP_NAME, "version": SEMVER, "demo": Localization.t(lang, "demo_started"), "seeded_frames": seeded}, ensure_ascii=False))
    return process_stream(bootstrap.iter_demo_frames(), db, lang, privacy_mode, InferenceBackend.INPUT, None, False, logger)


def configure_logger(debug: bool, log_file: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


BDD_FEATURE = """
Feature: Smart lane-aware traffic AI agent with free demo dataset
  Scenario: Demo mode seeds and runs
    Given demo samples are available
    When demo mode runs
    Then alert events are emitted and saved
""".strip()


def run_self_test() -> int:
    db_path = Path("./.traffic_ai_test.sqlite3")
    if db_path.exists():
        db_path.unlink()
    db = DB(db_path)
    bootstrap = DatasetBootstrapper(db, logging.getLogger(APP_NAME))
    seeded = bootstrap.seed_demo_samples()
    ok = seeded >= 4
    print("BDD Feature:\n" + BDD_FEATURE)
    print("Self-test:", "PASS" if ok else "FAIL")
    db_path.unlink(missing_ok=True)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traffic AI Assist real agent core")
    p.add_argument("--version", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset-manifest", action="store_true")
    p.add_argument("--sync-datasets", action="store_true")
    p.add_argument("--fetch-dataset-metadata", type=Path)
    p.add_argument("--export-demo-sample", type=Path)
    p.add_argument("--demo-mode", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)

    p.add_argument("--input", type=Path)
    p.add_argument("--db", type=Path, default=Path("./traffic_ai.sqlite3"))
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
    if args.export_demo_sample:
        export_demo_file(args.export_demo_sample)
        print(str(args.export_demo_sample))
        return 0

    db = DB(args.db)
    bootstrapper = DatasetBootstrapper(db, logger)

    if args.dataset_manifest:
        print(DatasetRegistry.to_json())
        return 0
    if args.sync_datasets:
        bootstrapper.sync_catalog()
        print("dataset catalog synced")
        return 0
    if args.fetch_dataset_metadata:
        bootstrapper.fetch_remote_metadata(args.fetch_dataset_metadata)
        print(str(args.fetch_dataset_metadata))
        return 0
    if args.demo_mode:
        return run_demo_mode(db, args.lang, PrivacyMode(args.privacy), logger)
    if args.serve:
        APIServer(db, args.host, args.port, logger).serve()
        return 0

    if args.input is None:
        print("Error: --input is required unless --demo-mode/--serve is used.", file=sys.stderr)
        return 2

    backend = InferenceBackend(args.inference_backend)
    triton = NvidiaTritonClient(args.nvidia_endpoint, args.nvidia_model, logger) if backend == InferenceBackend.NVIDIA_TRITON else None
    return process_stream(iter_lines(args.input), db, args.lang, PrivacyMode(args.privacy), backend, triton, args.interactive, logger)


if __name__ == "__main__":
    raise SystemExit(main())
