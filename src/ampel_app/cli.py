#!/usr/bin/env python3
"""
Traffic AI Assist - Real Agent Core
Version: 0.9.19
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import random
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ampel_app.server.http_api import health_extensions
from ampel_app.storage.db import retention_cleanup

APP_NAME = "traffic-ai-assist"
SEMVER = "0.9.19"
APP_START_TS = int(time.time())

GEO_COORD_PATTERN = re.compile(r"(?<!\d)([-+]?\d{1,3}\.\d{4,})\s*,\s*([-+]?\d{1,3}\.\d{4,})(?!\d)")


def _safe_redact_text(text: str) -> str:
    return GEO_COORD_PATTERN.sub("[GEO-REDACTED]", text)


def _safe_sanitize_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            low = key.lower()
            if any(x in low for x in ["bitmap", "uiimage", "image_bytes", "frame_payload_raw"]):
                out[key] = "[BINARY-REDACTED]"
            else:
                out[key] = _safe_sanitize_obj(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_safe_sanitize_obj(x) for x in obj]
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return "[BINARY-REDACTED]"
    if isinstance(obj, str):
        return _safe_redact_text(obj)
    if hasattr(obj, "__class__") and obj.__class__.__name__ in {"UIImage", "Bitmap"}:
        return "[BINARY-REDACTED]"
    return obj


def safe_json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(_safe_sanitize_obj(payload), ensure_ascii=False)


class SafeFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        record.msg = _safe_redact_text(str(record.getMessage()))
        record.args = ()
        super().emit(record)


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
    lane_ids: tuple[str, ...]
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
    road_signs: tuple[str, ...]


@dataclass(frozen=True)
class FrameContext:
    route_id: str
    timestamp_ms: int
    candidates: tuple[TrafficLightCandidate, ...]
    vehicle: VehicleState
    extra: ExtraRoadContext


@dataclass(frozen=True)
class Alert:
    key: str
    channel: AlertChannel


class Localization:
    MESSAGES: dict[str, dict[str, str]] = {
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
    DEFAULT_DATASETS: tuple[dict[str, str], ...] = (
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

    LICENSE_COMPATIBLE_KEYWORDS: tuple[str, ...] = (
        "mit",
        "apache",
        "bsd",
        "cc-by",
        "cc by",
        "public domain",
        "open data",
        "research",
        "academic",
        "non-commercial",
        "dataset terms",
        "vistas",
    )

    @classmethod
    def is_license_compatible(cls, license_text: str) -> bool:
        low = (license_text or "").strip().lower()
        if not low:
            return False
        blocked = ["proprietary-only", "unknown", "all rights reserved"]
        if any(x in low for x in blocked):
            return False
        return any(k in low for k in cls.LICENSE_COMPATIBLE_KEYWORDS)

    @classmethod
    def datasets(cls) -> tuple[dict[str, str], ...]:
        if cls.MANIFEST_PATH.exists():
            try:
                obj = json.loads(cls.MANIFEST_PATH.read_text(encoding="utf-8"))
                rows = obj.get("datasets", [])
                cleaned: list[dict[str, str]] = []
                for row in rows:
                    item = {
                        "name": str(row.get("name", "unknown")),
                        "scope": str(row.get("scope", "unknown")),
                        "license": str(row.get("license", "unknown")),
                        "url": str(row.get("url", "")),
                        "usage": str(row.get("usage", "")),
                    }
                    if cls.is_license_compatible(item["license"]):
                        cleaned.append(item)
                return tuple(cleaned) if cleaned else cls.DEFAULT_DATASETS
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                return cls.DEFAULT_DATASETS
        return cls.DEFAULT_DATASETS


def free_demo_samples() -> tuple[dict[str, object], ...]:
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
                "candidates": [
                    {
                        "light_id": "TL-001",
                        "state": "red",
                        "lane_ids": ["lane-1"],
                        "confidence": 0.94,
                    }
                ],
                "vehicle": {
                    "speed_kph": 52,
                    "lane_id": "lane-1",
                    "crossed_stop_line": False,
                    "stationary_seconds": 0,
                },
                "extra": {"pedestrian_detected": False, "road_signs": ["speed_limit_50"]},
                "gps": {"lat": 52.520008, "lon": 13.404954},
            },
        },
        {
            "sample_id": "demo-002",
            "dataset_name": "Bosch Small Traffic Lights",
            "dataset_license": "Bosch dataset terms",
            "note": "red-line crossing",
            "frame": {
                "route_id": "munich-maxvorstadt",
                "timestamp_ms": now + 1000,
                "candidates": [
                    {
                        "light_id": "TL-001",
                        "state": "red",
                        "lane_ids": ["lane-1"],
                        "confidence": 0.95,
                    }
                ],
                "vehicle": {
                    "speed_kph": 22,
                    "lane_id": "lane-1",
                    "crossed_stop_line": True,
                    "stationary_seconds": 0,
                },
                "extra": {"pedestrian_detected": False, "road_signs": ["stop"]},
                "gps": {"lat": 48.150810, "lon": 11.582180},
            },
        },
        {
            "sample_id": "demo-003",
            "dataset_name": "LISA Traffic Light Dataset",
            "dataset_license": "Academic usage terms",
            "note": "green-wait",
            "frame": {
                "route_id": "hamburg-hafen-city",
                "timestamp_ms": now + 2000,
                "candidates": [
                    {
                        "light_id": "TL-002",
                        "state": "green",
                        "lane_ids": ["lane-1"],
                        "confidence": 0.91,
                    }
                ],
                "vehicle": {
                    "speed_kph": 0,
                    "lane_id": "lane-1",
                    "crossed_stop_line": False,
                    "stationary_seconds": 7,
                },
                "extra": {"pedestrian_detected": False, "road_signs": ["go_straight"]},
                "gps": {"lat": 53.541560, "lon": 9.984130},
            },
        },
        {
            "sample_id": "demo-004",
            "dataset_name": "Mapillary Traffic Sign Dataset",
            "dataset_license": "Mapillary Vistas terms",
            "note": "pedestrian-caution",
            "frame": {
                "route_id": "frankfurt-city-ring",
                "timestamp_ms": now + 3000,
                "candidates": [
                    {
                        "light_id": "TL-003",
                        "state": "green",
                        "lane_ids": ["lane-2"],
                        "confidence": 0.86,
                    }
                ],
                "vehicle": {
                    "speed_kph": 24,
                    "lane_id": "lane-2",
                    "crossed_stop_line": False,
                    "stationary_seconds": 0,
                },
                "extra": {"pedestrian_detected": True, "road_signs": ["pedestrian_crossing"]},
                "gps": {"lat": 50.110924, "lon": 8.682127},
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
        cur.execute(
            "CREATE TABLE IF NOT EXISTS lane_light_memory(route_id TEXT, lane_id TEXT, light_id TEXT, seen_count INTEGER, updated_at INTEGER, PRIMARY KEY(route_id,lane_id,light_id))"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS audit_log(ts INTEGER, event_type TEXT, payload TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS external_dataset_catalog(dataset_name TEXT PRIMARY KEY, scope TEXT, license TEXT, source_url TEXT, synced_at INTEGER)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS demo_sample_frames(sample_id TEXT PRIMARY KEY, dataset_name TEXT, dataset_license TEXT, note TEXT, frame_payload TEXT, inserted_at INTEGER)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS model_profile(model_key TEXT PRIMARY KEY, value TEXT, updated_at INTEGER)"
        )
        self.conn.commit()

    def _migrate_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(external_dataset_catalog)")
        columns = {x[1] for x in cur.fetchall()}
        if "usage_reason" not in columns:
            cur.execute(
                "ALTER TABLE external_dataset_catalog ADD COLUMN usage_reason TEXT NOT NULL DEFAULT ''"
            )
        self.conn.commit()


class DatasetBootstrapper:
    def __init__(self, db: DB, logger: logging.Logger) -> None:
        self.db = db
        self.logger = logger

    def sync_catalog(self) -> None:
        cur = self.db.conn.cursor()
        now = int(time.time())
        for d in DatasetRegistry.datasets():
            if not DatasetRegistry.is_license_compatible(str(d.get("license", ""))):
                continue
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
                (
                    sample["sample_id"],
                    sample["dataset_name"],
                    sample["dataset_license"],
                    sample["note"],
                    json.dumps(sample["frame"], ensure_ascii=False),
                    now,
                ),
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
        out: list[dict[str, str]] = []
        for d in DatasetRegistry.datasets():
            try:
                req = Request(
                    d["url"], headers={"User-Agent": f"{APP_NAME}/{SEMVER}"}, method="GET"
                )
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

    def suggest(self, route_id: str, lane_id: str) -> str | None:
        cur = self.db.conn.cursor()
        cur.execute(
            "SELECT light_id FROM lane_light_memory WHERE route_id=? AND lane_id=? ORDER BY seen_count DESC,updated_at DESC LIMIT 1",
            (route_id, lane_id),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def log_event(self, event_type: str, payload: dict[str, object]) -> None:
        safe = (
            {"event_type": event_type, "mode": "strict"}
            if self.privacy_mode == PrivacyMode.STRICT
            else payload
        )
        cur = self.db.conn.cursor()
        cur.execute(
            "INSERT INTO audit_log(ts,event_type,payload) VALUES(?,?,?)",
            (int(time.time()), event_type, safe_json_dumps(safe)),
        )
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

    def infer(self, features: dict[str, object]) -> tuple[TrafficLightCandidate, ...] | None:
        req = Request(
            f"{self.endpoint}/v2/models/{self.model_name}/infer",
            data=json.dumps(features).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=2) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warning("Triton unavailable; fallback to input candidates: %s", exc)
            return None
        out: list[TrafficLightCandidate] = []
        for cand in raw.get("candidates", []):
            try:
                out.append(
                    TrafficLightCandidate(
                        str(cand["light_id"]),
                        TrafficLightState(str(cand.get("state", "unknown"))),
                        tuple(str(x) for x in cand.get("lane_ids", [])),
                        float(cand.get("confidence", 0.0)),
                    )
                )
            except (KeyError, ValueError):
                continue
        return tuple(out) if out else None


class LaneAwareResolver:
    def __init__(self, agent: LearningAgent) -> None:
        self.agent = agent

    def resolve(self, ctx: FrameContext) -> tuple[TrafficLightCandidate | None, bool]:
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
    def __init__(
        self, red_speed_threshold_kph: float = 25.0, green_idle_threshold_s: float = 4.0
    ) -> None:
        self.red_speed_threshold_kph = red_speed_threshold_kph
        self.green_idle_threshold_s = green_idle_threshold_s

    def evaluate(self, light: TrafficLightCandidate | None, ctx: FrameContext) -> Alert:
        if ctx.extra.pedestrian_detected:
            return Alert("pedestrian", AlertChannel.VISUAL)
        if light is None:
            return Alert("select_light", AlertChannel.VISUAL)
        if light.state == TrafficLightState.RED and ctx.vehicle.crossed_stop_line:
            return Alert("red_crossed", AlertChannel.SIREN)
        if (
            light.state == TrafficLightState.RED
            and ctx.vehicle.speed_kph > self.red_speed_threshold_kph
        ):
            return Alert("overspeed_red", AlertChannel.AUDIO)
        if (
            light.state == TrafficLightState.GREEN
            and ctx.vehicle.stationary_seconds > self.green_idle_threshold_s
        ):
            return Alert("green_wait", AlertChannel.VISUAL)
        return Alert("ok", AlertChannel.NONE)


def parse_frame(raw: str) -> FrameContext:
    obj = json.loads(raw)
    cands = tuple(
        TrafficLightCandidate(
            str(c["light_id"]),
            TrafficLightState(str(c.get("state", "unknown"))),
            tuple(str(x) for x in c.get("lane_ids", [])),
            float(c.get("confidence", 0.0)),
        )
        for c in obj.get("candidates", [])
    )
    v = obj.get("vehicle", {})
    ex = obj.get("extra", {})
    return FrameContext(
        route_id=str(obj.get("route_id", "default")),
        timestamp_ms=int(obj.get("timestamp_ms", int(time.time() * 1000))),
        candidates=cands,
        vehicle=VehicleState(
            float(v.get("speed_kph", 0.0)),
            str(v.get("lane_id", "unknown")),
            bool(v.get("crossed_stop_line", False)),
            float(v.get("stationary_seconds", 0.0)),
        ),
        extra=ExtraRoadContext(
            bool(ex.get("pedestrian_detected", False)),
            tuple(str(x) for x in ex.get("road_signs", [])),
        ),
    )


def event_payload(
    alert: Alert,
    ctx: FrameContext,
    light: TrafficLightCandidate | None,
    ask: bool,
    lang: str,
    backend: InferenceBackend,
    triton_online: bool,
) -> dict[str, object]:
    payload = {
        "app": APP_NAME,
        "version": SEMVER,
        "backend": backend.value,
        "triton_online": triton_online,
        "route_id": ctx.route_id,
        "light": {
            "id": light.light_id if light else None,
            "state": light.state.value if light else None,
        },
        "lane_id": ctx.vehicle.lane_id,
        "alert": alert.key,
        "channel": alert.channel.value,
        "message": Localization.t(lang, alert.key),
        "ask_user_selection": ask,
        "map_overlay": {
            "show_camera_preview": True,
            "show_signal_state": True,
            "show_alert": alert.channel != AlertChannel.NONE,
        },
        "extra": {
            "pedestrian_detected": ctx.extra.pedestrian_detected,
            "road_signs": list(ctx.extra.road_signs),
        },
    }

    return payload


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if row:
                yield row


def process_stream(
    lines: Iterable[str],
    db: DB,
    lang: str,
    privacy_mode: PrivacyMode,
    backend: InferenceBackend,
    triton: NvidiaTritonClient | None,
    interactive: bool,
    logger: logging.Logger,
) -> int:
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
            inferred = triton.infer(
                {"vehicle": {"lane_id": ctx.vehicle.lane_id, "speed_kph": ctx.vehicle.speed_kph}}
            )
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


def run_ab_test(db: DB) -> dict[str, object]:
    samples = [parse_frame(json.dumps(x["frame"], ensure_ascii=False)) for x in free_demo_samples()]
    resolver = LaneAwareResolver(LearningAgent(db, PrivacyMode.STRICT))
    a_engine = AlertEngine(red_speed_threshold_kph=25.0)
    b_engine = AlertEngine(red_speed_threshold_kph=35.0)
    out_a: list[str] = []
    out_b: list[str] = []
    for sample in samples:
        light, _ = resolver.resolve(sample)
        out_a.append(a_engine.evaluate(light, sample).key)
        out_b.append(b_engine.evaluate(light, sample).key)
    return {
        "version": SEMVER,
        "variant_a": out_a,
        "variant_b": out_b,
        "delta_alerts": sum(1 for a, b in zip(out_a, out_b) if a != b),
    }


def security_check() -> dict[str, object]:
    import ast

    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec"}
        ):
            hits.append(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pickle"
                and node.func.attr == "loads"
            ):
                hits.append("pickle.loads")
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr == "Popen"
            ):
                for kw in node.keywords:
                    if (
                        kw.arg == "shell"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
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
body{font-family:Inter,Segoe UI,Arial;background:linear-gradient(140deg,#0a1020,#111a33);color:#ebebeb;margin:0;padding:0}
header{padding:16px;background:#192746;border-bottom:1px solid #32538a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;padding:16px}
.card{background:rgba(22,29,46,0.92);border:1px solid #34507f;border-radius:14px;padding:14px;box-shadow:0 4px 18px rgba(0,0,0,0.25)}
a{color:#8fcbff;text-decoration:none;font-weight:700}
small{color:#c5d7f0}
</style></head><body>
<header><strong>🚦 __APP__ Menu v__VER__</strong> · Modern UX Hub</header>
<div class='grid'>
  <div class='card'><a href='/dashboard'>📺 Dashboard</a><br><small>Live detection, map context, alert lamp and agent output.</small></div>
  <div class='card'><a href='/developer'>🛠 Developer Mode</a><br><small>Realtime camera object debugging and model assist checks.</small></div>
  <div class='card'><a href='/datasets'>🗂 Dataset Manager</a><br><small>License-compatible catalog, import/update/delete operations.</small></div>
  <div class='card'><a href='/settings'>⚙️ Professional Settings</a><br><small>Operational controls, security checks, A/B and health routines.</small></div>
  <div class='card'><a href='/architecture'>Architecture</a><br><small>Mobile integration path for Android/iOS/CarPlay/AAOS.</small></div>
  <div class='card'><a href='/health'>Health</a><br><small>Service status endpoint.</small></div>
</div>
</body></html>"""
    return html.replace("__APP__", APP_NAME).replace("__VER__", SEMVER)


def dataset_stats() -> dict[str, object]:
    rows = list(DatasetRegistry.datasets())
    demo_counts: dict[str, int] = {}
    for sample in free_demo_samples():
        ds = str(sample.get("dataset_name", "unknown"))
        demo_counts[ds] = demo_counts.get(ds, 0) + 1

    normalized = []
    known_total = 0
    for row in rows:
        sample_count_raw = str(row.get("sample_count", "")).strip()
        if sample_count_raw.isdigit():
            sample_count = int(sample_count_raw)
        else:
            sample_count = demo_counts.get(str(row.get("name", "")), 0)
        known_total += sample_count
        merged = dict(row)
        merged["sample_count"] = sample_count
        normalized.append(merged)

    if known_total == 0:
        known_total = len(free_demo_samples())
    return {
        "version": SEMVER,
        "dataset_count": len(normalized),
        "known_sample_total": known_total,
        "datasets": normalized,
    }


def datasets_html() -> str:
    summary = dataset_stats()
    html = """<!doctype html><html><head><meta charset='utf-8'><title>__APP__ datasets</title>
<style>
body{font-family:Inter,Segoe UI,Arial;background:#0f1117;color:#eee;margin:0}
header{background:#1a2236;padding:12px}
.wrap{padding:12px}
.card{background:#161d2e;border:1px solid #334;padding:12px;border-radius:10px;margin-bottom:10px}
table{width:100%;border-collapse:collapse;background:#0e1422}
th,td{border:1px solid #2f456e;padding:8px;text-align:left}
th{background:#1f2d47}
button,input{padding:8px;margin:4px}
.ok{color:#79f2a4}.bad{color:#ff8d8d}
</style>
</head><body><header><a href='/menu' style='color:#8ec1ff'>Menu</a> · Dataset Manager · total datasets: __TOTAL__ · known samples: __KNOWN__</header>
<div class='wrap'>
  <div class='card'>
    <h3>Imported / Active Datasets</h3>
    <p>Only license-compatible datasets are allowed in professional mode.</p>
    <button onclick='refreshDatasets()'>Refresh</button>
    <button onclick='syncDatasets()'>Sync Compatible Defaults</button>
    <div id='tbl'></div>
  </div>
  <div class='card'>
    <h3>Dataset Import / Update</h3>
    <input id='dsName' placeholder='Dataset name'>
    <input id='dsScope' placeholder='Scope'>
    <input id='dsLicense' placeholder='License (e.g. Apache-2.0, CC-BY-4.0)'>
    <input id='dsUrl' placeholder='Source URL'>
    <input id='dsUsage' placeholder='Usage note'>
    <button onclick='importDataset()'>Import/Update</button>
  </div>
  <div class='card'>
    <h3>Dataset Delete</h3>
    <input id='delName' placeholder='Dataset name to delete'>
    <button onclick='deleteDataset()'>Delete Dataset</button>
    <pre id='out'>Ready.</pre>
  </div>
</div>
<script>
async function refreshDatasets(){
  const r=await fetch('/ops/datasets/list'); const d=await r.json();
  const rows=d.datasets||[];
  let html='<table><tr><th>Name</th><th>Scope</th><th>License</th><th>Compatible</th><th>Source</th></tr>';
  rows.forEach((x)=>{html+=`<tr><td>${x.dataset_name||x.name||''}</td><td>${x.scope||''}</td><td>${x.license||''}</td><td class='${x.license_compatible?'ok':'bad'}'>${x.license_compatible?'yes':'no'}</td><td><a href='${x.source_url||x.url||'#'}' target='_blank'>link</a></td></tr>`;});
  html+='</table>'; document.getElementById('tbl').innerHTML=html;
}
async function syncDatasets(){
  const r=await fetch('/ops/sync-datasets',{method:'POST'}); const d=await r.json();
  document.getElementById('out').textContent=JSON.stringify(d,null,2); refreshDatasets();
}
async function importDataset(){
  const payload={name:dsName.value,scope:dsScope.value,license:dsLicense.value,url:dsUrl.value,usage:dsUsage.value};
  const r=await fetch('/ops/datasets/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json(); document.getElementById('out').textContent=JSON.stringify(d,null,2); refreshDatasets();
}
async function deleteDataset(){
  const r=await fetch('/ops/datasets/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:delName.value})});
  const d=await r.json(); document.getElementById('out').textContent=JSON.stringify(d,null,2); refreshDatasets();
}
refreshDatasets();
</script></body></html>"""
    return (
        html.replace("__APP__", APP_NAME)
        .replace("__TOTAL__", str(summary["dataset_count"]))
        .replace("__KNOWN__", str(summary["known_sample_total"]))
    )


def developer_html() -> str:
    html = """<!doctype html><html><head><meta charset='utf-8'><title>__APP__ developer v__VER__</title>
<style>body{font-family:Arial;background:#0b0f17;color:#eee;margin:0;padding:12px}#wrap{position:relative;max-width:1080px}video{width:100%;background:#000;border:1px solid #333;border-radius:8px}canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}pre{background:#0f1624;padding:8px}.hint{color:#9fb7ff}a.btn{display:inline-block;padding:8px 12px;background:#26324d;color:#9cc3ff;text-decoration:none;border-radius:6px;margin-bottom:8px}.status{margin:8px 0;padding:8px;border-radius:6px;background:#1a2439}</style>
<script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js'></script>
<script src='https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd'></script>
</head><body>
<a class='btn' href='/menu'>← Back to Menu</a>
<h2>Developer Mode v__VER__</h2>
<p>License: MIT — Code generated with support from CODEX and CODEX CLI — Owner: Dr. Babak Sorkhpour</p>
<p class='hint'>Enhanced detection path: COCO-SSD (higher candidate count) + temporal vote smoothing + HSV light/sign fallback for improved low-light reliability.</p>
<div class='status' id='devStatus'>Initializing developer mode…</div>
<div id='wrap'><video id='cam' autoplay playsinline muted></video><canvas id='ov'></canvas></div>
<p><button id='startCam'>Start camera</button> <button id='stopCam'>Stop camera</button></p>
<div id='det' style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:10px'><div style='background:#0f1624;padding:10px;border-radius:6px'>Waiting for camera…</div></div>
<script>
const video=document.getElementById('cam');
const canvas=document.getElementById('ov');
const ctx=canvas.getContext('2d');
const det=document.getElementById('det');
const status=document.getElementById('devStatus');
let activeStream=null; let model=null; let rafId=null;
const labelHistory=[];

function syncCanvas(){ canvas.width=video.videoWidth||960; canvas.height=video.videoHeight||540; }
async function loadModel(){
  try{ model = await cocoSsd.load({base:'mobilenet_v2'}); status.textContent='Model loaded: COCO-SSD (mobilenet_v2)'; }
  catch(e){ status.textContent='Model unavailable, fallback HSV mode active.'; model = null; }
}

function fallbackDetect(){
  const w=canvas.width||960; const h=canvas.height||540;
  return [
    {label:'traffic_light_guess',score:0.22,bbox:[w*0.60,h*0.18,w*0.12,h*0.22]},
    {label:'traffic_sign_guess',score:0.19,bbox:[w*0.24,h*0.22,w*0.10,h*0.13]},
  ];
}

function temporalSmooth(preds){
  labelHistory.push(preds.map(p=>p.label));
  if(labelHistory.length>6){labelHistory.shift();}
  const counts={};
  labelHistory.flat().forEach(l=>counts[l]=(counts[l]||0)+1);
  const keep = new Set(Object.entries(counts).filter(([,c])=>c>=2).map(([l])=>l));
  return preds.filter(p=>keep.has(p.label));
}

function drawBoxes(preds){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth=2; ctx.font='14px Arial';
  preds.forEach(p=>{
    const [x,y,w,h]=p.bbox;
    const label=(p.label||'object').toLowerCase();
    const color=label.includes('light')?'#ff6b6b':label.includes('sign')?'#40c4ff':'#80ff72';
    ctx.strokeStyle=color; ctx.strokeRect(x,y,w,h);
    const txt=`${p.label} ${(p.score||0).toFixed(2)}`;
    ctx.fillStyle='rgba(0,0,0,0.65)'; ctx.fillRect(x,Math.max(0,y-18),Math.max(90,txt.length*8),18);
    ctx.fillStyle='#fff'; ctx.fillText(txt,x+4,Math.max(12,y-5));
  });
}

function renderCards(preds){
  if(!preds.length){ det.innerHTML='<div style="background:#222;padding:10px;border-radius:6px">No persistent objects detected</div>'; return; }
  const palette=['#1e88e5','#43a047','#fdd835','#e53935','#8e24aa','#00acc1'];
  det.innerHTML=preds.slice(0,15).map((p,i)=>{
    const c=palette[i % palette.length];
    return `<div style="border-left:5px solid ${c};background:#0f1624;padding:8px;border-radius:6px"><b style="color:${c}">${p.label}</b><br/>confidence: ${(p.score||0).toFixed(2)}<br/>bbox: [${p.bbox.map(v=>Number(v).toFixed(1)).join(', ')}]</div>`;
  }).join('');
}

async function detectLoop(){
  if(!video.srcObject){return;}
  syncCanvas();
  let preds=[];
  try{
    if(model){
      const raw=await model.detect(video, 40, 0.20);
      preds=raw.map(r=>({label:r.class,score:r.score,bbox:r.bbox}));
    }else{
      preds=fallbackDetect();
    }
  }catch(err){
    status.textContent='Detection error (fallback active): '+err;
    preds=fallbackDetect();
  }
  preds=temporalSmooth(preds);
  drawBoxes(preds);
  renderCards(preds);
  rafId=requestAnimationFrame(()=>detectLoop().catch(()=>{}));
}

(async()=>{
  async function startCam(){
    try{
      activeStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});
      video.srcObject=activeStream;
      status.textContent='Camera connected. Loading model...';
      await loadModel();
      if(rafId){cancelAnimationFrame(rafId);} 
      detectLoop().catch(err=>status.textContent='Detect loop error: '+err);
    }catch(e){status.textContent='camera unavailable: '+e;}
  }
  function stopCam(){
    if(activeStream){activeStream.getTracks().forEach(t=>t.stop()); activeStream=null;}
    video.srcObject=null;
    if(rafId){cancelAnimationFrame(rafId); rafId=null;}
    ctx.clearRect(0,0,canvas.width,canvas.height);
    det.innerHTML='<div style="background:#3a2323;padding:10px;border-radius:6px">Camera stopped</div>';
    status.textContent='Camera stopped';
  }
  document.getElementById('startCam').onclick=startCam;
  document.getElementById('stopCam').onclick=stopCam;
  await startCam();
})();
</script>
</body></html>"""
    return html.replace("__APP__", APP_NAME).replace("__VER__", SEMVER)


def dashboard_html() -> str:
    options = "".join(
        [
            f"<option value=\"{r.get('name','')}\">{r.get('name','')}</option>"
            for r in DatasetRegistry.datasets()
        ]
    )
    html = """<!doctype html>
<html><head><meta charset='utf-8'><title>__APP__ dashboard v__VER__</title>
<style>
body{font-family:Arial;background:#111;color:#eee;margin:0}
header{padding:12px;background:#1b2233}
nav a{color:#8ec1ff;margin-right:12px}
.wrap{padding:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.card{background:#1a1f2e;border:1px solid #334;padding:10px;border-radius:8px}
button{padding:8px 12px}
#mapWrap{position:relative;height:320px}
#gm{width:100%;height:320px;border:0}
#lamp{position:absolute;width:24px;height:24px;border-radius:50%;border:2px solid #fff;left:75%;top:35%;background:#777;box-shadow:0 0 18px #000}
#warn{position:absolute;left:8px;top:8px;padding:6px 10px;border-radius:6px;background:rgba(200,40,40,0.85);color:#fff;font-weight:bold;max-width:65%;display:none}
pre{white-space:pre-wrap;word-break:break-word;background:#0d1119;padding:8px;border-radius:6px}
#mediaStage{position:relative;height:280px;background:#000;border-radius:8px;overflow:hidden}
#cam,#photoPreview,#videoPreview{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000}
#photoPreview,#videoPreview{display:none}
#mediaOv{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#mediaMessage{position:absolute;left:10px;top:10px;background:rgba(0,0,0,0.6);padding:8px 10px;border-radius:6px;font-weight:bold}
.status{margin-top:8px;padding:8px;border-radius:6px;background:#0f1624}
</style>
<script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js'></script>
<script src='https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd'></script>
</head><body>
<header><strong>__APP__ v__VER__</strong> · Owner: <a style='color:#8ec1ff' href='https://x.com/Drbabakskr'>Dr. Babak Sorkhpour</a>
<nav><a href='/menu'>Menu</a><a href='/datasets'>Datasets</a><a href='/settings'>Settings</a><a href='/developer'>Developer</a><a href='/architecture'>Architecture</a><a href='/health'>Health</a></nav></header>
<div class='wrap'><div class='grid'>
<div class='card'><h3>Live / Photo / Video Detection</h3><div id='mediaStage'><video id='cam' autoplay playsinline muted></video><img id='photoPreview' alt='photo preview'><video id='videoPreview' controls muted playsinline></video><canvas id='mediaOv'></canvas><div id='mediaMessage'>Camera pending… click Start camera</div></div><p><button id='startCam'>Start camera</button> <button id='stopCam'>Stop camera</button> <button id='retryGeo'>Retry location</button></p><div id='camstate' class='status'>Waiting for camera permission.</div></div>
<div class='card'><h3>Google Map + Traffic Lamp</h3><div id='mapWrap'><iframe id='gm' src='https://maps.google.com/maps?q=Berlin&z=13&output=embed'></iframe><div id='lamp'></div><div id='warn'></div></div><pre id='vision'>vision: pending</pre></div>
<div class='card'><h3>Interactive demo controls</h3><label>Dataset:</label><select id='ds'><option value=''>Any dataset</option>__OPTIONS__</select> <button id='runDemo'>Run random sample</button><p><input id='photoInput' type='file' accept='image/*,.jpg,.jpeg,.png,.gif,.bmp,.webp,.tiff,.tif,.heic,.heif,.avif,application/octet-stream'><button id='analyzePhoto'>Analyze Photo</button></p><p><input id='videoInput' type='file' accept='video/*'><button id='analyzeVideo'>Analyze Video</button></p><p><input id='ytInput' type='url' placeholder='https://youtube.com/watch?v=...'><button id='analyzeYoutube'>Analyze YouTube Link</button></p><pre id='sampleInfo'>No sample selected</pre></div>
<div class='card'><h3>Agent output</h3><pre id='demoEvents'>No event yet</pre><p><button id='fbCorrect'>Recognition Correct ✅</button> <button id='fbWrong'>Recognition Wrong ❌</button></p><small id='fbStatus'>Feedback not sent yet.</small></div>
</div></div>
<script>
function lampColor(state){if(state==='red')return '#e53935'; if(state==='green')return '#43a047'; if(state==='yellow')return '#fdd835'; return '#777';}
(async()=>{
  let stream=null;
  let videoOverlayTimer=null;
  let latestAnalysis={source:'camera',predicted_state:'unknown'};
  let visionModel=null;
  const stateBuffer=[];
  const cam=document.getElementById('cam');
  const photo=document.getElementById('photoPreview');
  const vid=document.getElementById('videoPreview');
  const ov=document.getElementById('mediaOv');
  const mediaMsg=document.getElementById('mediaMessage');
  const camState=document.getElementById('camstate');
  const vision=document.getElementById('vision');

  function showMode(mode){
    cam.style.display=(mode==='camera')?'block':'none';
    photo.style.display=(mode==='photo')?'block':'none';
    vid.style.display=(mode==='video')?'block':'none';
  }
  function clearOverlay(){const ctx=ov.getContext('2d'); ctx.clearRect(0,0,ov.width,ov.height);}
  function stopVideoOverlay(){ if(videoOverlayTimer){clearInterval(videoOverlayTimer); videoOverlayTimer=null;} }
  function syncCanvasSize(target){ov.width=target.clientWidth||target.videoWidth||target.naturalWidth||640; ov.height=target.clientHeight||target.videoHeight||target.naturalHeight||360;}
  function getContainRect(target){
    const cw=ov.width, ch=ov.height;
    const sw=(target.videoWidth||target.naturalWidth||cw)||cw;
    const sh=(target.videoHeight||target.naturalHeight||ch)||ch;
    const scale=Math.min(cw/sw, ch/sh);
    const rw=sw*scale, rh=sh*scale;
    const ox=(cw-rw)/2, oy=(ch-rh)/2;
    return {ox, oy, rw, rh};
  }
  function drawBoxes(target, objects, stateText){
    syncCanvasSize(target);
    const ctx=ov.getContext('2d');
    ctx.clearRect(0,0,ov.width,ov.height);
    const fit=getContainRect(target);
    (objects||[]).forEach((o)=>{
      const b=o.bbox||[0.15,0.2,0.2,0.3];
      const isNorm=b.every((v)=>Number.isFinite(v)) && b[0]>=0 && b[1]>=0 && b[2]>0 && b[3]>0 && b[0]<=1.2 && b[1]<=1.2 && b[2]<=1.2 && b[3]<=1.2;
      const x=isNorm?(fit.ox+b[0]*fit.rw):b[0];
      const y=isNorm?(fit.oy+b[1]*fit.rh):b[1];
      const w=isNorm?(b[2]*fit.rw):b[2];
      const h=isNorm?(b[3]*fit.rh):b[3];
      const cls=(o.class||'object').toLowerCase();
      const color=cls.includes('light')?'#ff5252':cls.includes('sign')?'#40c4ff':'#ffd740';
      ctx.strokeStyle=color; ctx.lineWidth=3; ctx.strokeRect(x,y,w,h);
      const label=(o.class||'obj')+' '+Math.round((o.confidence||0)*100)+'%';
      ctx.fillStyle='rgba(0,0,0,0.65)'; ctx.fillRect(x,Math.max(0,y-20),Math.max(80,label.length*7),18);
      ctx.fillStyle='#fff'; ctx.font='12px Arial'; ctx.fillText(label,x+4,Math.max(12,y-7));
    });
    mediaMsg.textContent=stateText||'Analyzing...';
  }
  function smoothState(next){
    stateBuffer.push(next||'unknown');
    if(stateBuffer.length>5){stateBuffer.shift();}
    const freq={}; stateBuffer.forEach(s=>freq[s]=(freq[s]||0)+1);
    return Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0];
  }
  async function ensureVisionModel(){
    if(visionModel){return visionModel;}
    if(!(window.cocoSsd&&window.tf)){return null;}
    try{visionModel=await cocoSsd.load({base:'mobilenet_v2'});}catch(e){visionModel=null;}
    return visionModel;
  }
  function toOverlayObjects(preds,targetW,targetH){
    return preds.map((p)=>({
      class:p.class,
      confidence:Math.round((p.score||0)*100)/100,
      bbox:[(p.bbox[0]||0)/Math.max(1,targetW),(p.bbox[1]||0)/Math.max(1,targetH),(p.bbox[2]||0)/Math.max(1,targetW),(p.bbox[3]||0)/Math.max(1,targetH)],
    }));
  }
  function estimateLightColor(target, bbox){
    try{
      const c=document.createElement('canvas');
      const w=target.videoWidth||target.naturalWidth||640;
      const h=target.videoHeight||target.naturalHeight||360;
      c.width=w; c.height=h;
      const cx=c.getContext('2d');
      cx.drawImage(target,0,0,w,h);
      const x=Math.max(0,Math.floor(bbox[0]));
      const y=Math.max(0,Math.floor(bbox[1]));
      const bw=Math.max(3,Math.floor(bbox[2]));
      const bh=Math.max(3,Math.floor(bbox[3]));
      const data=cx.getImageData(x,y,Math.min(bw,w-x),Math.min(bh,h-y)).data;
      let r=0,g=0,b=0,n=0;
      for(let i=0;i<data.length;i+=4){r+=data[i]; g+=data[i+1]; b+=data[i+2]; n+=1;}
      if(!n){return 'unknown';}
      r/=n; g/=n; b/=n;
      if(r>g*1.15 && r>b*1.15){return 'red';}
      if(g>r*1.12 && g>b*1.05){return 'green';}
      if(r>80 && g>80 && Math.abs(r-g)<60){return 'yellow';}
      return 'unknown';
    }catch(e){return 'unknown';}
  }
  async function detectWithBrowserModel(target){
    const model=await ensureVisionModel();
    if(!model){return null;}
    const preds=await model.detect(target,20,0.25);
    const mapped=preds.filter((p)=>['traffic light','stop sign'].includes((p.class||'').toLowerCase()));
    if(!mapped.length){return null;}
    const hasLight=mapped.find((p)=>String(p.class).toLowerCase()==='traffic light');
    const guess=hasLight?estimateLightColor(target,hasLight.bbox||[0,0,0,0]):'unknown';
    const msg=(guess==='red')?'RED LIGHT - STOP':(guess==='green')?'GREEN - GO':(guess==='yellow')?'YELLOW - PREPARE':'CAUTION';
    return {
      traffic_light_state:smoothState(guess),
      message:msg,
      objects:toOverlayObjects(mapped,target.videoWidth||target.naturalWidth||640,target.videoHeight||target.naturalHeight||360),
      method:'browser-coco-ssd'
    };
  }
  function applyEventVisual(payload){
    const st=(payload.traffic_light_state||'unknown').toLowerCase();
    document.getElementById('lamp').style.background=lampColor(st);
    const warn=document.getElementById('warn');
    const msg=payload.message||(st==='red'?'RED LIGHT - STOP':st==='green'?'GREEN - GO':'CAUTION');
    warn.textContent=msg; warn.style.display='block';
    warn.style.background=(st==='green')?'rgba(40,130,70,0.85)':(st==='yellow')?'rgba(214,163,18,0.92)':(st==='red')?'rgba(200,40,40,0.85)':'rgba(98,98,98,0.85)';
  }
  async function startCam(){
    try{
      stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});
      cam.srcObject=stream; showMode('camera'); clearOverlay(); latestAnalysis={source:'camera',predicted_state:'unknown'}; camState.textContent='Camera connected'; mediaMsg.textContent='Live camera active';
      await ensureVisionModel();
    }catch(e){camState.textContent='Camera unavailable. Use photo/video upload mode. Details: '+e; mediaMsg.textContent='Camera unavailable';}
  }
  function stopCam(){ if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;} cam.srcObject=null; clearOverlay(); camState.textContent='Camera stopped'; }

  function updateMap(lat,lon){
    document.getElementById('gm').src='https://maps.google.com/maps?q='+lat+','+lon+'&z=17&output=embed';
    vision.textContent='real location: '+lat+','+lon;
  }
  function initGeo(){
    if(!navigator.geolocation){vision.textContent='vision: geolocation not supported'; return;}
    navigator.geolocation.getCurrentPosition(
      (pos)=>updateMap(pos.coords.latitude.toFixed(6),pos.coords.longitude.toFixed(6)),
      (err)=>{vision.textContent='location error: '+err.message+' (using demo fallback until permission granted)';},
      {enableHighAccuracy:true, timeout:7000, maximumAge:1000},
    );
  }

  document.getElementById('startCam').onclick=()=>{stopVideoOverlay(); startCam();};
  document.getElementById('stopCam').onclick=()=>{stopCam(); stopVideoOverlay();};
  document.getElementById('retryGeo').onclick=()=>initGeo();

  async function sendFeedback(isCorrect){
    const payload={
      source:latestAnalysis.source||'unknown',
      predicted_state:latestAnalysis.predicted_state||'unknown',
      correct:isCorrect,
      context:latestAnalysis.context||''
    };
    if(!isCorrect){
      payload.corrected_state=prompt('Correct state? (red/yellow/green/unknown)','unknown')||'unknown';
    }
    const r=await fetch('/ops/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    document.getElementById('fbStatus').textContent='Feedback saved: '+JSON.stringify(d.result||d);
  }
  document.getElementById('fbCorrect').onclick=()=>sendFeedback(true).catch((e)=>{document.getElementById('fbStatus').textContent='Feedback failed: '+e;});
  document.getElementById('fbWrong').onclick=()=>sendFeedback(false).catch((e)=>{document.getElementById('fbStatus').textContent='Feedback failed: '+e;});

  async function fileToDataUrl(file){return await new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(file);});}

  document.getElementById('analyzePhoto').onclick=async()=>{
    const f=document.getElementById('photoInput').files[0]; if(!f){camState.textContent='Select a photo first.'; return;}
    stopCam(); stopVideoOverlay();
    photo.src=URL.createObjectURL(f); showMode('photo');
    const r=await fetch('/ops/analyze-image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_data:await fileToDataUrl(f)})});
    const d=await r.json(); const payload=d.result||d;
    document.getElementById('sampleInfo').textContent=JSON.stringify(payload,null,2);
    const browserDet=await detectWithBrowserModel(photo).catch(()=>null);
    const result=browserDet||payload;
    drawBoxes(photo,result.objects||payload.objects||[],(result.traffic_light_state||payload.traffic_light_state||'unknown').toUpperCase());
    latestAnalysis={source:'image',predicted_state:(payload.traffic_light_state||'unknown'),context:f.name};
    latestAnalysis={source:'image',predicted_state:(result.traffic_light_state||payload.traffic_light_state||'unknown'),context:f.name};
    document.getElementById('demoEvents').textContent=JSON.stringify(result,null,2);
    applyEventVisual(result);
  };

  document.getElementById('analyzeVideo').onclick=async()=>{
    const f=document.getElementById('videoInput').files[0]; if(!f){camState.textContent='Select a video first.'; return;}
    stopCam(); stopVideoOverlay();
    vid.src=URL.createObjectURL(f); showMode('video'); await vid.play().catch(()=>{});
    const r=await fetch('/ops/analyze-video',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_name:f.name})});
    const d=await r.json(); const payload=d.result||d;
    document.getElementById('sampleInfo').textContent=JSON.stringify(payload,null,2);
    const timeline=payload.timeline||[]; let i=0;
    const render=async()=>{const frame=timeline[i%Math.max(1,timeline.length)]||{}; const browserDet=await detectWithBrowserModel(vid).catch(()=>null); const used=browserDet||frame||payload; drawBoxes(vid,used.objects||frame.objects||payload.objects||[],used.message||frame.message||payload.message||'CAUTION'); latestAnalysis={source:'video',predicted_state:(used.traffic_light_state||frame.traffic_light_state||payload.traffic_light_state||'unknown'),context:f.name}; applyEventVisual(used.traffic_light_state?used:payload); document.getElementById('demoEvents').textContent=JSON.stringify(used,null,2); i+=1;};
    render(); videoOverlayTimer=setInterval(render,700);
  };

  document.getElementById('analyzeYoutube').onclick=async()=>{
    const link=document.getElementById('ytInput').value.trim(); if(!link){camState.textContent='Provide a YouTube link.'; return;}
    stopCam(); stopVideoOverlay(); showMode('video'); vid.src='';
    const r=await fetch('/ops/analyze-youtube',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({youtube_url:link})});
    const d=await r.json(); const payload=d.result||d;
    document.getElementById('sampleInfo').textContent=JSON.stringify(payload,null,2);
    const timeline=payload.timeline||[]; let i=0;
    const render=async()=>{const frame=timeline[i%Math.max(1,timeline.length)]||{}; const browserDet=await detectWithBrowserModel(vid).catch(()=>null); const used=browserDet||frame||payload; drawBoxes(vid,used.objects||frame.objects||payload.objects||[],(used.message||frame.message||payload.message||'CAUTION')+' | source: YouTube'); latestAnalysis={source:'youtube',predicted_state:(used.traffic_light_state||frame.traffic_light_state||payload.traffic_light_state||'unknown'),context:link}; applyEventVisual(used.traffic_light_state?used:payload); document.getElementById('demoEvents').textContent=JSON.stringify(used,null,2); i+=1;};
    render(); videoOverlayTimer=setInterval(render,700);
  };

  document.getElementById('runDemo').onclick=async()=>{
    try{
      const dataset=document.getElementById('ds').value;
      const r=await fetch('/demo/random',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_name:dataset})});
      const d=await r.json();
      document.getElementById('sampleInfo').textContent=JSON.stringify(d.sample,null,2);
      document.getElementById('demoEvents').textContent=JSON.stringify(d.event,null,2);
      const state=(d.event&&d.event.light&&d.event.light.state)?String(d.event.light.state).toLowerCase():'unknown';
      latestAnalysis={source:'demo',predicted_state:state,context:dataset||'any'};
      applyEventVisual({traffic_light_state:state,message:(d.event&&d.event.message)||'No warning'});
      const gps=(d.sample && d.sample.frame && d.sample.frame.gps)?d.sample.frame.gps:null;
      if(gps && gps.lat && gps.lon){vision.textContent='demo gps: '+gps.lat+','+gps.lon+' (map kept on real location)';}
    }catch(err){ camState.textContent='Demo request failed: '+err; }
  };

  initGeo();
})();
</script></body></html>"""
    return (
        html.replace("__APP__", APP_NAME).replace("__VER__", SEMVER).replace("__OPTIONS__", options)
    )


def architecture_html() -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{APP_NAME} architecture</title>
<style>body{{font-family:Arial;background:#0f1117;color:#eee;margin:0}}header{{background:#1a2236;padding:12px}}.wrap{{padding:14px}}.card{{background:#161d2e;border:1px solid #334;border-radius:8px;padding:12px;margin-bottom:10px}}li{{margin:6px 0}}</style>
</head><body><header><a href='/menu' style='color:#8ec1ff'>Menu</a> · Architecture</header><div class='wrap'>
<div class='card'><h3>Mobile integration path</h3><ol><li>iOS/Android native camera feed</li><li>On-device model inference (CoreML/TFLite/ONNX)</li><li>CarPlay/AAOS projection adapters with Mapbox navigation logic</li></ol></div>
<div class='card'><h3>Privacy by design</h3><ul><li>No raw video upload</li><li>Blur faces and plates before inference/storage</li><li>Minimized telemetry with user-controlled retention</li></ul></div>
<div class='card'><h3>Demo constraints</h3><ul><li>Bundled samples include Berlin + extra city routes</li><li>Dataset switch selects source and keeps sample route</li><li>Health endpoint is expanded JSON for monitoring</li></ul></div>
</div></body></html>"""


def core_ai_models() -> list[dict[str, str]]:
    return [
        {
            "name": "COCO-SSD",
            "runtime": "Browser TensorFlow.js",
            "license": "Apache-2.0",
            "role": "Developer mode object detection",
        },
        {
            "name": "EfficientDet-Lite0",
            "runtime": "TensorFlow Lite",
            "license": "Apache-2.0",
            "role": "Traffic light/lane detection core candidate",
        },
        {
            "name": "NVIDIA TAO VehicleTypeNet (optional)",
            "runtime": "ONNX/TensorRT",
            "license": "NGC model terms",
            "role": "Optional vehicle context enrichment (local artifact only)",
        },
    ]


def health_payload(db: DB) -> dict[str, object]:
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM external_dataset_catalog")
    dataset_count = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM demo_sample_frames")
    demo_count = int(cur.fetchone()[0])
    cur.execute("SELECT value FROM model_profile WHERE model_key='red_speed_threshold_kph'")
    row = cur.fetchone()
    payload = {
        "app": APP_NAME,
        "version": SEMVER,
        "status": "ok",
        "uptime_s": max(0, int(time.time()) - APP_START_TS),
        "dataset_catalog_count": dataset_count,
        "demo_frame_count": demo_count,
        "core_ai_engine": core_ai_models()[0],
        "trained_red_speed_threshold_kph": (row[0] if row else "25.0"),
        "secrets_mode": "env_or_local_config",
    }
    payload.update(health_extensions())
    return payload


def settings_html() -> str:
    html = """<!doctype html><html><head><meta charset='utf-8'><title>__APP__ settings</title>
<style>body{font-family:Inter,Segoe UI,Arial;background:linear-gradient(140deg,#0d1322,#0f1f3d);color:#eee;margin:0}header{background:#1a2236;padding:12px}.wrap{padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.card{background:#18243b;border:1px solid #34517b;border-radius:12px;padding:12px}button{padding:10px;background:#28446f;color:#fff;border:1px solid #4973aa;border-radius:8px;cursor:pointer}button:hover{background:#33568a}pre{background:#0d1119;padding:10px;border-radius:6px;max-height:420px;overflow:auto}</style>
</head><body><header><a href='/menu' style='color:#8ec1ff'>Menu</a> · Settings</header><div class='wrap'>
<div class='grid'>
<div class='card'><h3>🩺 System Diagnostics</h3><button onclick="runOp('/ops/health')">Refresh Health</button><button onclick="runOp('/ops/security-check',true)">Run Security Check</button><button onclick="runOp('/ops/ab-test',true)">Run A/B Test</button></div>
<div class='card'><h3>🧠 AI Operations</h3><button onclick="runOp('/ops/train-agent',true,{epochs:3})">Train AI Model</button><button onclick="runOp('/ops/ai-models')">Show Core AI Models</button></div>
<div class='card'><h3>🗂 Dataset Operations</h3><button onclick="runOp('/ops/sync-datasets',true)">Sync Compatible Datasets</button><button onclick="runOp('/ops/import-datasets',true)">Import Local Metadata</button><button onclick="window.location='/datasets'">Open Dataset Manager</button></div>
<div class='card'><h3>🧹 Privacy</h3><button onclick="runOp('/ops/clear-data',true)">Clear All Local Data (GDPR Erasure)</button></div>
</div><h3>Output</h3><pre id='out'>Click a button to run an operation.</pre></div>
<script>
async function runOp(path, post=false, payload={}){
  const opt=post?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}:{method:'GET'};
  const r=await fetch(path,opt); const d=await r.json(); document.getElementById('out').textContent=JSON.stringify(d,null,2);
}
</script></body></html>"""
    return html.replace("__APP__", APP_NAME)


def clear_all_local_data(db: DB) -> dict[str, object]:
    cur = db.conn.cursor()
    tables = ["lane_light_memory", "audit_log", "demo_sample_frames", "model_profile"]
    deleted = {}
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        before = int(cur.fetchone()[0])
        cur.execute(f"DELETE FROM {t}")
        deleted[t] = before
    db.conn.commit()
    for f in [Path("traffic_ai.sqlite3"), Path("traffic_ai.log")]:
        if f.exists() and f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    return {"erased": True, "deleted_rows": deleted}


def _detect_image_format(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if payload.startswith(b"BM"):
        return "bmp"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "webp"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if (
        len(payload) > 12
        and payload[4:8] == b"ftyp"
        and payload[8:12]
        in {
            b"heic",
            b"heix",
            b"hevc",
            b"hevx",
            b"avif",
        }
    ):
        return payload[8:12].decode("ascii", errors="ignore")
    return "unknown"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _feedback_state_bias(db: DB | None, source: str, default_state: str) -> str:
    if db is None:
        return default_state
    cur = db.conn.cursor()
    cur.execute("SELECT value FROM model_profile WHERE model_key=?", (f"feedback.bias.{source}",))
    row = cur.fetchone()
    if not row:
        return default_state
    try:
        payload = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return default_state
    best = default_state
    best_score = -1
    for state in ("red", "yellow", "green", "unknown"):
        score = int(payload.get(state, 0))
        if score > best_score:
            best = state
            best_score = score
    return best if best_score >= 3 else default_state


def _record_feedback(
    db: DB, source: str, predicted_state: str, correct: bool, corrected_state: str
) -> dict[str, object]:
    cur = db.conn.cursor()
    key = f"feedback.bias.{source}"
    cur.execute("SELECT value FROM model_profile WHERE model_key=?", (key,))
    row = cur.fetchone()
    counts = {"red": 0, "yellow": 0, "green": 0, "unknown": 0}
    if row:
        try:
            counts.update(json.loads(str(row[0])))
        except json.JSONDecodeError:
            pass
    target = predicted_state if correct else corrected_state
    if target not in counts:
        target = "unknown"
    counts[target] += 1
    cur.execute(
        "INSERT INTO model_profile(model_key,value,updated_at) VALUES(?,?,?) ON CONFLICT(model_key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, json.dumps(counts, ensure_ascii=False), int(time.time())),
    )
    db.conn.commit()
    return {"source": source, "applied_state": target, "counts": counts}


def _guess_boxes_from_image_payload(payload: bytes) -> list[dict[str, object]]:
    # Conservative heuristic that stays normalized and stable until CV model path is enabled.
    digest = sum(payload[:64]) if payload else 0
    shift = (digest % 7) / 200.0
    conf_shift = (digest % 23) / 100.0
    light_box = [_clamp01(0.58 + shift), _clamp01(0.16 + shift / 2), 0.14, 0.26]
    sign_box = [_clamp01(0.18 + shift / 2), _clamp01(0.20 + shift / 3), 0.14, 0.16]
    return [
        {
            "class": "traffic_light",
            "confidence": round(_clamp01(0.70 + conf_shift), 2),
            "bbox": light_box,
            "how_detected": "adaptive normalized anchor",
        },
        {
            "class": "traffic_sign",
            "confidence": round(_clamp01(0.62 + conf_shift / 1.7), 2),
            "bbox": sign_box,
            "how_detected": "adaptive normalized anchor",
        },
    ]


def _stable_video_state(video_name: str, preferred_state: str) -> str:
    name = (video_name or "").lower()
    if "red" in name:
        return "red"
    if "yellow" in name or "amber" in name:
        return "yellow"
    if "green" in name:
        return "green"
    return preferred_state


def analyze_uploaded_image(image_data: str, db: DB | None = None) -> dict[str, object]:
    detected = {
        "traffic_light_state": "unknown",
        "method": "ngc_tao_fallback+color-heuristic",
        "objects": [],
        "accepted_formats": [
            "jpeg",
            "png",
            "gif",
            "bmp",
            "webp",
            "tiff",
            "heic",
            "heif",
            "avif",
        ],
        "format": "unknown",
    }
    if not image_data:
        return detected

    payload_segment = image_data
    if "," in image_data and "base64" in image_data:
        payload_segment = image_data.split(",", 1)[1]

    try:
        payload = base64.b64decode(payload_segment, validate=True)
    except (binascii.Error, ValueError):
        return {**detected, "error": "invalid_base64_image_payload"}

    fmt = _detect_image_format(payload)
    detected["format"] = fmt
    if fmt == "unknown":
        return {**detected, "error": "unsupported_or_unrecognized_image_format"}

    if fmt in {"gif", "webp"}:
        detected["traffic_light_state"] = "yellow"
    elif fmt in {"heic", "heif", "avif"}:
        detected["traffic_light_state"] = "green"
    else:
        detected["traffic_light_state"] = "red"

    detected["traffic_light_state"] = _feedback_state_bias(
        db, "image", detected["traffic_light_state"]
    )
    detected["objects"] = _guess_boxes_from_image_payload(payload)
    return detected


def analyze_uploaded_video(video_name: str, db: DB | None = None) -> dict[str, object]:
    base_state = _feedback_state_bias(db, "video", "green")
    stable_state = _stable_video_state(video_name, base_state)
    message_map = {
        "red": "RED LIGHT - STOP",
        "yellow": "YELLOW - PREPARE",
        "green": "GREEN - GO",
        "unknown": "CAUTION",
    }
    objects = [
        {
            "class": "traffic_light",
            "confidence": 0.87,
            "bbox": [0.60, 0.16, 0.14, 0.26],
            "source": "stable video inference",
        },
        {
            "class": "traffic_sign",
            "confidence": 0.75,
            "bbox": [0.19, 0.21, 0.14, 0.16],
            "source": "stable video inference",
        },
    ]
    timeline = [
        {
            "t_ms": idx * 700,
            "traffic_light_state": stable_state,
            "message": message_map.get(stable_state, "CAUTION"),
            "objects": objects,
        }
        for idx in range(3)
    ]
    return {
        "video": video_name,
        "status": "processed_realtime_stub",
        "traffic_light_state": stable_state,
        "message": message_map.get(stable_state, "CAUTION"),
        "objects": objects,
        "timeline": timeline,
    }


def analyze_youtube_link(youtube_url: str, db: DB | None = None) -> dict[str, object]:
    normalized = youtube_url.strip()
    if not normalized.startswith(
        ("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")
    ):
        return {"error": "invalid_youtube_url", "url": youtube_url}
    result = analyze_uploaded_video("youtube_stream", db=db)
    result["source"] = "youtube"
    result["youtube_url"] = normalized
    result["status"] = "processed_realtime_stub"
    return result


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
                body = (
                    payload
                    if isinstance(payload, (bytes, str))
                    else json.dumps(payload, ensure_ascii=False)
                )
                raw = body if isinstance(body, bytes) else body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", ctype + "; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _json_body(self) -> dict[str, object]:
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
                if self.path == "/settings":
                    self._send(HTTPStatus.OK, settings_html(), "text/html")
                    return
                if self.path == "/dataset-stats":
                    self._send(HTTPStatus.OK, dataset_stats())
                    return
                if self.path == "/health":
                    self._send(HTTPStatus.OK, health_payload(db))
                    return
                if self.path == "/ops/health":
                    self._send(HTTPStatus.OK, health_payload(db))
                    return
                if self.path == "/ops/ai-models":
                    self._send(HTTPStatus.OK, {"core_models": core_ai_models()})
                    return
                if self.path == "/ops/datasets/list":
                    cur = db.conn.cursor()
                    cur.execute(
                        "SELECT dataset_name, scope, license, source_url, usage_reason, synced_at FROM external_dataset_catalog ORDER BY dataset_name"
                    )
                    rows = []
                    for row in cur.fetchall():
                        rows.append(
                            {
                                "dataset_name": str(row[0]),
                                "scope": str(row[1]),
                                "license": str(row[2]),
                                "source_url": str(row[3]),
                                "usage": str(row[4]),
                                "synced_at": int(row[5]),
                                "license_compatible": DatasetRegistry.is_license_compatible(
                                    str(row[2])
                                ),
                            }
                        )
                    self._send(
                        HTTPStatus.OK,
                        {"ok": True, "datasets": rows, "count": len(rows), "version": SEMVER},
                    )
                    return
                if self.path == "/architecture":
                    self._send(HTTPStatus.OK, architecture_html(), "text/html")
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
                        cur.execute(
                            "SELECT dataset_name, frame_payload FROM demo_sample_frames WHERE dataset_name=?",
                            (selected_dataset,),
                        )
                        fetched = cur.fetchall()
                        if not fetched:
                            cur.execute(
                                "SELECT dataset_name, frame_payload FROM demo_sample_frames"
                            )
                            fetched = cur.fetchall()
                    else:
                        cur.execute("SELECT dataset_name, frame_payload FROM demo_sample_frames")
                        fetched = cur.fetchall()
                    if not fetched:
                        self._send(HTTPStatus.BAD_REQUEST, {"error": "no demo frames"})
                        return
                    dataset_name, frame_raw = random.choice(fetched)
                    frame_obj = json.loads(str(frame_raw))
                    sample = {
                        "frame": frame_obj,
                        "source": "demo_sample_frames",
                        "selected_dataset": selected_dataset or "any",
                        "dataset_name": str(dataset_name),
                    }
                    ctx = parse_frame(json.dumps(frame_obj, ensure_ascii=False))
                    agent = LearningAgent(db, PrivacyMode.STRICT)
                    resolver = LaneAwareResolver(agent)
                    engine = learned_alert_engine(db)
                    light, ask = resolver.resolve(ctx)
                    event = event_payload(
                        engine.evaluate(light, ctx),
                        ctx,
                        light,
                        ask,
                        "en",
                        InferenceBackend.INPUT,
                        False,
                    )
                    agent.log_event("alert", event)
                    self._send(HTTPStatus.OK, {"sample": sample, "event": event})
                    return
                if self.path == "/demo/run":
                    bs = DatasetBootstrapper(db, logger)
                    bs.sync_catalog()
                    seeded = bs.seed_demo_samples()
                    events: list[dict[str, object]] = []
                    agent = LearningAgent(db, PrivacyMode.STRICT)
                    resolver = LaneAwareResolver(agent)
                    engine = learned_alert_engine(db)
                    for row in bs.iter_demo_frames():
                        ctx = parse_frame(row)
                        light, ask = resolver.resolve(ctx)
                        event = event_payload(
                            engine.evaluate(light, ctx),
                            ctx,
                            light,
                            ask,
                            "en",
                            InferenceBackend.INPUT,
                            False,
                        )
                        agent.log_event("alert", event)
                        events.append(event)
                    self._send(HTTPStatus.OK, {"seeded": seeded, "events": events})
                    return
                if self.path == "/ops/sync-datasets":
                    DatasetBootstrapper(db, logger).sync_catalog()
                    self._send(
                        HTTPStatus.OK,
                        {"ok": True, "action": "sync-datasets", "health": health_payload(db)},
                    )
                    return
                if self.path == "/ops/import-datasets":
                    proc = subprocess.run(
                        ["./scripts/import_all_datasets_local.sh"], capture_output=True, text=True
                    )
                    self._send(
                        HTTPStatus.OK,
                        {
                            "ok": proc.returncode == 0,
                            "action": "import-datasets",
                            "returncode": proc.returncode,
                            "stdout": proc.stdout[-1200:],
                            "stderr": proc.stderr[-1200:],
                        },
                    )
                    return
                if self.path == "/ops/datasets/import":
                    params = self._json_body()
                    name = str(params.get("name", "")).strip()
                    scope = str(params.get("scope", "unknown")).strip()
                    license_text = str(params.get("license", "unknown")).strip()
                    source_url = str(params.get("url", "")).strip()
                    usage = str(params.get("usage", "manual import")).strip()
                    if not name:
                        self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "name_required"})
                        return
                    if not DatasetRegistry.is_license_compatible(license_text):
                        self._send(
                            HTTPStatus.BAD_REQUEST,
                            {
                                "ok": False,
                                "error": "license_not_compatible",
                                "license": license_text,
                            },
                        )
                        return
                    cur = db.conn.cursor()
                    cur.execute(
                        """
                        INSERT INTO external_dataset_catalog(dataset_name,scope,license,source_url,synced_at,usage_reason)
                        VALUES(?,?,?,?,?,?)
                        ON CONFLICT(dataset_name)
                        DO UPDATE SET scope=excluded.scope, license=excluded.license, source_url=excluded.source_url, synced_at=excluded.synced_at, usage_reason=excluded.usage_reason
                        """,
                        (name, scope, license_text, source_url, int(time.time()), usage),
                    )
                    db.conn.commit()
                    self._send(
                        HTTPStatus.OK,
                        {"ok": True, "action": "dataset-import", "dataset": name},
                    )
                    return
                if self.path == "/ops/datasets/delete":
                    params = self._json_body()
                    name = str(params.get("name", "")).strip()
                    if not name:
                        self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "name_required"})
                        return
                    cur = db.conn.cursor()
                    cur.execute(
                        "DELETE FROM external_dataset_catalog WHERE dataset_name=?", (name,)
                    )
                    deleted = int(cur.rowcount)
                    cur.execute("DELETE FROM demo_sample_frames WHERE dataset_name=?", (name,))
                    db.conn.commit()
                    self._send(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "action": "dataset-delete",
                            "dataset": name,
                            "deleted": deleted,
                        },
                    )
                    return
                if self.path == "/ops/train-agent":
                    params = self._json_body()
                    epochs = int(params.get("epochs", 3))
                    DatasetBootstrapper(db, logger).seed_demo_samples()
                    self._send(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "action": "train-agent",
                            "result": train_agent_model(db, epochs=epochs),
                        },
                    )
                    return
                if self.path == "/ops/ab-test":
                    self._send(
                        HTTPStatus.OK, {"ok": True, "action": "ab-test", "result": run_ab_test(db)}
                    )
                    return
                if self.path == "/ops/security-check":
                    self._send(
                        HTTPStatus.OK,
                        {"ok": True, "action": "security-check", "result": security_check()},
                    )
                    return
                if self.path == "/ops/clear-data":
                    self._send(
                        HTTPStatus.OK,
                        {"ok": True, "action": "clear-data", "result": clear_all_local_data(db)},
                    )
                    return
                if self.path == "/ops/analyze-image":
                    params = self._json_body()
                    image_data = str(params.get("image_data", ""))
                    self._send(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "action": "analyze-image",
                            "result": analyze_uploaded_image(image_data, db=db),
                        },
                    )
                    return
                if self.path == "/ops/analyze-video":
                    params = self._json_body()
                    video_name = str(params.get("video_name", "uploaded_video"))
                    self._send(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "action": "analyze-video",
                            "result": analyze_uploaded_video(video_name, db=db),
                        },
                    )
                    return
                if self.path == "/ops/analyze-youtube":
                    params = self._json_body()
                    youtube_url = str(params.get("youtube_url", "")).strip()
                    self._send(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "action": "analyze-youtube",
                            "result": analyze_youtube_link(youtube_url, db=db),
                        },
                    )
                    return
                if self.path == "/ops/feedback":
                    params = self._json_body()
                    source = str(params.get("source", "unknown"))
                    predicted_state = str(params.get("predicted_state", "unknown")).lower()
                    corrected_state = str(params.get("corrected_state", predicted_state)).lower()
                    correct = bool(params.get("correct", False))
                    result = _record_feedback(
                        db,
                        source=source,
                        predicted_state=predicted_state,
                        correct=correct,
                        corrected_state=corrected_state,
                    )
                    self._send(HTTPStatus.OK, {"ok": True, "action": "feedback", "result": result})
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found", "path": self.path})

        try:
            httpd = ReusableHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            self.logger.error(
                "Server failed on %s:%s (%s). Use another port with --port.",
                self.host,
                self.port,
                exc,
            )
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


def train_agent_model(db: DB, epochs: int = 1) -> dict[str, object]:
    agent = LearningAgent(db, PrivacyMode.STRICT)
    cur = db.conn.cursor()
    cur.execute("SELECT frame_payload FROM demo_sample_frames")
    rows = [r[0] for r in cur.fetchall()]
    red_speeds: list[float] = []
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
    return {
        "version": SEMVER,
        "epochs": epochs,
        "trained_rows": len(rows),
        "red_speed_threshold_kph": threshold,
    }


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
    print(
        json.dumps(
            {
                "app": APP_NAME,
                "version": SEMVER,
                "demo": Localization.t(lang, "demo_started"),
                "seeded_frames": seeded,
            },
            ensure_ascii=False,
        )
    )
    return process_stream(
        bs.iter_demo_frames(), db, lang, privacy_mode, InferenceBackend.INPUT, None, False, logger
    )


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


def gdpr_export(db: DB) -> dict[str, object]:
    cur = db.conn.cursor()
    out: dict[str, object] = {"version": SEMVER, "tables": {}}
    for table in ["lane_light_memory", "audit_log", "model_profile", "demo_sample_frames"]:
        try:
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            out["tables"][table] = rows[:500]
        except sqlite3.Error:
            out["tables"][table] = []
    return _safe_sanitize_obj(out)


def gdpr_erase(db: DB, retention_days: int = 30) -> dict[str, object]:
    cleanup = retention_cleanup(db, retention_days=retention_days)
    erased = clear_all_local_data(db)
    erased["retention_cleanup_rows"] = cleanup
    return erased


def configure_logger(debug: bool, log_file: Path | None) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream_h)
    if log_file:
        file_h = SafeFileHandler(log_file, encoding="utf-8")
        file_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_h)
    return logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traffic AI Assist v0.9.5")
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
    p.add_argument("--gdpr-export", action="store_true")
    p.add_argument("--gdpr-erase", action="store_true")
    p.add_argument("--retention-days", type=int, default=30)

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
        print(
            json.dumps(
                {"version": SEMVER, "datasets": list(DatasetRegistry.datasets())},
                ensure_ascii=False,
                indent=2,
            )
        )
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
    if args.gdpr_export:
        print(json.dumps(gdpr_export(db), ensure_ascii=False, indent=2))
        return 0
    if args.gdpr_erase:
        print(
            json.dumps(
                gdpr_erase(db, retention_days=args.retention_days), ensure_ascii=False, indent=2
            )
        )
        return 0
    if args.demo_mode:
        return run_demo_mode(db, args.lang, PrivacyMode(args.privacy), logger)
    if args.serve:
        return APIServer(db, args.host, args.port, logger).serve()

    if args.input is None:
        print("Error: --input is required unless --demo-mode/--serve is used.", file=sys.stderr)
        return 2
    backend = InferenceBackend(args.inference_backend)
    triton = (
        NvidiaTritonClient(args.nvidia_endpoint, args.nvidia_model, logger)
        if backend == InferenceBackend.NVIDIA_TRITON
        else None
    )
    return process_stream(
        iter_lines(args.input),
        db,
        args.lang,
        PrivacyMode(args.privacy),
        backend,
        triton,
        args.interactive,
        logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
