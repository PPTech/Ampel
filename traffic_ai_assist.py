#!/usr/bin/env python3
"""
Traffic AI Assist - Real Agent Core
Version: 0.4.0
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
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

APP_NAME = "traffic-ai-assist"
SEMVER = "0.4.0"


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
        "de": {
            "overspeed_red": "Warnung: Rote Ampel voraus. Geschwindigkeit reduzieren.",
            "red_crossed": "SIRENE: Rotlichtverstoß erkannt.",
            "green_wait": "Grün aktiv. Bitte sicher anfahren.",
            "select_light": "Spur-Ampel unklar. Bitte korrektes Signal wählen.",
            "pedestrian": "Fußgänger erkannt. Vorsichtig fahren.",
            "ok": "Keine aktive Warnung.",
            "demo_started": "Demomodus mit freien Beispieldaten gestartet.",
        },
    }

    @classmethod
    def t(cls, lang: str, key: str) -> str:
        return cls.MESSAGES.get(lang, cls.MESSAGES["en"]).get(key, key)


class DatasetRegistry:
    """External base datasets and their license context."""

    DATASETS: Tuple[Dict[str, str], ...] = (
        {
            "name": "BDD100K",
            "scope": "traffic lights, lanes, drivable area",
            "license": "Berkeley DeepDrive dataset terms",
            "url": "https://bdd-data.berkeley.edu/",
            "usage": "Dataset metadata catalog and demo scenario taxonomy.",
        },
        {
            "name": "Bosch Small Traffic Lights",
            "scope": "traffic light detection",
            "license": "Bosch dataset terms",
            "url": "https://hci.iwr.uni-heidelberg.de/node/6132",
            "usage": "Traffic-light class vocabulary and confidence fields.",
        },
        {
            "name": "LISA Traffic Light Dataset",
            "scope": "traffic lights",
            "license": "Academic usage terms",
            "url": "https://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html",
            "usage": "Signal-state behavior examples for red/green transitions.",
        },
        {
            "name": "Mapillary Traffic Sign Dataset",
            "scope": "road signs",
            "license": "Mapillary Vistas terms",
            "url": "https://www.mapillary.com/dataset/vistas",
            "usage": "Road-sign labels in extra context fields.",
        },
    )

    @classmethod
    def to_json(cls) -> str:
        return json.dumps({"version": SEMVER, "datasets": list(cls.DATASETS)}, ensure_ascii=False, indent=2)


def free_demo_samples() -> Tuple[Dict[str, object], ...]:
    """Synthetic sample frames derived from public dataset taxonomies (license-safe, no raw media)."""

    now = int(time.time() * 1000)
    return (
        {
            "sample_id": "demo-001",
            "dataset_name": "BDD100K",
            "dataset_license": "Berkeley DeepDrive dataset terms",
            "note": "Urban straight lane with red light and overspeed risk.",
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
            "note": "Red light violation scenario to trigger siren.",
            "frame": {
                "route_id": "berlin-city-center",
                "timestamp_ms": now + 1000,
                "candidates": [{"light_id": "TL-001", "state": "red", "lane_ids": ["lane-1"], "confidence": 0.95}],
                "vehicle": {"speed_kph": 20, "lane_id": "lane-1", "crossed_stop_line": True, "stationary_seconds": 0},
                "extra": {"pedestrian_detected": False, "road_signs": ["stop"]},
            },
        },
        {
            "sample_id": "demo-003",
            "dataset_name": "LISA Traffic Light Dataset",
            "dataset_license": "Academic usage terms",
            "note": "Green but stationary reminder case.",
            "frame": {
                "route_id": "berlin-city-center",
                "timestamp_ms": now + 2000,
                "candidates": [{"light_id": "TL-002", "state": "green", "lane_ids": ["lane-1"], "confidence": 0.92}],
                "vehicle": {"speed_kph": 0, "lane_id": "lane-1", "crossed_stop_line": False, "stationary_seconds": 6},
                "extra": {"pedestrian_detected": False, "road_signs": ["go_straight"]},
            },
        },
        {
            "sample_id": "demo-004",
            "dataset_name": "Mapillary Traffic Sign Dataset",
            "dataset_license": "Mapillary Vistas terms",
            "note": "Pedestrian alert with road-sign context.",
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
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lane_light_memory (
                route_id TEXT NOT NULL,
                lane_id TEXT NOT NULL,
                light_id TEXT NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY(route_id, lane_id, light_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                ts INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS external_dataset_catalog (
                dataset_name TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                license TEXT NOT NULL,
                source_url TEXT NOT NULL,
                usage_reason TEXT NOT NULL,
                synced_at INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS demo_sample_frames (
                sample_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                dataset_license TEXT NOT NULL,
                note TEXT NOT NULL,
                frame_payload TEXT NOT NULL,
                inserted_at INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()


class DatasetBootstrapper:
    def __init__(self, db: DB, logger: logging.Logger) -> None:
        self.db = db
        self.logger = logger

    def sync_catalog(self) -> None:
        now = int(time.time())
        cur = self.db.conn.cursor()
        for d in DatasetRegistry.DATASETS:
            cur.execute(
                """
                INSERT INTO external_dataset_catalog(dataset_name, scope, license, source_url, usage_reason, synced_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_name)
                DO UPDATE SET
                    scope=excluded.scope,
                    license=excluded.license,
                    source_url=excluded.source_url,
                    usage_reason=excluded.usage_reason,
                    synced_at=excluded.synced_at
                """,
                (d["name"], d["scope"], d["license"], d["url"], d["usage"], now),
            )
        self.db.conn.commit()

    def fetch_remote_metadata(self, output_path: Path) -> None:
        payload: List[Dict[str, str]] = []
        for d in DatasetRegistry.DATASETS:
            try:
                req = Request(d["url"], headers={"User-Agent": f"{APP_NAME}/{SEMVER}"}, method="GET")
                with urlopen(req, timeout=8) as resp:
                    snippet = resp.read(512).decode("utf-8", errors="replace")
                payload.append({"name": d["name"], "url": d["url"], "sample": snippet})
            except (URLError, HTTPError, TimeoutError) as exc:
                payload.append({"name": d["name"], "url": d["url"], "error": str(exc)})
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Dataset metadata exported to %s", output_path)

    def seed_demo_samples(self) -> int:
        now = int(time.time())
        cur = self.db.conn.cursor()
        count = 0
        for sample in free_demo_samples():
            cur.execute(
                """
                INSERT INTO demo_sample_frames(sample_id, dataset_name, dataset_license, note, frame_payload, inserted_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id)
                DO UPDATE SET
                    dataset_name=excluded.dataset_name,
                    dataset_license=excluded.dataset_license,
                    note=excluded.note,
                    frame_payload=excluded.frame_payload,
                    inserted_at=excluded.inserted_at
                """,
                (
                    str(sample["sample_id"]),
                    str(sample["dataset_name"]),
                    str(sample["dataset_license"]),
                    str(sample["note"]),
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


class LearningAgent:
    def __init__(self, db: DB, privacy_mode: PrivacyMode) -> None:
        self.db = db
        self.privacy_mode = privacy_mode

    def remember(self, route_id: str, lane_id: str, light_id: str) -> None:
        cur = self.db.conn.cursor()
        cur.execute(
            """
            INSERT INTO lane_light_memory(route_id, lane_id, light_id, seen_count, updated_at)
            VALUES(?, ?, ?, 1, ?)
            ON CONFLICT(route_id, lane_id, light_id)
            DO UPDATE SET seen_count = seen_count + 1, updated_at = excluded.updated_at
            """,
            (route_id, lane_id, light_id, int(time.time())),
        )
        self.db.conn.commit()

    def suggest(self, route_id: str, lane_id: str) -> Optional[str]:
        cur = self.db.conn.cursor()
        cur.execute(
            """
            SELECT light_id FROM lane_light_memory
            WHERE route_id = ? AND lane_id = ?
            ORDER BY seen_count DESC, updated_at DESC
            LIMIT 1
            """,
            (route_id, lane_id),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def log_event(self, event_type: str, payload: Dict[str, object]) -> None:
        safe = {"mode": "strict", "event_type": event_type} if self.privacy_mode == PrivacyMode.STRICT else payload
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO audit_log(ts, event_type, payload) VALUES(?, ?, ?)", (int(time.time()), event_type, json.dumps(safe, ensure_ascii=False)))
        self.db.conn.commit()


class NvidiaTritonClient:
    def __init__(self, endpoint: str, model_name: str, logger: logging.Logger) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.logger = logger

    def infer(self, inference_features: Dict[str, object]) -> Optional[Tuple[TrafficLightCandidate, ...]]:
        url = f"{self.endpoint}/v2/models/{self.model_name}/infer"
        req = Request(url, data=json.dumps(inference_features).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=3) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warning("Triton infer failed: %s", exc)
            return None

        out: List[TrafficLightCandidate] = []
        for c in raw.get("candidates", []):
            try:
                out.append(
                    TrafficLightCandidate(
                        light_id=str(c["light_id"]),
                        state=TrafficLightState(str(c.get("state", "unknown"))),
                        lane_ids=tuple(str(x) for x in c.get("lane_ids", [])),
                        confidence=float(c.get("confidence", 0.0)),
                    )
                )
            except (KeyError, ValueError):
                continue
        return tuple(out) if out else None


class LaneAwareResolver:
    def __init__(self, agent: LearningAgent) -> None:
        self.agent = agent

    def resolve(self, ctx: FrameContext) -> Tuple[Optional[TrafficLightCandidate], bool]:
        lane_match = [c for c in ctx.candidates if ctx.vehicle.lane_id in c.lane_ids]
        if lane_match:
            lane_match.sort(key=lambda c: c.confidence, reverse=True)
            selected = lane_match[0]
            self.agent.remember(ctx.route_id, ctx.vehicle.lane_id, selected.light_id)
            return selected, False

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


class LogIngestor:
    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, str]]:
        text = line.strip()
        if not text:
            return None
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return {"source": "unknown", "message": text}
            return {
                "source": "docker",
                "time": str(obj.get("time", "")),
                "stream": str(obj.get("stream", "stdout")),
                "message": str(obj.get("log", "")).strip(),
            }
        parts = text.split(maxsplit=5)
        if len(parts) >= 6:
            return {
                "source": "systemd",
                "timestamp": f"{parts[0]} {parts[1]} {parts[2]}",
                "host": parts[3],
                "unit": parts[4].rstrip(":"),
                "message": parts[5],
            }
        return {"source": "unknown", "message": text}


def parse_frame(raw: str) -> FrameContext:
    obj = json.loads(raw)
    candidates = tuple(
        TrafficLightCandidate(
            light_id=str(c["light_id"]),
            state=TrafficLightState(str(c.get("state", "unknown"))),
            lane_ids=tuple(str(x) for x in c.get("lane_ids", [])),
            confidence=float(c.get("confidence", 0.0)),
        )
        for c in obj.get("candidates", [])
    )
    vehicle = obj.get("vehicle", {})
    extra = obj.get("extra", {})
    return FrameContext(
        route_id=str(obj.get("route_id", "default")),
        timestamp_ms=int(obj.get("timestamp_ms", int(time.time() * 1000))),
        candidates=candidates,
        vehicle=VehicleState(
            speed_kph=float(vehicle.get("speed_kph", 0.0)),
            lane_id=str(vehicle.get("lane_id", "unknown")),
            crossed_stop_line=bool(vehicle.get("crossed_stop_line", False)),
            stationary_seconds=float(vehicle.get("stationary_seconds", 0.0)),
        ),
        extra=ExtraRoadContext(
            pedestrian_detected=bool(extra.get("pedestrian_detected", False)),
            road_signs=tuple(str(x) for x in extra.get("road_signs", [])),
        ),
    )


def to_event(alert: Alert, ctx: FrameContext, light: Optional[TrafficLightCandidate], ask: bool, lang: str, backend: InferenceBackend) -> Dict[str, object]:
    return {
        "app": APP_NAME,
        "version": SEMVER,
        "backend": backend.value,
        "route_id": ctx.route_id,
        "timestamp_ms": ctx.timestamp_ms,
        "lane_id": ctx.vehicle.lane_id,
        "light_id": light.light_id if light else None,
        "light_state": light.state.value if light else None,
        "alert": alert.key,
        "channel": alert.channel.value,
        "message": Localization.t(lang, alert.key),
        "ask_user_selection": ask,
        "map_overlay": {
            "show_camera_preview": True,
            "show_signal_state": True,
            "show_alert": alert.channel != AlertChannel.NONE,
        },
        "extra": {"pedestrian_detected": ctx.extra.pedestrian_detected, "road_signs": list(ctx.extra.road_signs)},
    }


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if row:
                yield row


def process_stream(
    lines: Iterable[str],
    db: DB,
    lang: str,
    privacy_mode: PrivacyMode,
    backend: InferenceBackend,
    triton: Optional[NvidiaTritonClient],
    interactive: bool,
    logger: logging.Logger,
) -> int:
    agent = LearningAgent(db, privacy_mode)
    resolver = LaneAwareResolver(agent)
    alert_engine = AlertEngine()

    for line in lines:
        try:
            ctx = parse_frame(line)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error("Invalid frame: %s", exc)
            continue

        if backend == InferenceBackend.NVIDIA_TRITON and triton is not None:
            triton_cands = triton.infer(
                {
                    "vehicle": {"lane_id": ctx.vehicle.lane_id, "speed_kph": ctx.vehicle.speed_kph},
                    "meta": {"route_id": ctx.route_id, "timestamp_ms": ctx.timestamp_ms},
                }
            )
            if triton_cands:
                ctx = FrameContext(ctx.route_id, ctx.timestamp_ms, triton_cands, ctx.vehicle, ctx.extra)

        light, ask = resolver.resolve(ctx)

        if ask and interactive and ctx.candidates:
            print("Unable to resolve lane-specific light. Select index:")
            for i, c in enumerate(ctx.candidates):
                print(f"[{i}] id={c.light_id} state={c.state.value} lanes={','.join(c.lane_ids)}")
            selected_raw = sys.stdin.readline().strip()
            if selected_raw.isdigit():
                idx = int(selected_raw)
                if 0 <= idx < len(ctx.candidates):
                    light = ctx.candidates[idx]
                    ask = False
                    agent.remember(ctx.route_id, ctx.vehicle.lane_id, light.light_id)

        alert = alert_engine.evaluate(light, ctx)
        event = to_event(alert, ctx, light, ask, lang, backend)
        agent.log_event("alert", event)
        print(json.dumps(event, ensure_ascii=False))

    return 0


def export_demo_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in free_demo_samples():
            handle.write(json.dumps(sample["frame"], ensure_ascii=False) + "\n")


def run_demo_mode(db: DB, lang: str, privacy_mode: PrivacyMode, logger: logging.Logger) -> int:
    bootstrapper = DatasetBootstrapper(db, logger)
    seeded = bootstrapper.seed_demo_samples()
    print(json.dumps({"app": APP_NAME, "version": SEMVER, "demo": Localization.t(lang, "demo_started"), "seeded_frames": seeded}, ensure_ascii=False))
    return process_stream(
        lines=bootstrapper.iter_demo_frames(),
        db=db,
        lang=lang,
        privacy_mode=privacy_mode,
        backend=InferenceBackend.INPUT,
        triton=None,
        interactive=False,
        logger=logger,
    )


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
Feature: Smart lane-aware traffic AI agent with demo database
  Scenario: Free sample data demo mode
    Given demo samples are seeded from free dataset taxonomies
    When demo mode is executed
    Then alert events are emitted and persisted in SQLite

  Scenario: Overspeed under red light
    Given speed is above red threshold
    And lane matched light is red
    When event is evaluated
    Then audio warning is emitted

  Scenario: Crossing stop-line under red
    Given vehicle crossed stop line
    And lane matched light is red
    When event is evaluated
    Then siren warning is emitted
""".strip()


def export_gherkin(path: Path) -> None:
    path.write_text(BDD_FEATURE + "\n", encoding="utf-8")


def run_self_test() -> int:
    db_path = Path("./.traffic_ai_test.sqlite3")
    if db_path.exists():
        db_path.unlink()

    db = DB(db_path)
    agent = LearningAgent(db, PrivacyMode.STRICT)
    resolver = LaneAwareResolver(agent)
    engine = AlertEngine()

    frames: Sequence[FrameContext] = (
        FrameContext("R1", 1, (TrafficLightCandidate("A", TrafficLightState.RED, ("L1",), 0.95),), VehicleState(45.0, "L1", False, 0), ExtraRoadContext(False, tuple())),
        FrameContext("R1", 2, (TrafficLightCandidate("A", TrafficLightState.RED, ("L1",), 0.95),), VehicleState(12.0, "L1", True, 0), ExtraRoadContext(False, tuple())),
        FrameContext("R1", 3, (TrafficLightCandidate("B", TrafficLightState.GREEN, ("L2",), 0.88),), VehicleState(0.0, "L1", False, 5), ExtraRoadContext(False, tuple())),
    )
    expected = ["overspeed_red", "red_crossed", "select_light"]
    got: List[str] = []
    for frame in frames:
        light, _ = resolver.resolve(frame)
        got.append(engine.evaluate(light, frame).key)

    bootstrapper = DatasetBootstrapper(db, logging.getLogger(APP_NAME))
    demo_seeded = bootstrapper.seed_demo_samples()
    ok = got == expected and demo_seeded >= 4
    print("BDD Feature:\n" + BDD_FEATURE)
    print("Self-test:", "PASS" if ok else "FAIL")
    db_path.unlink(missing_ok=True)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic AI Assist real agent core")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--export-gherkin", type=Path)
    parser.add_argument("--parse-log-line", type=str)
    parser.add_argument("--dataset-manifest", action="store_true")
    parser.add_argument("--sync-datasets", action="store_true")
    parser.add_argument("--fetch-dataset-metadata", type=Path)
    parser.add_argument("--export-demo-sample", type=Path, help="Write free sample JSONL file")
    parser.add_argument("--demo-mode", action="store_true", help="Seed free sample data into DB and run live demo output")

    parser.add_argument("--input", type=Path)
    parser.add_argument("--db", type=Path, default=Path("./traffic_ai.sqlite3"))
    parser.add_argument("--lang", default="en", choices=["en", "fa", "de"])
    parser.add_argument("--privacy", default="strict", choices=["strict", "balanced"])
    parser.add_argument("--inference-backend", default="input", choices=["input", "nvidia_triton"])
    parser.add_argument("--nvidia-endpoint", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--nvidia-model", type=str, default="traffic_light_detector")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-file", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = configure_logger(args.debug, args.log_file)

    if args.version:
        print(f"{APP_NAME} {SEMVER}")
        return 0

    if args.export_gherkin:
        export_gherkin(args.export_gherkin)
        print(str(args.export_gherkin))
        return 0

    if args.self_test:
        return run_self_test()

    if args.parse_log_line is not None:
        print(json.dumps(LogIngestor.parse_line(args.parse_log_line), ensure_ascii=False))
        return 0

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
        bootstrapper.sync_catalog()
        return run_demo_mode(db, args.lang, PrivacyMode(args.privacy), logger)

    if args.input is None:
        print("Error: --input is required for stream processing unless --demo-mode is enabled.", file=sys.stderr)
        return 2

    backend = InferenceBackend(args.inference_backend)
    triton = NvidiaTritonClient(args.nvidia_endpoint, args.nvidia_model, logger) if backend == InferenceBackend.NVIDIA_TRITON else None
    return process_stream(
        lines=iter_lines(args.input),
        db=db,
        lang=args.lang,
        privacy_mode=PrivacyMode(args.privacy),
        backend=backend,
        triton=triton,
        interactive=bool(args.interactive),
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
