#!/usr/bin/env python3
"""
Traffic AI Assist - Real Agent Core
Version: 0.3.0
License: MIT

Implements:
- Lane-aware traffic light safety agent
- External dataset catalog + local metadata bootstrap
- NVIDIA Triton embedded inference integration hook
- SQLite operational state and memory
- Docker/systemd log ingestion
- BDD/Gherkin export and self-test scenarios
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
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

APP_NAME = "traffic-ai-assist"
SEMVER = "0.3.0"


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
        },
        "fa": {
            "overspeed_red": "هشدار: چراغ قرمز جلو است. سرعت را کم کنید.",
            "red_crossed": "آژیر: عبور از چراغ قرمز تشخیص داده شد.",
            "green_wait": "چراغ سبز است. در صورت ایمن بودن حرکت کنید.",
            "select_light": "تشخیص چراغ مرتبط با باند نامشخص است. لطفاً چراغ صحیح را انتخاب کنید.",
            "pedestrian": "عابر پیاده تشخیص داده شد. با احتیاط حرکت کنید.",
            "ok": "هشدار فعالی وجود ندارد.",
        },
        "de": {
            "overspeed_red": "Warnung: Rote Ampel voraus. Geschwindigkeit reduzieren.",
            "red_crossed": "SIRENE: Rotlichtverstoß erkannt.",
            "green_wait": "Grün aktiv. Bitte sicher anfahren.",
            "select_light": "Spur-Ampel unklar. Bitte korrektes Signal wählen.",
            "pedestrian": "Fußgänger erkannt. Vorsichtig fahren.",
            "ok": "Keine aktive Warnung.",
        },
    }

    @classmethod
    def t(cls, lang: str, key: str) -> str:
        return cls.MESSAGES.get(lang, cls.MESSAGES["en"]).get(key, key)


class DatasetRegistry:
    """External base datasets for real-world traffic AI training/inference pipelines."""

    DATASETS: Tuple[Dict[str, str], ...] = (
        {
            "name": "BDD100K",
            "scope": "traffic lights, lanes, drivable area",
            "license": "Berkeley DeepDrive terms",
            "url": "https://bdd-data.berkeley.edu/",
        },
        {
            "name": "Bosch Small Traffic Lights",
            "scope": "traffic light detection",
            "license": "Bosch dataset terms",
            "url": "https://hci.iwr.uni-heidelberg.de/node/6132",
        },
        {
            "name": "LISA Traffic Light Dataset",
            "scope": "traffic lights",
            "license": "Academic usage terms",
            "url": "https://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html",
        },
        {
            "name": "Mapillary Traffic Sign Dataset",
            "scope": "road signs",
            "license": "Mapillary Vistas terms",
            "url": "https://www.mapillary.com/dataset/vistas",
        },
    )

    @classmethod
    def to_json(cls) -> str:
        return json.dumps({"datasets": list(cls.DATASETS)}, ensure_ascii=False, indent=2)


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
                synced_at INTEGER NOT NULL
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
                INSERT INTO external_dataset_catalog(dataset_name, scope, license, source_url, synced_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(dataset_name)
                DO UPDATE SET scope=excluded.scope, license=excluded.license, source_url=excluded.source_url, synced_at=excluded.synced_at
                """,
                (d["name"], d["scope"], d["license"], d["url"], now),
            )
        self.db.conn.commit()

    def fetch_remote_metadata(self, output_path: Path) -> None:
        """Downloads only lightweight metadata/landing pages, not full datasets."""
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
        cur.execute(
            "INSERT INTO audit_log(ts, event_type, payload) VALUES(?, ?, ?)",
            (int(time.time()), event_type, json.dumps(safe, ensure_ascii=False)),
        )
        self.db.conn.commit()


class NvidiaTritonClient:
    """HTTP client for NVIDIA embedded Triton inference server."""

    def __init__(self, endpoint: str, model_name: str, logger: logging.Logger) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.logger = logger

    def infer(self, inference_features: Dict[str, object]) -> Optional[Tuple[TrafficLightCandidate, ...]]:
        url = f"{self.endpoint}/v2/models/{self.model_name}/infer"
        body = json.dumps(inference_features).encode("utf-8")
        req = Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=3) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warning("Triton infer failed: %s", exc)
            return None

        candidates = raw.get("candidates", [])
        parsed: List[TrafficLightCandidate] = []
        for c in candidates:
            try:
                parsed.append(
                    TrafficLightCandidate(
                        light_id=str(c["light_id"]),
                        state=TrafficLightState(str(c.get("state", "unknown"))),
                        lane_ids=tuple(str(x) for x in c.get("lane_ids", [])),
                        confidence=float(c.get("confidence", 0.0)),
                    )
                )
            except (KeyError, ValueError):
                continue
        return tuple(parsed) if parsed else None


class LaneAwareResolver:
    def __init__(self, agent: LearningAgent) -> None:
        self.agent = agent

    def resolve(self, ctx: FrameContext) -> Tuple[Optional[TrafficLightCandidate], bool]:
        lane_match = [c for c in ctx.candidates if ctx.vehicle.lane_id in c.lane_ids]
        if lane_match:
            lane_match.sort(key=lambda c: c.confidence, reverse=True)
            pick = lane_match[0]
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


class LogIngestor:
    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, str]]:
        line = line.strip()
        if not line:
            return None
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return {"source": "unknown", "message": line}
            return {
                "source": "docker",
                "time": str(obj.get("time", "")),
                "stream": str(obj.get("stream", "stdout")),
                "message": str(obj.get("log", "")).strip(),
            }
        parts = line.split(maxsplit=5)
        if len(parts) >= 6:
            return {
                "source": "systemd",
                "timestamp": f"{parts[0]} {parts[1]} {parts[2]}",
                "host": parts[3],
                "unit": parts[4].rstrip(":"),
                "message": parts[5],
            }
        return {"source": "unknown", "message": line}


def parse_frame(raw: str) -> FrameContext:
    obj = json.loads(raw)
    cands = tuple(
        TrafficLightCandidate(
            light_id=str(c["light_id"]),
            state=TrafficLightState(c.get("state", "unknown")),
            lane_ids=tuple(str(x) for x in c.get("lane_ids", [])),
            confidence=float(c.get("confidence", 0.0)),
        )
        for c in obj.get("candidates", [])
    )
    vo = obj.get("vehicle", {})
    eo = obj.get("extra", {})
    return FrameContext(
        route_id=str(obj.get("route_id", "default")),
        timestamp_ms=int(obj.get("timestamp_ms", int(time.time() * 1000))),
        candidates=cands,
        vehicle=VehicleState(
            speed_kph=float(vo.get("speed_kph", 0.0)),
            lane_id=str(vo.get("lane_id", "unknown")),
            crossed_stop_line=bool(vo.get("crossed_stop_line", False)),
            stationary_seconds=float(vo.get("stationary_seconds", 0.0)),
        ),
        extra=ExtraRoadContext(
            pedestrian_detected=bool(eo.get("pedestrian_detected", False)),
            road_signs=tuple(str(x) for x in eo.get("road_signs", [])),
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
        "extra": {
            "pedestrian_detected": ctx.extra.pedestrian_detected,
            "road_signs": list(ctx.extra.road_signs),
        },
    }


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
            triton_cands = triton.infer({
                "vehicle": {
                    "lane_id": ctx.vehicle.lane_id,
                    "speed_kph": ctx.vehicle.speed_kph,
                },
                "meta": {"route_id": ctx.route_id, "timestamp_ms": ctx.timestamp_ms},
            })
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


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


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
Feature: Smart lane-aware traffic AI agent
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

  Scenario: Lane mismatch
    Given no lane-specific light exists
    When event is evaluated
    Then user selection is required
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
    ae = AlertEngine()

    samples: Sequence[FrameContext] = (
        FrameContext("R1", 1, (TrafficLightCandidate("A", TrafficLightState.RED, ("L1",), 0.9),), VehicleState(42.0, "L1", False, 0), ExtraRoadContext(False, tuple())),
        FrameContext("R1", 2, (TrafficLightCandidate("A", TrafficLightState.RED, ("L1",), 0.9),), VehicleState(10.0, "L1", True, 0), ExtraRoadContext(False, tuple())),
        FrameContext("R1", 3, (TrafficLightCandidate("B", TrafficLightState.GREEN, ("L2",), 0.8),), VehicleState(0.0, "L1", False, 5), ExtraRoadContext(False, tuple())),
    )
    expected = ["overspeed_red", "red_crossed", "select_light"]
    got: List[str] = []
    for s in samples:
        light, _ask = resolver.resolve(s)
        got.append(ae.evaluate(light, s).key)

    ok = got == expected
    print("BDD Feature:\n" + BDD_FEATURE)
    print("Self-test:", "PASS" if ok else "FAIL")

    db_path.unlink(missing_ok=True)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traffic AI Assist real agent core")
    p.add_argument("--version", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--export-gherkin", type=Path)
    p.add_argument("--parse-log-line", type=str)
    p.add_argument("--dataset-manifest", action="store_true")
    p.add_argument("--sync-datasets", action="store_true", help="Store external dataset catalog in SQLite")
    p.add_argument("--fetch-dataset-metadata", type=Path, help="Write remote metadata samples to JSON")

    p.add_argument("--input", type=Path)
    p.add_argument("--db", type=Path, default=Path("./traffic_ai.sqlite3"))
    p.add_argument("--lang", default="en", choices=["en", "fa", "de"])
    p.add_argument("--privacy", default="strict", choices=["strict", "balanced"])
    p.add_argument("--inference-backend", default="input", choices=["input", "nvidia_triton"])
    p.add_argument("--nvidia-endpoint", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--nvidia-model", type=str, default="traffic_light_detector")
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

    if args.export_gherkin:
        export_gherkin(args.export_gherkin)
        print(str(args.export_gherkin))
        return 0

    if args.self_test:
        return run_self_test()

    if args.parse_log_line is not None:
        print(json.dumps(LogIngestor.parse_line(args.parse_log_line), ensure_ascii=False))
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

    if args.input is None:
        print("Error: --input is required for stream processing.", file=sys.stderr)
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
