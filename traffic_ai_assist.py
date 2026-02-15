#!/usr/bin/env python3
"""
Traffic AI Assist (Integrated Runnable Program)
Version: 0.2.0 (SemVer)
License: MIT

A dependency-minimal, privacy-aware, lane-aware traffic safety assistant core that can run:
- as a CLI stream processor
- as an HTTP service for mobile apps (CarPlay / Android Auto adapters)

Design goals:
- secure data handling (no shell execution, parameterized SQL)
- low CPU/RAM (stdlib only, lightweight rules)
- extensible AI-agent memory and user interaction
- BDD/Gherkin aligned validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SEMVER = "0.2.0"


class TrafficLightState(str, Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    UNKNOWN = "unknown"


class AlertType(str, Enum):
    NONE = "none"
    VISUAL = "visual"
    AUDIO = "audio"
    SIREN = "siren"


class PrivacyMode(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"


@dataclass
class TrafficLightCandidate:
    light_id: str
    state: TrafficLightState
    lane_ids: Tuple[str, ...]
    confidence: float


@dataclass
class VehicleState:
    speed_kph: float
    lane_id: str
    crossed_stop_line: bool
    stationary_seconds: float


@dataclass
class CameraMeta:
    angle_score: float
    focus_score: float
    fps: int


@dataclass
class FrameContext:
    route_id: str
    timestamp_ms: int
    candidates: List[TrafficLightCandidate]
    vehicle: VehicleState
    camera: CameraMeta
    signs: List[str]
    pedestrians_detected: bool


@dataclass
class Alert:
    key: str
    severity: AlertType


class Localization:
    MESSAGES: Dict[str, Dict[str, str]] = {
        "en": {
            "overspeed_red": "Warning: Red light ahead. Reduce speed now.",
            "red_crossed": "SIREN: Red light violation detected.",
            "green_wait": "Green light is active. Please move if safe.",
            "select_light": "Unable to resolve lane-specific light. Please select the correct signal.",
            "pedestrian_warning": "Pedestrian nearby. Drive carefully.",
            "ok": "No active warning.",
        },
        "de": {
            "overspeed_red": "Warnung: Rote Ampel voraus. Geschwindigkeit sofort reduzieren.",
            "red_crossed": "SIRENE: Rotlichtverstoß erkannt.",
            "green_wait": "Grüne Ampel aktiv. Bitte sicher anfahren.",
            "select_light": "Spurspezifisches Signal unklar. Bitte korrektes Signal auswählen.",
            "pedestrian_warning": "Fußgänger in der Nähe. Vorsichtig fahren.",
            "ok": "Keine aktive Warnung.",
        },
        "fa": {
            "overspeed_red": "هشدار: چراغ قرمز جلو است. فوراً سرعت را کم کنید.",
            "red_crossed": "آژیر: عبور از چراغ قرمز تشخیص داده شد.",
            "green_wait": "چراغ سبز است. در صورت ایمن بودن حرکت کنید.",
            "select_light": "تشخیص چراغ مرتبط با باند ممکن نیست. لطفاً چراغ صحیح را انتخاب کنید.",
            "pedestrian_warning": "عابر نزدیک است. با احتیاط رانندگی کنید.",
            "ok": "هشدار فعالی وجود ندارد.",
        },
    }

    @classmethod
    def t(cls, lang: str, key: str) -> str:
        return cls.MESSAGES.get(lang, cls.MESSAGES["en"]).get(key, key)


class LogIngestor:
    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, str]]:
        s = line.strip()
        if not s:
            return None
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                return {"source": "unknown", "message": s}
            return {
                "source": "docker",
                "message": str(obj.get("log", "")).strip(),
                "stream": str(obj.get("stream", "stdout")),
                "time": str(obj.get("time", "")),
            }
        parts = s.split(maxsplit=5)
        if len(parts) >= 6:
            return {
                "source": "systemd",
                "timestamp": f"{parts[0]} {parts[1]} {parts[2]}",
                "host": parts[3],
                "unit": parts[4].rstrip(":"),
                "message": parts[5],
            }
        return {"source": "unknown", "message": s}


class LearningAgent:
    """On-device learning memory + simple interactive policy agent."""

    def __init__(self, db_path: Path, privacy_mode: PrivacyMode = PrivacyMode.STRICT) -> None:
        self.db_path = db_path
        self.privacy_mode = privacy_mode
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS lane_light_memory (
                    route_id TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    light_id TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (route_id, lane_id, light_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS route_pattern_memory (
                    route_id TEXT PRIMARY KEY,
                    pass_count INTEGER NOT NULL DEFAULT 1,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_log (
                    ts INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            self.conn.commit()

    @staticmethod
    def _anonymize_user_id(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]

    def remember_lane_light(self, route_id: str, lane_id: str, light_id: str) -> None:
        ts = int(time.time())
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO lane_light_memory(route_id, lane_id, light_id, seen_count, updated_at)
                VALUES(?, ?, ?, 1, ?)
                ON CONFLICT(route_id, lane_id, light_id)
                DO UPDATE SET seen_count = seen_count + 1, updated_at = excluded.updated_at
                """,
                (route_id, lane_id, light_id, ts),
            )
            cur.execute(
                """
                INSERT INTO route_pattern_memory(route_id, pass_count, updated_at)
                VALUES(?, 1, ?)
                ON CONFLICT(route_id)
                DO UPDATE SET pass_count = pass_count + 1, updated_at = excluded.updated_at
                """,
                (route_id, ts),
            )
            self.conn.commit()

    def suggest_light(self, route_id: str, lane_id: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT light_id
                FROM lane_light_memory
                WHERE route_id = ? AND lane_id = ?
                ORDER BY seen_count DESC, updated_at DESC
                LIMIT 1
                """,
                (route_id, lane_id),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def audit_event(self, event_type: str, payload: Dict[str, object], user_id: Optional[str] = None) -> None:
        safe = dict(payload)
        if user_id:
            safe["user_id"] = self._anonymize_user_id(user_id)
        if self.privacy_mode == PrivacyMode.STRICT:
            safe = {"mode": "strict", "event_type": event_type}
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO interaction_log(ts, event_type, payload) VALUES(?, ?, ?)",
                (int(time.time()), event_type, json.dumps(safe, ensure_ascii=False)),
            )
            self.conn.commit()

    def chat(self, user_text: str, lang: str = "en") -> str:
        text = user_text.lower()
        if "privacy" in text:
            return {
                "en": "Privacy mode is active. Raw camera frames are not persisted by default.",
                "de": "Der Datenschutzmodus ist aktiv. Rohkameradaten werden standardmäßig nicht gespeichert.",
                "fa": "حالت حریم خصوصی فعال است و فریم خام دوربین به طور پیش‌فرض ذخیره نمی‌شود.",
            }.get(lang, "Privacy mode is active.")
        if "status" in text or "health" in text:
            return {
                "en": "System is running. Lane-memory and alert engine are active.",
                "de": "System läuft. Spur-Speicher und Alarm-Engine sind aktiv.",
                "fa": "سیستم فعال است و حافظه باند و موتور هشدار کار می‌کند.",
            }.get(lang, "System is running.")
        return {
            "en": "I can help with lane signal selection, safety alerts, and privacy settings.",
            "de": "Ich kann bei Spur-Signalwahl, Sicherheitswarnungen und Datenschutzeinstellungen helfen.",
            "fa": "می‌توانم در انتخاب چراغ باند، هشدار ایمنی و تنظیمات حریم خصوصی کمک کنم.",
        }.get(lang, "I can help.")


class LaneAwareResolver:
    def __init__(self, agent: LearningAgent, min_confidence: float = 0.4) -> None:
        self.agent = agent
        self.min_confidence = min_confidence

    def resolve_light(self, ctx: FrameContext) -> Tuple[Optional[TrafficLightCandidate], bool]:
        matches = [c for c in ctx.candidates if ctx.vehicle.lane_id in c.lane_ids and c.confidence >= self.min_confidence]
        if matches:
            matches.sort(key=lambda c: c.confidence, reverse=True)
            choice = matches[0]
            self.agent.remember_lane_light(ctx.route_id, ctx.vehicle.lane_id, choice.light_id)
            return choice, False

        suggested = self.agent.suggest_light(ctx.route_id, ctx.vehicle.lane_id)
        if suggested:
            for c in ctx.candidates:
                if c.light_id == suggested:
                    return c, False

        return None, True


class AlertEngine:
    def __init__(self, overspeed_threshold_red_kph: float = 25.0, green_wait_threshold_s: float = 4.0) -> None:
        self.overspeed_threshold_red_kph = overspeed_threshold_red_kph
        self.green_wait_threshold_s = green_wait_threshold_s

    def evaluate(self, light: Optional[TrafficLightCandidate], ctx: FrameContext) -> Alert:
        if ctx.pedestrians_detected and ctx.vehicle.speed_kph > 10:
            return Alert("pedestrian_warning", AlertType.AUDIO)
        if light is None:
            return Alert("select_light", AlertType.VISUAL)
        if light.state == TrafficLightState.RED and ctx.vehicle.crossed_stop_line:
            return Alert("red_crossed", AlertType.SIREN)
        if light.state == TrafficLightState.RED and ctx.vehicle.speed_kph > self.overspeed_threshold_red_kph:
            return Alert("overspeed_red", AlertType.AUDIO)
        if light.state == TrafficLightState.GREEN and ctx.vehicle.stationary_seconds > self.green_wait_threshold_s:
            return Alert("green_wait", AlertType.VISUAL)
        return Alert("ok", AlertType.NONE)


class DecisionEngine:
    """Main orchestrator to process a frame and produce map/device overlays."""

    def __init__(self, agent: LearningAgent, lang: str = "en") -> None:
        self.agent = agent
        self.lang = lang
        self.resolver = LaneAwareResolver(agent)
        self.alerts = AlertEngine()

    def process_frame(self, ctx: FrameContext) -> Dict[str, object]:
        chosen, needs_user_selection = self.resolver.resolve_light(ctx)
        alert = self.alerts.evaluate(chosen, ctx)

        decision = {
            "route_id": ctx.route_id,
            "lane_id": ctx.vehicle.lane_id,
            "camera_quality": round((ctx.camera.angle_score + ctx.camera.focus_score) / 2.0, 3),
            "selected_light_id": chosen.light_id if chosen else None,
            "selected_light_state": chosen.state.value if chosen else TrafficLightState.UNKNOWN.value,
            "needs_user_selection": needs_user_selection,
            "detected_signs": ctx.signs,
            "pedestrians_detected": ctx.pedestrians_detected,
            "alert": alert.key,
            "severity": alert.severity.value,
            "message": Localization.t(self.lang, alert.key),
            "map_overlay": {
                "color": "red" if alert.key in {"overspeed_red", "red_crossed"} else "green",
                "badge": alert.key,
                "show_camera_preview": True,
            },
            "carplay_overlay": {
                "headline": Localization.t(self.lang, alert.key),
                "icon": alert.severity.value,
            },
            "android_auto_overlay": {
                "headline": Localization.t(self.lang, alert.key),
                "icon": alert.severity.value,
            },
        }
        self.agent.audit_event("decision", decision)
        return decision


def parse_frame(line: str) -> FrameContext:
    obj = json.loads(line)
    candidates = [
        TrafficLightCandidate(
            light_id=str(c["light_id"]),
            state=TrafficLightState(c.get("state", "unknown")),
            lane_ids=tuple(c.get("lane_ids", [])),
            confidence=float(c.get("confidence", 0.0)),
        )
        for c in obj.get("candidates", [])
    ]
    vehicle_obj = obj.get("vehicle", {})
    camera_obj = obj.get("camera", {})
    return FrameContext(
        route_id=str(obj.get("route_id", "default")),
        timestamp_ms=int(obj.get("timestamp_ms", int(time.time() * 1000))),
        candidates=candidates,
        vehicle=VehicleState(
            speed_kph=float(vehicle_obj.get("speed_kph", 0.0)),
            lane_id=str(vehicle_obj.get("lane_id", "unknown")),
            crossed_stop_line=bool(vehicle_obj.get("crossed_stop_line", False)),
            stationary_seconds=float(vehicle_obj.get("stationary_seconds", 0.0)),
        ),
        camera=CameraMeta(
            angle_score=float(camera_obj.get("angle_score", 0.5)),
            focus_score=float(camera_obj.get("focus_score", 0.5)),
            fps=int(camera_obj.get("fps", 30)),
        ),
        signs=[str(x) for x in obj.get("signs", [])],
        pedestrians_detected=bool(obj.get("pedestrians_detected", False)),
    )


BDD_FEATURE = """
Feature: Lane-aware traffic safety assistant with privacy-aware AI memory
  Scenario: Overspeed while red on current lane
    Given vehicle lane is L1 and speed is 45 kph
    And lane light on L1 is RED
    When frame is processed
    Then an AUDIO overspeed_red warning is emitted

  Scenario: Driver crosses red light
    Given vehicle crossed stop line on RED
    When frame is processed
    Then a SIREN red_crossed warning is emitted

  Scenario: Green light but no movement
    Given lane light is GREEN and stationary_seconds is 5
    When frame is processed
    Then a VISUAL green_wait warning is emitted

  Scenario: Lane cannot be resolved
    Given no light candidate matches vehicle lane
    When frame is processed
    Then user must select correct light
""".strip()


def run_self_test() -> int:
    temp_db = Path("./.traffic_ai_self_test.sqlite3")
    temp_db.unlink(missing_ok=True)

    agent = LearningAgent(temp_db, privacy_mode=PrivacyMode.BALANCED)
    engine = DecisionEngine(agent, lang="en")

    scenarios = [
        {
            "name": "overspeed_red",
            "frame": {
                "route_id": "R1",
                "vehicle": {"speed_kph": 45, "lane_id": "L1", "crossed_stop_line": False, "stationary_seconds": 0},
                "camera": {"angle_score": 0.9, "focus_score": 0.9, "fps": 30},
                "candidates": [{"light_id": "A", "state": "red", "lane_ids": ["L1"], "confidence": 0.98}],
                "signs": ["speed_limit_50"],
                "pedestrians_detected": False,
            },
            "expect": "overspeed_red",
        },
        {
            "name": "red_crossed",
            "frame": {
                "route_id": "R1",
                "vehicle": {"speed_kph": 20, "lane_id": "L1", "crossed_stop_line": True, "stationary_seconds": 0},
                "camera": {"angle_score": 0.9, "focus_score": 0.9, "fps": 30},
                "candidates": [{"light_id": "A", "state": "red", "lane_ids": ["L1"], "confidence": 0.96}],
                "signs": [],
                "pedestrians_detected": False,
            },
            "expect": "red_crossed",
        },
        {
            "name": "green_wait",
            "frame": {
                "route_id": "R1",
                "vehicle": {"speed_kph": 0, "lane_id": "L1", "crossed_stop_line": False, "stationary_seconds": 5.2},
                "camera": {"angle_score": 0.8, "focus_score": 0.8, "fps": 30},
                "candidates": [{"light_id": "A", "state": "green", "lane_ids": ["L1"], "confidence": 0.95}],
                "signs": [],
                "pedestrians_detected": False,
            },
            "expect": "green_wait",
        },
        {
            "name": "lane_unresolved",
            "frame": {
                "route_id": "R2",
                "vehicle": {"speed_kph": 10, "lane_id": "L9", "crossed_stop_line": False, "stationary_seconds": 0},
                "camera": {"angle_score": 0.7, "focus_score": 0.7, "fps": 30},
                "candidates": [{"light_id": "B", "state": "red", "lane_ids": ["L2"], "confidence": 0.82}],
                "signs": [],
                "pedestrians_detected": False,
            },
            "expect": "select_light",
        },
    ]

    ok = True
    for s in scenarios:
        decision = engine.process_frame(parse_frame(json.dumps(s["frame"])))
        if decision["alert"] != s["expect"]:
            ok = False
            logging.error("Scenario failed: %s expected=%s got=%s", s["name"], s["expect"], decision["alert"])

    print("BDD Feature:\n" + BDD_FEATURE)
    print("Self-test status:", "PASS" if ok else "FAIL")
    temp_db.unlink(missing_ok=True)
    return 0 if ok else 1


class AppState:
    def __init__(self, db_path: Path, lang: str, privacy_mode: PrivacyMode):
        self.agent = LearningAgent(db_path=db_path, privacy_mode=privacy_mode)
        self.engine = DecisionEngine(self.agent, lang=lang)


class TrafficAPIHandler(BaseHTTPRequestHandler):
    state: AppState = None  # type: ignore[assignment]

    def _read_json(self) -> Dict[str, object]:
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size) if size > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._write(
                HTTPStatus.OK,
                {
                    "service": "traffic-ai-assist",
                    "version": SEMVER,
                    "algorithm": [
                        "1) Parse frame context from input payload",
                        "2) Resolve lane-specific traffic light (or memory suggestion)",
                        "3) Evaluate safety rules (pedestrian, red crossing, overspeed, green wait)",
                        "4) Return localized alert + overlays and audit event",
                    ],
                    "endpoints": {
                        "GET": ["/", "/health", "/bdd"],
                        "POST": ["/ingest/frame", "/feedback/select-light", "/agent/chat", "/logs/parse"],
                    },
                    "examples": {
                        "ingest_frame": {
                            "method": "POST",
                            "path": "/ingest/frame",
                            "body": {
                                "route_id": "R100",
                                "timestamp_ms": 1730000000000,
                                "candidates": [
                                    {
                                        "light_id": "L-A",
                                        "state": "red",
                                        "lane_ids": ["lane-1"],
                                        "confidence": 0.91,
                                    }
                                ],
                                "vehicle": {
                                    "speed_kph": 50,
                                    "lane_id": "lane-1",
                                    "crossed_stop_line": False,
                                    "stationary_seconds": 0,
                                },
                                "camera": {"angle_score": 0.9, "focus_score": 0.9, "fps": 30},
                                "signs": ["speed_limit_50"],
                                "pedestrians_detected": False,
                            },
                        },
                        "agent_chat": {
                            "method": "POST",
                            "path": "/agent/chat",
                            "body": {"text": "status", "lang": "en"},
                        },
                    },
                },
            )
            return
        if self.path == "/health":
            self._write(HTTPStatus.OK, {"status": "ok", "version": SEMVER})
            return
        if self.path == "/bdd":
            self._write(HTTPStatus.OK, {"feature": BDD_FEATURE})
            return
        self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/ingest/frame":
            body = self._read_json()
            ctx = parse_frame(json.dumps(body, ensure_ascii=False))
            decision = self.state.engine.process_frame(ctx)
            self._write(HTTPStatus.OK, decision)
            return

        if self.path == "/feedback/select-light":
            body = self._read_json()
            route_id = str(body.get("route_id", "default"))
            lane_id = str(body.get("lane_id", "unknown"))
            light_id = str(body.get("light_id", ""))
            if not light_id:
                self._write(HTTPStatus.BAD_REQUEST, {"error": "light_id required"})
                return
            self.state.agent.remember_lane_light(route_id, lane_id, light_id)
            self.state.agent.audit_event("user_selection", body)
            self._write(HTTPStatus.OK, {"status": "saved"})
            return

        if self.path == "/agent/chat":
            body = self._read_json()
            text = str(body.get("text", ""))
            lang = str(body.get("lang", "en"))
            response = self.state.agent.chat(text, lang=lang)
            self._write(HTTPStatus.OK, {"response": response})
            return

        if self.path == "/logs/parse":
            body = self._read_json()
            line = str(body.get("line", ""))
            parsed = LogIngestor.parse_line(line)
            self._write(HTTPStatus.OK, {"parsed": parsed})
            return

        self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:  # keep lightweight log format
        logging.info("api %s - %s", self.address_string(), fmt % args)


def run_server(host: str, port: int, state: AppState) -> None:
    TrafficAPIHandler.state = state
    server = ThreadingHTTPServer((host, port), TrafficAPIHandler)
    logging.info("Traffic AI Assist service started on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def stream_frames(frames: Iterable[str], state: AppState) -> int:
    for raw in frames:
        raw = raw.strip()
        if not raw:
            continue
        ctx = parse_frame(raw)
        decision = state.engine.process_frame(ctx)
        print(json.dumps(decision, ensure_ascii=False))
    return 0


def configure_logging(debug: bool, log_file: Optional[Path]) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Traffic AI Assist integrated runnable program")
    parser.add_argument("--db", type=Path, default=Path("./traffic_ai.sqlite3"), help="SQLite memory db path")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "de", "fa"], help="Output language")
    parser.add_argument("--privacy", type=str, default="strict", choices=["strict", "balanced"])
    parser.add_argument("--input", type=Path, help="JSONL input frame file")
    parser.add_argument("--self-test", action="store_true", help="Run embedded BDD tests")
    parser.add_argument("--parse-log-line", type=str, help="Parse one docker/systemd log line")
    parser.add_argument("--serve", action="store_true", help="Run HTTP API service")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", type=Path, help="Optional service log file path")
    args = parser.parse_args()

    configure_logging(args.debug, args.log_file)

    if args.self_test:
        return run_self_test()

    if args.parse_log_line is not None:
        print(json.dumps(LogIngestor.parse_line(args.parse_log_line), ensure_ascii=False))
        return 0

    state = AppState(args.db, args.lang, PrivacyMode(args.privacy))

    if args.serve:
        run_server(args.host, args.port, state)
        return 0

    if args.input:
        with args.input.open("r", encoding="utf-8") as f:
            return stream_frames(f, state)

    parser.error("Provide one mode: --self-test or --parse-log-line or --serve or --input <jsonl>.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
