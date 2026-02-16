#!/usr/bin/env python3
"""
Version: 0.9.9
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
Author: Dr. Babak Sorkhpour with support from ChatGPT
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ampel_app.cli import main
from shared.interfaces.hal import (
    IAlertSink,
    ICameraProvider,
    IInferenceEngine,
    ILocationProvider,
    ISecureStorage,
    Detection,
)
from src.core.logic.rules_engine import AlertType, TrafficRulesEngine
from src.core.vision.detector import TFLiteTrafficDetector


class ProtoCameraProvider(ICameraProvider):
    def __init__(self) -> None:
        self.started = False

    def start_stream(self) -> None:
        self.started = True

    def get_frame(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "proto-camera",
            "started": self.started,
        }


class ProtoLocationProvider(ILocationProvider):
    def get_anonymized_location(self) -> str:
        token = os.environ.get("AMPEL_ANON_LOCATION", "loc:proto:berlin")
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8")[:48]


class ProtoInferenceEngine(IInferenceEngine):
    def __init__(self) -> None:
        model_path = os.environ.get("AMPEL_TFLITE_MODEL", "")
        self.detector = TFLiteTrafficDetector(model_path=model_path or None)

    def detect(self, frame: Any) -> list[Detection]:
        return self.detector.detect(frame)


class ProtoAlertSink(IAlertSink):
    def play_audio(self, alert_type: str) -> None:
        _ = alert_type

    def show_visual(self, alert_type: str) -> None:
        _ = alert_type


class ProtoSecureStorage(ISecureStorage):
    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def save_encrypted(self, key: str, value: str) -> None:
        blob = json.loads(self.path.read_text(encoding="utf-8"))
        # TODO: REMOVE SECURITY RISK - replace base64 placeholder with strong encryption (TEE/Keystore).
        blob[key] = base64.b64encode(value.encode("utf-8")).decode("utf-8")
        self.path.write_text(json.dumps(blob), encoding="utf-8")

    def load_encrypted(self, key: str) -> Optional[str]:
        blob = json.loads(self.path.read_text(encoding="utf-8"))
        raw = blob.get(key)
        if raw is None:
            return None
        # TODO: REMOVE SECURITY RISK - replace base64 placeholder with strong encryption (TEE/Keystore).
        return base64.b64decode(str(raw).encode("utf-8")).decode("utf-8")


def build_hal_runtime() -> tuple[
    ICameraProvider,
    ILocationProvider,
    IInferenceEngine,
    IAlertSink,
    ISecureStorage,
]:
    camera = ProtoCameraProvider()
    location = ProtoLocationProvider()
    inference = ProtoInferenceEngine()
    alerts = ProtoAlertSink()
    storage = ProtoSecureStorage(ROOT / ".proto_secure_store.json")
    return camera, location, inference, alerts, storage


def hal_smoke_check() -> int:
    camera, location, inference, alerts, storage = build_hal_runtime()
    camera.start_stream()
    frame = camera.get_frame()
    detections = inference.detect(frame)
    anon = location.get_anonymized_location()
    storage.save_encrypted("last_location", anon)
    _ = storage.load_encrypted("last_location")
    engine = TrafficRulesEngine()
    events = inference.detector.detect_events(frame, speed_kph=42.0, distance_to_stopline_m=12.0)
    event = events[0] if events else None
    rule_alert = engine.evaluate(event, history=[]) if event else AlertType.NONE
    if rule_alert == AlertType.CRITICAL_ALERT:
        alerts.play_audio("CRITICAL_STOP")
        alerts.show_visual("CRITICAL_STOP")
    elif rule_alert == AlertType.WARN_ALERT:
        alerts.show_visual("WARN")
    print(
        json.dumps(
            {
                "hal": "ok",
                "timestamp": frame["timestamp"],
                "anonymized_location": anon,
                "detections": [d.__dict__ for d in detections],
                "rule_engine_alert": rule_alert.value,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    if "--hal-smoke" in sys.argv:
        raise SystemExit(hal_smoke_check())
    raise SystemExit(main())
