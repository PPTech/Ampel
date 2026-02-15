#!/usr/bin/env python3
"""
Version: 0.9.3
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None
try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    class _NP:
        ndarray = object
    np = _NP()

ModelType = Literal["efficientdet_lite0", "mediapipe_objectron"]


@dataclass(frozen=True)
class ModelConfig:
    model_type: ModelType
    weights_path: str
    input_size: Tuple[int, int]
    confidence_threshold: float = 0.35


@dataclass(frozen=True)
class LightDetection:
    light_id: str
    lane_hint: Optional[int]
    heading_deg: float
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class SensorSnapshot:
    gyro_yaw_rate: float
    accel_x: float
    accel_y: float
    accel_z: float


@dataclass(frozen=True)
class VehiclePoseEstimate:
    lateral_index: int
    confidence: float


class EdgeDetector:
    """Model loader abstraction for future CoreML/TFLite conversion."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.backend = self._load_backend(config)

    def _load_backend(self, config: ModelConfig):
        # Recommended loading paths (training/prototyping):
        # 1) EfficientDet-Lite0: TensorFlow Lite interpreter (Apache-2.0)
        # 2) MediaPipe Objectron: lightweight edge model family (Apache-2.0)
        # This placeholder keeps the API stable for mobile conversion.
        if config.model_type not in ("efficientdet_lite0", "mediapipe_objectron"):
            raise ValueError(f"Unsupported model_type: {config.model_type}")
        return {"model_type": config.model_type, "weights": config.weights_path}


def estimate_lane_from_imu(gps_heading: float, imu: SensorSnapshot) -> VehiclePoseEstimate:
    """Estimate lane index from heading stability + lateral acceleration.

    Heuristic:
    - Small yaw rate and low lateral accel => keep center lane (0)
    - Positive lateral accel with yaw drift => lane +1 (right shift in RHS countries)
    - Negative lateral accel with yaw drift => lane -1
    """
    yaw_weight = min(1.0, abs(imu.gyro_yaw_rate) / 0.8)
    lat_acc = imu.accel_x
    if lat_acc > 0.6:
        return VehiclePoseEstimate(lateral_index=1, confidence=0.55 + 0.35 * yaw_weight)
    if lat_acc < -0.6:
        return VehiclePoseEstimate(lateral_index=-1, confidence=0.55 + 0.35 * yaw_weight)
    _ = gps_heading
    return VehiclePoseEstimate(lateral_index=0, confidence=0.75)


def LaneContextFilter(
    detected_lights: Sequence[LightDetection],
    gps_heading: float,
    imu: SensorSnapshot,
) -> List[LightDetection]:
    """Pseudo-code style lane-aware light filtering.

    Algorithm (LaneContextFilter):
    1. Read `detected_lights` and current `gps_heading`.
    2. Read gyroscope + accelerometer snapshot.
    3. Estimate vehicle lateral lane index from IMU drift.
    4. For each light:
       - Reject low confidence (<0.35).
       - Compare light heading with car heading.
       - Keep light only if lane hint matches estimated lane (or unknown lane hint).
    5. Sort remaining by confidence descending.
    """
    pose = estimate_lane_from_imu(gps_heading=gps_heading, imu=imu)
    filtered: List[LightDetection] = []
    for light in detected_lights:
        if light.confidence < 0.35:
            continue
        heading_diff = abs((light.heading_deg - gps_heading + 180.0) % 360.0 - 180.0)
        if heading_diff > 55.0:
            continue
        if light.lane_hint is not None and abs(light.lane_hint - pose.lateral_index) > 1:
            continue
        filtered.append(light)
    filtered.sort(key=lambda d: d.confidence, reverse=True)
    return filtered


def _blur_regions(frame: np.ndarray, regions: Iterable[Tuple[int, int, int, int]], ksize: Tuple[int, int] = (31, 31)) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for x, y, rw, rh in regions:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, ksize, sigmaX=0, sigmaY=0)
    return out


def anonymizeFrame(frame: np.ndarray) -> np.ndarray:
    """Detect faces and license plates and blur them before inference/storage."""
    if frame.ndim != 3:
        raise ValueError("Expected color frame (H, W, C)")
    if cv2 is None:
        raise RuntimeError("OpenCV is required for anonymizeFrame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 12))

    out = _blur_regions(frame, faces)
    out = _blur_regions(out, plates, ksize=(41, 41))
    return out
