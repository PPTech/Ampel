# Ampel — Samsung-First Traffic Light Assistant (PR-0)

Version: 0.9.18  
License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)  
Author: Dr. Babak Sorkhpour with support from ChatGPT

## Samsung-First Strategy

Ampel is being migrated from Python prototype scripts to a production-grade Samsung-first Android app.

### Platform Priorities
1. **Primary**: `mobile/android` (Android Studio / Kotlin / Samsung optimization path).
2. **Prototype**: `proto/python` (legacy Python behavior preserved during migration).
3. **Shared Contracts**: `shared/specs` (schemas and interop formats).
4. **Embedded Target**: `gadget/linux` (placeholder for Raspberry Pi / Jetson runtime).

### PR-0 Repository Layout
- `mobile/android/` → Android app home (new primary runtime).
- `proto/python/` → migrated Python prototype code and models.
- `shared/specs/` → JSON and interface specs.
- `gadget/linux/` → future embedded deployment.
- `scripts/` → automation, setup, CI helper scripts.

### Privacy-First Policy
- No cloud-upload logic for raw camera streams.
- Edge-only processing by default.
- Sensitive values must use local secure storage and redaction paths.

### Compatibility Notes
- Existing entry points remain callable via compatibility wrappers while code is moved.
- Follow-up PRs (PR-1..PR-6) will continue migration from prototype to Android-first modules.


## PR-1 Shared Contract & HAL

- `shared/specs/traffic_event.schema.json` defines portable event payload contract.
- `shared/interfaces/hal.py` defines platform-agnostic runtime interfaces.
- `proto/python/traffic_ai_assist.py --hal-smoke` validates prototype adherence to HAL interfaces.


## PR-2 Deterministic Rule Engine

- Feature file: `features/traffic_rules.feature`
- Engine: `src/core/logic/rules_engine.py`
- Unit tests: `tests/unit/test_rules_engine.py`
- BDD run command: `pytest features/` (or `pytest tests/bdd/test_traffic_rules_bdd.py`).


## PR-3 Detection Logic & Lane Association

- TFLite-first detector adapter: `src/core/vision/detector.py`
- Anti-flicker temporal smoothing: `src/core/vision/smoothing.py`
- Lane heuristic estimator: `src/core/vision/lane_estimator.py`
- New unit test for one-frame glitch rejection: `tests/unit/test_temporal_smoothing.py`


## PR-4 Android MVP Implementation

- Android app skeleton initialized under `mobile/android/app/` with CameraX, Jetpack Compose, Accompanist, and TFLite dependencies.
- Camera pipeline scaffold: `CameraManager.kt` (background analyzer + VGA thermal safety target).
- On-device inference scaffold: `TrafficLightDetector.kt` (NNAPI preferred, GPU fallback).
- Compose UI scaffold: `MainScreen.kt` with camera preview, overlay boxes, and top status banner.


## PR-5 Security Hardening & Privacy Compliance

- ProGuard/R8 hardening rules in `mobile/android/app/proguard-rules.pro`.
- Runtime tamper detection in `IntegrityManager.kt` (signature verification + Knox/TIMA capability check).
- Encrypted preference storage via `SecureStorage.kt` (`EncryptedSharedPreferences` + `MasterKey`).
- Privacy manager kill-switch in `PrivacyManager.kt` for GDPR erase and edge-only request enforcement.


## PR-6 Unified CI/CD & Test Automation

- Unified monorepo workflow: `.github/workflows/main_pipeline.yml`
- Predictive test selection: `scripts/pts_selector.py`
- Failure diagnostics artifacting: `scripts/diagnose_failure.py`
- One-command setup: `setup_dev_env.sh`
- Upload analyzer now accepts broad image formats (including webp/heic/avif) in demo API path.


## PR-6.1 Media Overlay Behavior Fixes

- Uploaded photos now replace live camera preview and display traffic sign/light bounding boxes with color reaction text.
- Uploaded videos now replace live camera preview and show timeline-based bounding boxes plus message overlays over playback.


## PHASE 7 Real-World Optimization & Polish

- Audio ducking alert manager for safer coexistence with navigation/music apps.
- Tunnel-mode location handling to avoid GPS-loss false stop-and-go alerts.
- Samsung chipset-aware TFLite delegate factory (Exynos GPU preference, Snapdragon NNAPI preference, CPU fallback).
- Dashboard now supports YouTube link analysis and real-time overlay messages/boxes over uploaded media modes.


## Phase 7.1 Repo Stabilization & CI Unification

- Single CI workflow now runs on `master` with Predictive Test Selection and hygiene gates.
- Threat model documented in `docs/THREAT_MODEL.md`.
- Runtime artifacts are blocked from git by `.gitignore` and CI checks.

## Phase 7.2 Dashboard UX Reliability

- Fixed camera pending UX with explicit permission/retry messaging and better fallback states.
- Improved media stage behavior so uploaded photo/video/YouTube content consistently replaces live camera and shows overlay/action reactions.
- Enhanced developer mode object detection using COCO-SSD temporal smoothing and clearer status telemetry.

## Phase 7.3 Overlay Accuracy + Feedback Learning

- Bounding-box rendering now aligns with contain-fit media viewport to reduce misplaced boxes on uploaded photos/videos.
- Overlay state is reset on source switches so stale detections do not leak into new camera/media sessions.
- Added local user-feedback learning controls (correct/wrong) to tune future prediction bias without cloud upload.


## Professional Runtime Status Matrix

| Capability | Previous behavior | Current behavior (0.9.18) | Notes |
|---|---|---|---|
| Photo overlay | Fixed placeholder confidence and near-static box positions | Adaptive confidence variance + contain-fit projection | Improves visual trust and object placement |
| Video/YouTube overlay | Timeline-only stub, repeated state patterns | Browser-assisted per-frame detection + state smoothing | Better live response without server GPU |
| Agent output panel | Mainly updated from random demo endpoint | Updated for media uploads and YouTube flow too | Better observability for users/testing |
| Map display | Could be overwritten by dataset demo coordinates | Keeps real user location map, shows demo GPS separately | Reduces confusion for field validation |
| Feedback learning | Basic endpoint only | Active UI controls + persisted local bias memory | Enables user-guided iterative improvement |

## Visual QA & Reporting Format

| Dashboard card | Required signal | Validation method |
|---|---|---|
| Live / Photo / Video Detection | Current state text + overlay boxes | Screenshot + UI interaction test |
| Google Map + Traffic Lamp | Real location display + lamp state | Geolocation permission + event update |
| Interactive Demo Controls | Analyze photo/video/youtube actions | API response + rendered overlay |
| Agent output | Structured JSON output for each action | Runtime panel + endpoint payload |
