# Ampel — Samsung-First Traffic Light Assistant (PR-0)

Version: 0.9.10  
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
