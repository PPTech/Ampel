# 🚦 Ampel - Traffic AI Assist

> **Version:** `0.9.2`  
> **License: MIT**  
> **Code generated with support from CODEX and CODEX CLI.**  
> **Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Demo-Visual%20UX-green)
![BDD](https://img.shields.io/badge/BDD-Gherkin-purple)

---

## ✨ What is implemented now

### Visual App UX
- Visual menu page: `GET /menu`
- Visual settings page with action buttons: `GET /settings`
- Visual datasets page with legal/source metadata cards: `GET /datasets`
- Visual dashboard: `GET /dashboard`
  - Google map iframe
  - Traffic-lamp overlay at map position
  - Dataset selector for random demo source
  - Camera preview with **manual Start/Stop**
  - Agent output panel
- Developer mode: `GET /developer`
- Expanded health endpoint with uptime/dataset/model status: `GET /health`
  - Live camera
  - Manual Start/Stop
  - Real-time client-side object guesses

### AI Agent + Learning
- Core free model in app developer flow: **COCO-SSD (TensorFlow.js, Apache-2.0)**.
- Core edge detector target for app engine: **EfficientDet-Lite0 (TensorFlow Lite, Apache-2.0)**.
- AGPL-risk model dependency removed from core path.
- Lane-aware resolver + memory-based fallback
- Rule engine: red overspeed, red crossing siren, green wait, pedestrian warnings
- Optional Triton backend fallback support
- `--train-agent` learning routine stores learned threshold profile in DB
- `--ab-test` variant comparison before release

### Dataset orchestration
- External manifest with requested sources and legal fields
- Compile + deduplicate manifest
- Sync dataset legal metadata to DB
- Online health checks for source reachability
- Training-plan generation and manual dataset registration

---

## 🧰 Install / Setup

### Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

## ▶️ Run commands (user-friendly)

| Command | What it does | When to use |
|---|---|---|
| `./traffic_ai_runner.sh serve 8080` | Starts visual dashboard/API on port 8080. | Daily local usage and UI checks. |
| `./traffic_ai_runner.sh demo` | Runs demo alert pipeline on sample frames. | Quick functional smoke test. |
| `./traffic_ai_runner.sh datasync` | Syncs dataset manifest metadata to SQLite. | After updating dataset manifests. |
| `./traffic_ai_runner.sh compilemanifest` | Compiles and deduplicates dataset manifest. | Before training-plan generation. |
| `./traffic_ai_runner.sh syncmain` | Merges branch to main with post-check scripts. | Release prep / branch sync. |
| `./traffic_ai_runner.sh abtest` | Runs A/B behavior comparison report. | Before publishing a version. |
| `./traffic_ai_runner.sh security` | Runs local security static checks. | Before PR/release. |

### Dataset manager commands

| Command | Purpose | Output |
|---|---|---|
| `python3 scripts/dataset_manager.py --compile-manifest` | Build normalized deduplicated dataset manifest. | `data/compiled_datasets_manifest.json` |
| `python3 scripts/dataset_manager.py --sync` | Store license/source metadata in DB. | Updated `external_dataset_catalog` |
| `python3 scripts/dataset_manager.py --online-health` | Check source URL reachability. | URL health report |
| `python3 scripts/dataset_manager.py --build-training-plan docs/TRAINING_PLAN.json` | Build training roadmap from manifest. | `docs/TRAINING_PLAN.json` |
| `python3 scripts/dataset_manager.py --register-manual ...` | Register private/manual dataset metadata. | `data/manual_extra_datasets.json` |
| `./scripts/import_all_datasets_local.sh` | Creates local metadata stubs for **all listed datasets**. | `data/local_datasets/*` |

### App commands

| Command | Purpose |
|---|---|
| `python3 traffic_ai_assist.py --serve --host 0.0.0.0 --port 8080` | Start web dashboard/API service. |
| `python3 traffic_ai_assist.py --demo-mode --db traffic_ai.sqlite3` | Execute end-to-end demo event generation. |
| `python3 traffic_ai_assist.py --train-agent --epochs 3 --db traffic_ai.sqlite3` | Train local adaptive threshold profile. |
| `python3 traffic_ai_assist.py --ab-test --db traffic_ai.sqlite3` | Run policy variant comparison. |
| `python3 traffic_ai_assist.py --security-check` | Run built-in security scan. |

---

## 📊 Dataset coverage and counts
- Source manifest: `data/external_datasets_manifest.json`
- Compiled manifest: `data/compiled_datasets_manifest.json`
- API stats: `GET /dataset-stats`

`/dataset-stats` returns:
- dataset_count
- known_sample_total (if sample_count values are provided)

`/health` now also includes:
- uptime_s
- dataset_catalog_count
- demo_frame_count
- core_ai_engine
- trained_red_speed_threshold_kph

---


## 🏗️ Mobile production architecture scaffold
- `docs/MOBILE_CLEAN_ARCHITECTURE.md` adds a Clean Architecture target tree for `ios/`, `android/`, and `ai_engine/`.
- New module placeholders: `ios/README.md`, `android/README.md`, `ai_engine/README.md`.
- New core BDD file: `features/traffic_light_core.feature` (strict Cucumber Gherkin).

## 📱 Architecture / mobile integration path
See: `GET /architecture`

Planned integration path:
1. Native Android/iOS camera pipeline
2. On-device inference (CoreML / TFLite / ONNX)
3. Projection adapter for CarPlay / AAOS
4. Edge + cloud hybrid learning loop

---


## 🚘 CarPlay + Edge CV additions
- New edge CV typed module: `ai_engine/inference/edge_detection.py`
  - model loader recommendation hooks for EfficientDet-Lite0 / MediaPipe Objectron (permissive)
  - `LaneContextFilter(detected_lights, gps_heading, imu)` pseudo-code style algorithm
  - `anonymizeFrame(frame)` OpenCV face/plate blur before inference/storage
- New iOS CarPlay snippets:
  - `ios/App/Presentation/CarPlay/CarPlaySceneDelegate.swift`
  - `ios/App/Presentation/CarPlay/CarPlayAlertManager.swift`
- `GET /architecture` now renders visual HTML (health endpoint remains JSON).
- Demo random endpoint now preserves route from selected dataset sample; `berlin-city-center` remains default for bundled demo samples only.

## 🧪 BDD + governance files

| File | Type | Purpose |
|---|---|---|
| `features/traffic_ai_agent.feature` | BDD | Main runtime safety scenarios |
| `features/traffic_light_core.feature` | BDD | Core signal/lane/privacy rules |
| `features/developer_mode.feature` | BDD | Developer camera behavior |
| `features/dataset_pipeline.feature` | BDD | Dataset compile/sync pipeline |
| `features/sync_workflow.feature` | BDD | Branch sync/health workflow |
| `features/edge_cv_carplay.feature` | BDD | Edge CV + CarPlay fallback scenarios |
| `features/federated_agent.feature` | BDD | On-device adaptive memory scenarios |
| `features/setup_test_suite.feature` | BDD | Setup/test automation behavior |
| `features/settings_menu.feature` | BDD | Settings buttons and operations UI |
| `features/secrets_privacy_guard.feature` | BDD | Secrets + logging + erasure safeguards |
| `CHANGELOG.md` | Governance | Release history |
| `MEMORY.md` | Governance | Persistent project memory |
| `docs/ALGORITHMS.md` | Governance | Algorithm inventory |
| `docs/SECURITY_BASELINE.md` | Governance | Security baseline references |
| `docs/DATASET_LEGAL_MATRIX.md` | Governance | License/legal matrix |
| `.github/workflows/*.yml` | Governance | CI quality gates |



## 🧠 Federated Driver Agent (On-device)
- Schema and local agent: `ai_engine/inference/adaptive_agent.py`
- Route memory table uses `intersection_hash` (not raw coordinates).
- Agent predicts time-to-green from local pattern and updates local weights on false-positive hard brake feedback.
- Privacy rule: **NO GPS history upload**; learning is strictly on-device.

## 🛠️ Setup + QA scripts
- Install script: `scripts/setup_project.sh`
- Unified test runner: `scripts/test_runner.py`
  - Executes BDD via `pytest`/`pytest-bdd`
  - Runs inference benchmark gate (`< 50ms`)
  - Runs privacy audit scan for risky logs

## 🖼️ Demo photo
```text
[AmpelAI Demo Preview]
Route: sample route from selected dataset (Berlin default in bundled samples)
Map Layer: Mapbox target (mobile)
Signal Overlay: RED / YELLOW / GREEN
Privacy: face/plate blur before inference
```


## 📦 Local dataset mirror (metadata stubs)
Because many upstream datasets are large/restricted by license terms, this repo stores local metadata stubs for all listed sources.

Run:
```bash
./scripts/import_all_datasets_local.sh
```

Generated structure:
- `data/local_datasets/INDEX.json`
- `data/local_datasets/<dataset-slug>/SOURCE.json`
- `data/local_datasets/<dataset-slug>/README.txt`


## ⚖️ License and attribution
- Project license file: `LICENSE` (MIT)
- Open-source usage and license mapping: `ATTRIBUTION.md`
- iOS permissive inference dependency path: `ios/Podfile` (`TensorFlowLiteSwift`)
- Python dependency list: `requirements.txt`


## 🔐 Secrets management (Android/iOS)
- Run: `./scripts/setup_mobile_secrets.sh`
- Android token source: `android/local.properties` (git-ignored) and `BuildConfig.MAPBOX_ACCESS_TOKEN`.
- iOS token source: `ios/Config.xcconfig` (git-ignored) -> `Info.plist` key `MAPBOX_ACCESS_TOKEN`.
- Never hardcode tokens in source files.
