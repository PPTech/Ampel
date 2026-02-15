# 🚦 Ampel - Traffic AI Assist

> **Version:** `0.9.0`  
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
- Visual datasets page with legal/source metadata cards: `GET /datasets`
- Visual dashboard: `GET /dashboard`
  - Google map iframe
  - Traffic-lamp overlay at map position
  - Dataset selector for random demo source
  - Camera preview with **manual Start/Stop**
  - Agent output panel
- Developer mode: `GET /developer`
  - Live camera
  - Manual Start/Stop
  - Real-time client-side object guesses

### AI Agent + Learning
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

### Optional tool install command you requested
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

> ⚠️ Always verify external install scripts and licenses before executing.

---

## ▶️ Run commands

```bash
./traffic_ai_runner.sh serve 8080
./traffic_ai_runner.sh demo
./traffic_ai_runner.sh datasync
./traffic_ai_runner.sh compilemanifest
./traffic_ai_runner.sh syncmain
./traffic_ai_runner.sh abtest
./traffic_ai_runner.sh security
```

### Dataset manager
```bash
python3 scripts/dataset_manager.py --compile-manifest
python3 scripts/dataset_manager.py --sync
python3 scripts/dataset_manager.py --online-health
python3 scripts/dataset_manager.py --build-training-plan docs/TRAINING_PLAN.json
python3 scripts/dataset_manager.py --register-manual --name "my-ds" --scope "traffic lights" --license "private" --url "file:///data/local" --usage "internal fine-tuning"
```

### App commands
```bash
python3 traffic_ai_assist.py --serve --host 0.0.0.0 --port 8080
python3 traffic_ai_assist.py --demo-mode --db traffic_ai.sqlite3
python3 traffic_ai_assist.py --train-agent --epochs 3 --db traffic_ai.sqlite3
python3 traffic_ai_assist.py --ab-test --db traffic_ai.sqlite3
python3 traffic_ai_assist.py --security-check
```

---

## 📊 Dataset coverage and counts
- Source manifest: `data/external_datasets_manifest.json`
- Compiled manifest: `data/compiled_datasets_manifest.json`
- API stats: `GET /dataset-stats`

`/dataset-stats` returns:
- dataset_count
- known_sample_total (if sample_count values are provided)

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
  - model loader recommendation hooks for YOLOv8-Nano / EfficientDet-Lite
  - `LaneContextFilter(detected_lights, gps_heading, imu)` pseudo-code style algorithm
  - `anonymizeFrame(frame)` OpenCV face/plate blur before inference/storage
- New iOS CarPlay snippets:
  - `ios/App/Presentation/CarPlay/CarPlaySceneDelegate.swift`
  - `ios/App/Presentation/CarPlay/CarPlayAlertManager.swift`
- `GET /architecture` now renders visual HTML (health endpoint remains JSON).
- Demo random endpoint keeps route fixed at `berlin-city-center` even when dataset source changes.

## 🧪 BDD + governance files
- `features/traffic_ai_agent.feature`
- `features/developer_mode.feature`
- `features/dataset_pipeline.feature`
- `features/sync_workflow.feature`
- `features/edge_cv_carplay.feature`
- `features/federated_agent.feature`
- `features/setup_test_suite.feature`
- `CHANGELOG.md`
- `MEMORY.md`
- `docs/ALGORITHMS.md`
- `docs/SECURITY_BASELINE.md`
- `docs/DATASET_LEGAL_MATRIX.md`
- `.github/workflows/*.yml`



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
Route: berlin-city-center
Map Layer: Mapbox target (mobile)
Signal Overlay: RED / YELLOW / GREEN
Privacy: face/plate blur before inference
```
