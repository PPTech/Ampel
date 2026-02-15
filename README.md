# Ampel - Traffic AI Assist

**Version:** 0.4.0  
**Code generation:** CODEX and CODEX CLI  
**Owner / Idea / Management:** Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Overview
This project provides a runnable traffic-safety AI agent core for lane-aware signal handling, alerts, learning memory, external dataset cataloging, and NVIDIA Triton inference integration hooks.

## What is new in v0.4.0
- Added **free sample data pipeline** with license-safe synthetic frames (`free_demo_samples`) mapped to open dataset taxonomies.
- Added **demo database table** `demo_sample_frames` and seeding workflow.
- Added **demo mode** (`--demo-mode`) that seeds DB + runs end-to-end inference/alert output immediately.
- Added `--export-demo-sample` to generate `data/demo_frames.jsonl` for offline testing.
- Expanded dataset catalog DB schema with `usage_reason` for traceability.

## Free sample datasets used and why
> Important: no raw external images are redistributed here. We use synthetic JSON frame samples derived from public dataset **taxonomies** for safe demo/testing.

1. **BDD100K** (`https://bdd-data.berkeley.edu/`)  
   License: Berkeley DeepDrive dataset terms.  
   Why used: route/lane/drivable-area style scenario taxonomy for urban demo.

2. **Bosch Small Traffic Lights** (`https://hci.iwr.uni-heidelberg.de/node/6132`)  
   License: Bosch dataset terms.  
   Why used: traffic-light class and confidence-oriented demo cases.

3. **LISA Traffic Light Dataset** (`https://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html`)  
   License: Academic usage terms.  
   Why used: signal state transition example patterns (red/green behavior).

4. **Mapillary Traffic Sign Dataset** (`https://www.mapillary.com/dataset/vistas`)  
   License: Mapillary Vistas terms.  
   Why used: road-sign labels in `extra.road_signs` for context.

## AI architecture
- `LearningAgent`: lane/light memory and privacy-aware event logging.
- `LaneAwareResolver`: lane-specific signal resolution with memory fallback.
- `AlertEngine`: overspeed-red, red-crossing, green-wait, pedestrian alerts.
- `NvidiaTritonClient`: optional embedded inference via Triton HTTP API.

## Database tables
- `lane_light_memory`
- `audit_log`
- `external_dataset_catalog`
- `demo_sample_frames`

## Commands
### Version and self-test
```bash
python3 traffic_ai_assist.py --version
python3 traffic_ai_assist.py --self-test
```

### Dataset catalog and metadata
```bash
python3 traffic_ai_assist.py --dataset-manifest
python3 traffic_ai_assist.py --sync-datasets --db traffic_ai.sqlite3
python3 traffic_ai_assist.py --fetch-dataset-metadata dataset_meta.json
```

### Demo mode (auto seed + run)
```bash
python3 traffic_ai_assist.py --demo-mode --db traffic_ai.sqlite3 --lang en
```

### Export free sample JSONL
```bash
python3 traffic_ai_assist.py --export-demo-sample data/demo_frames.jsonl
```

### Stream mode from JSONL
```bash
python3 traffic_ai_assist.py --input data/demo_frames.jsonl --db traffic_ai.sqlite3 --inference-backend input --lang en
```

### Triton backend
```bash
python3 traffic_ai_assist.py --input data/demo_frames.jsonl --inference-backend nvidia_triton --nvidia-endpoint http://127.0.0.1:8000 --nvidia-model traffic_light_detector
```

## BDD / Gherkin
Feature file:
- `features/traffic_ai_agent.feature`

## GitHub workflow
Environment limitations may block direct push. On your machine:
```bash
git checkout main
git pull --rebase
git merge <working-branch>
git push origin main
```
