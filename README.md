# Ampel - Traffic AI Assist (Real Agent Core)

## Overview
This project provides a runnable core for a traffic safety assistant that can:
- detect/consume traffic light candidates,
- resolve lane-specific relevant signals,
- issue visual/audio/siren alerts,
- use local learning memory for repeated routes,
- integrate with NVIDIA Triton inference endpoints,
- manage external base AI dataset catalog metadata,
- follow BDD/Gherkin standards.

Current version: `0.3.0`.

## AI Agent and AI System
The runtime agent is implemented by:
- `LearningAgent` (local memory + interaction log)
- `LaneAwareResolver` (lane-aware signal selection)
- `AlertEngine` (traffic safety policy)
- `NvidiaTritonClient` (external inference backend integration)

Inference backends:
- `input` (candidates are provided in JSONL input)
- `nvidia_triton` (candidates can be fetched from Triton HTTP endpoint)

## External Base Datasets Included (Catalog)
The app includes a built-in catalog of external base datasets:
- BDD100K
- Bosch Small Traffic Lights
- LISA Traffic Light Dataset
- Mapillary Traffic Sign Dataset

> Note: this project syncs dataset metadata/catalog only. Full dataset download/training pipelines should be run in dedicated training infrastructure.

## Privacy and Security
- SQL operations use parameterized queries.
- Privacy mode:
  - `strict`: stores minimal audit payload
  - `balanced`: stores full event payload
- No shell command execution from untrusted input.

## Run

### 1) Self test
```bash
python3 traffic_ai_assist.py --self-test
```

### 2) Export Gherkin
```bash
python3 traffic_ai_assist.py --export-gherkin features/traffic_ai_agent.feature
```

### 3) Show dataset catalog
```bash
python3 traffic_ai_assist.py --dataset-manifest
```

### 4) Sync dataset catalog into DB
```bash
python3 traffic_ai_assist.py --sync-datasets --db traffic_ai.sqlite3
```

### 5) (Optional) Fetch remote metadata snippets
```bash
python3 traffic_ai_assist.py --fetch-dataset-metadata dataset_meta.json
```

### 6) Runtime with input backend
```bash
python3 traffic_ai_assist.py --input sample_frames.jsonl --inference-backend input --lang en
```

### 7) Runtime with NVIDIA Triton backend
```bash
python3 traffic_ai_assist.py --input sample_frames.jsonl --inference-backend nvidia_triton --nvidia-endpoint http://127.0.0.1:8000 --nvidia-model traffic_light_detector
```

## JSONL Input Format
```json
{
  "route_id": "R100",
  "timestamp_ms": 1730000000000,
  "candidates": [
    {"light_id":"L-A","state":"red","lane_ids":["lane-1"],"confidence":0.91}
  ],
  "vehicle": {
    "speed_kph": 50,
    "lane_id": "lane-1",
    "crossed_stop_line": false,
    "stationary_seconds": 0
  },
  "extra": {
    "pedestrian_detected": false,
    "road_signs": ["speed_50"]
  }
}
```

## BDD / Gherkin
Feature file:
- `features/traffic_ai_agent.feature`

Embedded feature text and self-test are also included in `traffic_ai_assist.py`.

## GitHub Deployment
```bash
git remote add origin https://github.com/PPTech/Ampel.git
git push -u origin <branch-name>
```

If branch protection or auth is enabled, configure a GitHub token/SSH key first.
