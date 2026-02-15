# Ampel - Traffic AI Assist

**Version:** 0.4.1  
**Code generation:** CODEX and CODEX CLI  
**Owner / Idea / Management:** Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

## Fixes included for reported errors
- Fixed SQLite migration issue (`no column named usage_reason`) with automatic schema migration using `PRAGMA table_info` + `ALTER TABLE`.
- Added `traffic_ai_runner.sh` launcher so app usage is consistent (`demo`, `serve`, `sync`, `triton`).
- Added dashboard/API mode with menus and demo action (`--serve`, `GET /`, `GET /menu`, `POST /demo/run`).
- Added `.github/workflows/ci.yml` so `.github/workflows` exists on branch and CI checks run.

## Dashboard and menus
Run:
```bash
./traffic_ai_runner.sh serve 8080
```
Then open:
- `http://127.0.0.1:8080/` (dashboard)
- `http://127.0.0.1:8080/menu` (menu JSON)
- `http://127.0.0.1:8080/datasets` (dataset catalog)

## Free sample datasets used and why
The project uses **license-safe synthetic records** inspired by dataset taxonomy (no redistribution of raw dataset media):
- BDD100K — route/lane taxonomy examples
- Bosch Small Traffic Lights — traffic-light candidate structure examples
- LISA Traffic Light Dataset — signal transition scenario examples
- Mapillary Traffic Sign Dataset — road-sign context labels

Dataset metadata with license and usage reasons is stored in `external_dataset_catalog`.

## Demo mode
```bash
./traffic_ai_runner.sh demo
```
This will:
1. sync dataset catalog
2. seed demo samples into SQLite
3. run inference/alert output stream

## Commands
```bash
python3 traffic_ai_assist.py --sync-datasets --db traffic_ai.sqlite3
python3 traffic_ai_assist.py --fetch-dataset-metadata dataset_meta.json
python3 traffic_ai_assist.py --input data/demo_frames.jsonl --inference-backend nvidia_triton --nvidia-endpoint http://127.0.0.1:8000 --nvidia-model traffic_light_detector
```

If Triton is unavailable, the app logs warning and safely falls back to input candidates.
