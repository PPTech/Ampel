# Ampel - Traffic AI Assist

**Version:** 0.6.0  
**License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)**

## What this delivers now
- Runnable backend + dashboard app with camera block, map, menu, and demo trigger.
- Lane-aware traffic lamp decision + safety alerts.
- Dataset catalog sync with legal metadata and usage reasons.
- Offline/online dataset management script for ingest + health checks + training plan generation.
- Sync script for merging latest branch into `main` and running health checks.

## Why your serve error happened (`Address already in use`)
`8080` was already occupied by another process.
Fix now:
- `traffic_ai_runner.sh serve 8080` auto-picks a free port.
- API returns clear bind failure message with return code `98`.

## Where are traffic lamps, camera, AI action?
- **Traffic lamps:** alert engine + lane resolver produce lamp-specific events.
- **Camera:** dashboard requests browser camera via WebRTC (`getUserMedia`).
- **AI role:** lane-aware resolver + memory + optional Triton inference fallback.
- **Apple/Android:** current repo is backend/web prototype; `/architecture` endpoint documents native integration path (CarPlay/AAOS wrappers).

## Dataset integration (requested links)
Manifest: `data/external_datasets_manifest.json`
Includes legal/usage metadata for:
- dtld_parsing
- S2TLD repo
- Kaggle FasterRCNN notebooks/datasets
- LISA reference
- Mendeley dataset

> Important legal note: external datasets are **not redistributed** in this repository. Use upstream license terms before download/train/export.

## New scripts
### 1) Sync latest into main + health checks
```bash
./scripts/sync_latest_to_main.sh origin main
```
Runs:
- git fetch + merge workflow
- dataset sync
- A/B test
- security check

### 2) Dataset manager (offline/online + manual add)
```bash
python3 scripts/dataset_manager.py --sync
python3 scripts/dataset_manager.py --online-health
python3 scripts/dataset_manager.py --build-training-plan docs/TRAINING_PLAN.json
python3 scripts/dataset_manager.py --register-manual --name "My Local Dataset" --scope "traffic lights" --license "private" --url "file:///datasets/local" --usage "internal fine-tuning"
```

## Run
```bash
./traffic_ai_runner.sh sync
./traffic_ai_runner.sh demo
./traffic_ai_runner.sh serve 8080
./traffic_ai_runner.sh abtest
./traffic_ai_runner.sh security
```

## Standards files
- `CHANGELOG.md`
- `README.md`
- `.github/workflows/ci.yml`
- `MEMORY.md`
- `features/traffic_ai_agent.feature`
- `docs/ALGORITHMS.md`
- `docs/SECURITY_BASELINE.md`
