# Ampel - Traffic AI Assist

**Version:** 0.7.0  
**License: MIT  
Code generated with support from CODEX and CODEX CLI.  
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)**

## UX/UI update delivered
- Menus are now **visual HTML pages/cards** (`/menu`, `/datasets`) instead of raw JSON.
- Dashboard demo is visualized on a map-canvas with traffic-lamp rendering.
- Demo selects random frame data from dataset pool and shows agent reaction.
- In-browser visual check reads pixel color from rendered lamp and reports recognized state.

## Current app flow
1. Seed/sync demo dataset records into SQLite.
2. Pick random sample frame (`/demo/random`).
3. Draw lamp over map canvas and recognize color from visual pixel.
4. Run lane-aware AI rules and show alert action.

## Dataset orchestration (requested links)
- `data/external_datasets_manifest.json` keeps legal metadata for all requested sources.
- `scripts/dataset_manager.py` compiles, deduplicates, syncs, health-checks, and builds training plan.
- `docs/DATASET_LEGAL_MATRIX.md` and `docs/TRAINING_PLAN.json` describe lawful usage and optimization pipeline.

## Important legal note
External datasets are not redistributed in this repo. Download/train only under upstream licenses/terms.

## Scripts
```bash
./traffic_ai_runner.sh serve 8080
./traffic_ai_runner.sh demo
./traffic_ai_runner.sh datasync
./traffic_ai_runner.sh compilemanifest
./traffic_ai_runner.sh syncmain
```

## Dataset manager commands
```bash
python3 scripts/dataset_manager.py --compile-manifest
python3 scripts/dataset_manager.py --sync
python3 scripts/dataset_manager.py --online-health
python3 scripts/dataset_manager.py --build-training-plan docs/TRAINING_PLAN.json
python3 scripts/dataset_manager.py --register-manual --name "extra" --scope "lights" --license "private" --url "file:///path" --usage "local tuning"
```

## Standards artifacts
- `CHANGELOG.md`
- `README.md`
- `.github/workflows/ci.yml`
- `MEMORY.md`
- `features/traffic_ai_agent.feature`
- `docs/ALGORITHMS.md`
- `docs/SECURITY_BASELINE.md`
- `docs/DATASET_LEGAL_MATRIX.md`
