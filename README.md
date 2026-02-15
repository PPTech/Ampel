# Ampel - Traffic AI Assist

## Overview
Ampel provides a lane-aware traffic safety assistant with:
- traffic light selection for the active lane,
- safety alert evaluation (audio/visual/siren),
- local memory with SQLite,
- HTTP API endpoints for integration,
- CLI stream-processing mode.

Current service branch version: `0.2.0`.

## Main Files
- `traffic_ai_assist.py`: Core application (API + decision engine + CLI modes).
- `traffic_ai_runner.sh`: Simple wrapper to compile/run/serve/self-test.
- `sync_latest_to_main.sh`: Automation script to merge latest remote branch into `main`, push `main`, and run repository instruction steps.

## Quick Start
Run self-test:
```bash
./traffic_ai_runner.sh self-test
```

Compile check:
```bash
./traffic_ai_runner.sh compile
```

Run API server:
```bash
./traffic_ai_runner.sh serve
```

If port `8080` is already used:
```bash
PORT=8081 ./traffic_ai_runner.sh serve
```

## API Endpoints
GET:
- `/`
- `/health`
- `/bdd`

POST:
- `/ingest/frame`
- `/feedback/select-light`
- `/agent/chat`
- `/logs/parse`

## `traffic_ai_runner.sh` Modes
- `compile`: Python compile check for `traffic_ai_assist.py`.
- `run [input_jsonl]`: Process JSONL frames in CLI mode.
- `serve`: Start API server (`--serve` mode when supported).
- `self-test`: Run built-in self-test scenarios.
- `status`: Show whether configured port is listening.

Environment overrides:
- `PYTHON_BIN` (default `python3`)
- `HOST` (default `127.0.0.1`)
- `PORT` (default `8080`)
- `DB_PATH` (default `./var/data/traffic_ai.sqlite3`)
- `LOG_FILE` (default `./var/log/ampel.log`)
- `LANG_CODE` (default `en`)

## `sync_latest_to_main.sh`
This script automates update + publish flow:
1. Fetches latest refs from `origin`.
2. Detects newest `origin/*` branch (excluding `origin/main`, `origin/HEAD`).
3. Switches to `main` and updates it.
4. Merges newest branch into `main`.
5. Pushes `main` to GitHub.
6. Runs instruction steps (`compile`, `self-test`, gherkin export, dataset sync/metadata when supported).
7. Returns to your original branch.

Run:
```bash
./sync_latest_to_main.sh
```

## Notes
- Runtime/generated files such as `__pycache__/`, `var/`, `*.sqlite3`, and generated JSON files are local artifacts.
- For local service mode managed by systemd user unit:
```bash
systemctl --user status ampel.service
systemctl --user restart ampel.service
journalctl --user -u ampel.service -f
```
