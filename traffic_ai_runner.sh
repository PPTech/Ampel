#!/usr/bin/env bash
# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
set -euo pipefail

APP="traffic_ai_assist.py"
DB="traffic_ai.sqlite3"

find_free_port() {
  local base_port="$1"
  python3 - "$base_port" <<'PY'
import socket, sys
start = int(sys.argv[1])
for p in range(start, start + 30):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", p))
        print(p)
        break
    except OSError:
        continue
    finally:
        s.close()
else:
    raise SystemExit("no free port in range")
PY
}

usage() {
  cat <<USAGE
Usage:
  ./traffic_ai_runner.sh demo
  ./traffic_ai_runner.sh serve [port]
  ./traffic_ai_runner.sh sync
  ./traffic_ai_runner.sh triton [jsonl]
  ./traffic_ai_runner.sh abtest
  ./traffic_ai_runner.sh security
  ./traffic_ai_runner.sh syncmain
  ./traffic_ai_runner.sh datasync
  ./traffic_ai_runner.sh compilemanifest
USAGE
}

cmd="${1:-}"
case "$cmd" in
  demo)
    python3 "$APP" --demo-mode --db "$DB" --lang en
    ;;
  serve)
    port="${2:-8080}"
    free_port="$(find_free_port "$port")"
    if [[ "$free_port" != "$port" ]]; then
      echo "Port $port busy; using $free_port" >&2
    fi
    python3 "$APP" --serve --host 0.0.0.0 --port "$free_port" --db "$DB"
    ;;
  sync)
    python3 "$APP" --sync-datasets --db "$DB"
    ;;
  triton)
    input="${2:-data/demo_frames.jsonl}"
    python3 "$APP" --input "$input" --db "$DB" --inference-backend nvidia_triton --nvidia-endpoint http://127.0.0.1:8000 --nvidia-model traffic_light_detector
    ;;
  abtest)
    python3 "$APP" --ab-test --db "$DB"
    ;;
  security)
    python3 "$APP" --security-check
    ;;
  syncmain)
    ./scripts/sync_latest_to_main.sh origin main
    ;;
  datasync)
    python3 scripts/dataset_manager.py --sync
    ;;
  compilemanifest)
    python3 scripts/dataset_manager.py --compile-manifest
    ;;
  *)
    usage
    exit 1
    ;;
esac
