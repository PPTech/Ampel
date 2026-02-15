#!/usr/bin/env bash
set -euo pipefail

APP="traffic_ai_assist.py"
DB="traffic_ai.sqlite3"

usage() {
  cat <<USAGE
Usage:
  ./traffic_ai_runner.sh demo
  ./traffic_ai_runner.sh serve [port]
  ./traffic_ai_runner.sh sync
  ./traffic_ai_runner.sh triton [jsonl]
USAGE
}

cmd="${1:-}"
case "$cmd" in
  demo)
    python3 "$APP" --demo-mode --db "$DB" --lang en
    ;;
  serve)
    port="${2:-8080}"
    python3 "$APP" --serve --host 0.0.0.0 --port "$port" --db "$DB"
    ;;
  sync)
    python3 "$APP" --sync-datasets --db "$DB"
    ;;
  triton)
    input="${2:-data/demo_frames.jsonl}"
    python3 "$APP" --input "$input" --db "$DB" --inference-backend nvidia_triton --nvidia-endpoint http://127.0.0.1:8000 --nvidia-model traffic_light_detector
    ;;
  *)
    usage
    exit 1
    ;;
esac
