#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
APP_FILE="$ROOT_DIR/traffic_ai_assist.py"
HOST_VALUE="${HOST:-127.0.0.1}"
PORT_VALUE="${PORT:-8080}"

supports_arg() {
  local arg="$1"
  "$PYTHON_BIN" "$APP_FILE" --help 2>&1 | grep -q -- "$arg"
}

is_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {found=1} END {exit(found?0:1)}'
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {found=1} END {exit(found?0:1)}'
    return $?
  fi
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  ./traffic_ai_runner.sh compile
  ./traffic_ai_runner.sh run [input_jsonl]
  ./traffic_ai_runner.sh serve
  ./traffic_ai_runner.sh self-test
  ./traffic_ai_runner.sh status

Environment overrides:
  PYTHON_BIN  Python executable (default: python3)
  HOST        Service host (default: 127.0.0.1)
  PORT        Service port (default: 8080)
  DB_PATH     SQLite path (default: ./var/data/traffic_ai.sqlite3)
  LOG_FILE    Log file path (default: ./var/log/ampel.log)
  LANG_CODE   Output language for run mode (default: en)
EOF
}

if [[ ! -f "$APP_FILE" ]]; then
  echo "Error: $APP_FILE not found"
  exit 1
fi

MODE="${1:-serve}"
shift || true

case "$MODE" in
  compile)
    "$PYTHON_BIN" -m py_compile "$APP_FILE"
    echo "Compile check passed: $APP_FILE"
    ;;

  run)
    INPUT_FILE="${1:-$ROOT_DIR/sample_frames.jsonl}"
    if [[ ! -f "$INPUT_FILE" ]]; then
      echo "Error: input file not found: $INPUT_FILE"
      echo "Provide a JSONL input file: ./traffic_ai_runner.sh run /path/to/file.jsonl"
      exit 1
    fi
    if supports_arg "--inference-backend"; then
      exec "$PYTHON_BIN" "$APP_FILE" \
        --input "$INPUT_FILE" \
        --inference-backend input \
        --lang "${LANG_CODE:-en}"
    fi
    exec "$PYTHON_BIN" "$APP_FILE" \
      --input "$INPUT_FILE" \
      --lang "${LANG_CODE:-en}"
    ;;

  serve)
    if ! supports_arg "--serve"; then
      echo "This version of traffic_ai_assist.py does not support --serve."
      echo "Use run mode instead: ./traffic_ai_runner.sh run [input_jsonl]"
      exit 0
    fi

    if is_port_in_use "$PORT_VALUE"; then
      echo "Port $PORT_VALUE is already in use."
      if command -v curl >/dev/null 2>&1 && curl -fsS "http://$HOST_VALUE:$PORT_VALUE/health" >/dev/null 2>&1; then
        echo "A traffic_ai_assist service is already running on http://$HOST_VALUE:$PORT_VALUE"
        curl -sS "http://$HOST_VALUE:$PORT_VALUE/health" 2>/dev/null || true
        exit 0
      fi
      if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active --quiet ampel.service 2>/dev/null; then
        echo "ampel.service is already running on http://$HOST_VALUE:$PORT_VALUE"
      else
        echo "Another process is using port $PORT_VALUE."
      fi
      echo "Use another port, for example: PORT=8081 ./traffic_ai_runner.sh serve"
      exit 0
    fi

    mkdir -p "$ROOT_DIR/var/log" "$ROOT_DIR/var/data"
    exec "$PYTHON_BIN" "$APP_FILE" \
      --serve \
      --host "$HOST_VALUE" \
      --port "$PORT_VALUE" \
      --db "${DB_PATH:-$ROOT_DIR/var/data/traffic_ai.sqlite3}" \
      --log-file "${LOG_FILE:-$ROOT_DIR/var/log/ampel.log}"
    ;;

  self-test)
    exec "$PYTHON_BIN" "$APP_FILE" --self-test
    ;;

  status)
    if is_port_in_use "$PORT_VALUE"; then
      echo "Port $PORT_VALUE is in use."
      if command -v curl >/dev/null 2>&1; then
        curl -sS "http://$HOST_VALUE:$PORT_VALUE/health" || true
      fi
    else
      echo "Nothing is listening on $HOST_VALUE:$PORT_VALUE"
    fi
    ;;

  *)
    usage
    exit 2
    ;;
esac
