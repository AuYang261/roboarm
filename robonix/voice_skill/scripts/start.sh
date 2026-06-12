#!/usr/bin/env bash
set -euo pipefail
echo "[voice_skill] starting..."

ROBOARM_PATH="${ROBOARM_PATH:-/home/xjy/roboarm}"
PYTHON="${ROBOARM_PATH}/.venv/bin/python3"

ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${PYTHONPATH:-}"

exec "$PYTHON" -m roboarm_voice.node
