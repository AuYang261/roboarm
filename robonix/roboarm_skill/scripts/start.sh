#!/usr/bin/env bash
set -euo pipefail
echo "[roboarm_skill] starting..."

# Use roboarm venv python (has camera, arm, YOLO, ultralytics deps)
ROBOARM_PATH="${ROBOARM_PATH:-/home/xjy/roboarm}"
PYTHON="${ROBOARM_PATH}/.venv/bin/python3"

# Add robonix-api to PYTHONPATH so `from robonix_api import Skill` resolves
ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${PYTHONPATH:-}"

exec "$PYTHON" -m roboarm_grasp.node
