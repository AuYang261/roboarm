#!/usr/bin/env bash
# Start the Piper arm health primitive.
# Requires piper-sdk in the Python environment.
set -eo pipefail
PKG_ROOT="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG_ROOT"

# Source the roboarm venv for piper_sdk.
if [ -f "$HOME/roboarm/.venv/bin/activate" ]; then
  source "$HOME/roboarm/.venv/bin/activate"
fi

# Make robonix-api and codegen stubs importable.
export PYTHONPATH="$(rbnx path robonix-api):$PKG_ROOT:${PYTHONPATH:-}"

exec python3 -m health_piper.main
