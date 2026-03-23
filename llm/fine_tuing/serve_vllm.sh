#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

LOG="output/vllm_serve.log"
PID_FILE="output/vllm.pid"
mkdir -p output

# MERGED="output/merged-qwen3.5-9b-graspdet"
MERGED="Qwen/Qwen3.5-9B"

echo "Starting vLLM server on port 8001 (merged model)..."

nohup vllm serve "$MERGED" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8001 \
    --host 0.0.0.0 \
    > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "PID: $(cat $PID_FILE)  Log: $LOG"
