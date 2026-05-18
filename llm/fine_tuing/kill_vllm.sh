#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/output/vllm.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Killed vLLM server (PID $PID)"
    else
        echo "Process $PID not running"
    fi
    rm -f "$PID_FILE"
else
    # 兜底：按进程名查找
    PIDS=$(pgrep -f "vllm serve" || true)
    if [ -n "$PIDS" ]; then
        echo "$PIDS" | xargs kill
        echo "Killed vLLM processes: $PIDS"
    else
        echo "No vLLM server found"
    fi
fi
