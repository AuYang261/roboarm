#!/bin/bash
# 将 LoRA adapter 合并进基础模型，输出完整模型权重
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

ADAPTER="output/qwen3.5-9b-graspdet/v1-20260318-114022/checkpoint-100"
OUT="output/merged-qwen3.5-9b-graspdet"

echo "Merging LoRA into base model..."
echo "Adapter : $ADAPTER"
echo "Output  : $OUT"

python merge_lora_peft.py

echo "Done: $OUT"
