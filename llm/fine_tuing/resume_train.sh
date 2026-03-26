#!/bin/bash
# 从 checkpoint-400 恢复训练
# 用法：cd /home/xjy/llm_tuning && bash resume_train.sh

set -e
cd "$(dirname "$0")"

source "$HOME/.local/bin/env"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

swift sft \
    --model Qwen/Qwen3.5-9B \
    --train_type lora \
    --dataset train_split.jsonl \
    --val_dataset val.jsonl \
    --output_dir output/qwen3.5-9b-graspdet \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --max_length 4096 \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_impl sdpa \
    --dataloader_num_workers 4 \
    --resume_from_checkpoint output/qwen3.5-9b-graspdet/v1-20260318-114022/checkpoint-400 \
    2>&1 | tee output/resume_train.log
