"""
直接操作 safetensors 文件合并 LoRA，绕开 transformers 模型加载问题。
base model keys: model.language_model.layers.N.xxx.weight
LoRA keys:       base_model.model.model.language_model.layers.N.xxx.lora_A/B.weight
"""
import json, shutil
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR  = Path(__file__).parent
ADAPTER   = BASE_DIR / "output/qwen3.5-9b-graspdet/v1-20260318-114022/checkpoint-100"
MODEL_DIR = Path("/home/xjy/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B")
OUT_DIR   = BASE_DIR / "output/merged-qwen3.5-9b-graspdet"

# ── 读 adapter config ────────────────────────────────────────────────────────
cfg   = json.load(open(ADAPTER / "adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
print(f"LoRA rank={cfg['r']}, alpha={cfg['lora_alpha']}, scale={scale:.4f}")

# ── 读 LoRA 权重 ──────────────────────────────────────────────────────────────
lora_weights = {}
with safe_open(str(ADAPTER / "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        lora_weights[k] = f.get_tensor(k).to(torch.float32)

# 构建 module_path -> {lora_A, lora_B}
# key 格式: base_model.model.model.language_model.xxx.lora_A.weight
STRIP = "base_model.model."
modules = defaultdict(dict)
for k, v in lora_weights.items():
    if not k.startswith(STRIP):
        continue
    rest = k[len(STRIP):]           # model.language_model.xxx.lora_A.weight
    *parts, ab, _ = rest.split(".")  # ab = lora_A / lora_B
    mod_path = ".".join(parts)       # model.language_model.xxx
    modules[mod_path][ab] = v

print(f"LoRA modules: {len(modules)}")

# ── 预计算所有 delta ─────────────────────────────────────────────────────────
deltas = {}  # weight_key -> delta tensor (float32)
for mod_path, ab in modules.items():
    w_key = f"{mod_path}.weight"
    lora_A = ab["lora_A"]
    lora_B = ab["lora_B"]
    deltas[w_key] = (lora_B @ lora_A) * scale
print(f"Delta keys: {len(deltas)}")

# ── 复制并修改 safetensors 文件 ───────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 读 index
index_path = MODEL_DIR / "model.safetensors.index.json"
index = json.load(open(index_path))
weight_map = index["weight_map"]  # key -> shard filename

# 按 shard 分组
shard_keys = defaultdict(list)
for k, shard in weight_map.items():
    shard_keys[shard].append(k)

merged_count = 0
for shard_name, keys in shard_keys.items():
    src = MODEL_DIR / shard_name
    tensors = {}
    with safe_open(str(src), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).to(torch.float32)
            if k in deltas:
                t = t + deltas[k]
                merged_count += 1
            tensors[k] = t.to(torch.bfloat16)
    dst = OUT_DIR / shard_name
    save_file(tensors, str(dst))
    print(f"  Saved {shard_name}  ({len(tensors)} tensors, {merged_count} merged so far)")

# 复制非权重文件
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "generation_config.json",
              "preprocessor_config.json", "video_preprocessor_config.json",
              "configuration.json", "model.safetensors.index.json"]:
    src = MODEL_DIR / fname
    if src.exists():
        shutil.copy(src, OUT_DIR / fname)

print(f"\nDone. Merged {merged_count}/{len(deltas)} LoRA deltas → {OUT_DIR}")
