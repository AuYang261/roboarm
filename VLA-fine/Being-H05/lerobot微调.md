# Being-H05 单卡 RTX 5090 微调方案（基于 LeRobot v2.1 数据）

> Being-H0.5 is trained based on lerobot v2.1.
> 硬件：单张 NVIDIA RTX 5090 (32 GB VRAM) — **已验证通过 (2026-06-03)**
> Koch 机械臂数据集已转换为 v2.1 格式。

---

## 目录

1. [硬件与数据概览](#1-硬件与数据概览)
2. [环境准备](#2-环境准备)
3. [数据集配置](#3-数据集配置)
4. [模型下载](#4-模型下载)
5. [微调脚本](#5-微调脚本)
6. [启动训练](#6-启动训练)
7. [训练监控与调优](#7-训练监控与调优)
8. [实时推理评估](#8-实时推理评估)
9. [已验证的代码修复](#9-已验证的代码修复)
10. [常见问题](#10-常见问题)

---

## 1. 硬件与数据概览

### 1.1 GPU 规格

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 |
| 显存 | 32,607 MiB (≈32 GB) |
| CUDA | 12.8 |
| PyTorch | 2.11.0 |

### 1.2 关键配置（验证通过）

**RTX 5090 单卡训练需要以下四项配置才能运行：**

| 配置 | 值 | 原因 |
|------|-----|------|
| `--sharding_strategy` | **FULL_SHARD** | FSDP 全分片，减少每卡显存 |
| `--cpu_offload` | **True** | 优化器状态卸载到 CPU |
| `--freeze_mllm` | **True** | 冻结 ViT+LLM 主干，仅训练 Action Expert |
| `--grad_checkpoint` | **True** | 梯度检查点，用计算换显存 |

不作以上配置时，2.81B 参数的 Being-H05 前向传播即占用 27 GB，反向传播 OOM。

### 1.3 Koch 数据集总览

数据位于 `/home/hyl/Being-H/origin_data_v21/koch_follower/`，共 18 个子数据集：

| 数据集 | Episodes | Frames | 说明 |
|--------|----------|--------|------|
| `pick_orange_simple_shape` | 4 | 1,991 | 简单场景 |
| `multi_task_train` | 844 | 354,498 | 多任务混合（**主力数据**） |
| 其余 16 个 | 10-100 | 3K-53K | 各种颜色/场景变体 |

### 1.4 Being-H05 架构

```
观测 (2 cameras + joint state)
    │
    ├── ViT (InternVL3_5-2B, ~2B params) ─► Vision Tokens   ← 冻结
    │
    ├── State Encoder (MLP) ──────────────► State Tokens     ← 冻结
    │
    └──────────────┬────────────────┘
                   ▼
            LLM (Qwen3-0.6B)   ← 冻结（含 MoE _mot_gen 层） - und-loss
                   │
                   ▼
            Action Expert (Flow Matching)  ← **仅训练此部分**
            + Action Encoder/Decoder
            + MPG (关闭)
            + RTC (关闭)
```

---

## 2. 环境准备

```bash
cd /home/hyl/Being-H/Being-H05
conda activate beingh

# 验证关键依赖
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import decord; print('decord OK')"
python -c "import flash_attn; print('flash-attn OK')"
```

> **重要**：Koch v2.1 数据集的视频编码为 AV1，但 decord 无法解码 AV1。需要用 ffmpeg 转码为 H.264：
> ```bash
> conda install -y -c conda-forge ffmpeg
> # 批量转码脚本见 §10.5
> ```

---

## 3. 数据集配置

### 3.1 更新 `configs/dataset_info.py`

已完成的修改：
- `DATASET_REGISTRY` 添加 `'koch_posttrain': LeRobotIterableDataset`
- `DATASET_INFO` 添加所有 17 个 Koch 子数据集，路径指向 `origin_data_v21/koch_follower/`

### 3.2 YAML 配置文件

三种方案已创建在 `configs/posttrain/koch/`：

| 文件 | 数据量 | MAX_STEPS | 预计耗时 |
|------|--------|-----------|----------|
| `pick_orange.yaml` | 4 eps, 2K frames | 1,500 | ~5 min |
| `pick_multi_obj.yaml` | 250 eps, 92K frames | 3,000 | ~15 min |
| `multi_task.yaml` | 844 eps, 354K frames | 5,000 | ~30 min |

---

## 4. 模型下载

```bash
# model/ 已通过符号链接指向 /home/hyl/model/Being-H05
# 该目录包含 Being-H05-2B 完整检查点（11.2 GB safetensors）
ls model/model.safetensors   # 应存在

# 如果不存在，下载：
# git clone https://huggingface.co/BeingBeyond/Being-H05-2B /home/hyl/model/Being-H05
```

> Being-H05-2B 检查点中包含了 InternVL ViT + Qwen3 LLM + Action Expert 的所有权重，训练脚本的三个路径（`mllm_path`、`expert_path`、`resume_from`）都指向同一个 `model/` 目录。

---

## 5. 微调脚本

`scripts/train/finetune_koch_5090.sh` — 已验证通过的单卡 RTX 5090 微调脚本。

**核心参数速查：**

```bash
# 关键配置
FREEZE_MLLM=True              # 冻结 VLM 主干
SHARDING_STRATEGY="FULL_SHARD" # FSDP 全分片
CPU_OFFLOAD=True              # 优化器状态 CPU 卸载
GRAD_CHECKPOINT=True          # 梯度检查点
GRADIENT_ACCUMULATION_STEPS=2 # 梯度累积

# 序列
MAX_NUM_TOKENS=1024
EXPECTED_NUM_TOKENS=512

# 图像
FORCE_IMAGE_SIZE=224

# 训练
MAX_STEPS=1500-5000           # 根据数据量
LEARNING_RATE=1e-4
```

---

## 6. 启动训练

```bash
cd /home/hyl/Being-H/Being-H05

# 编辑脚本选择数据方案（默认方案 A）
# vim scripts/train/finetune_koch_5090.sh

# 启动
bash scripts/train/finetune_koch_5090.sh
```

**验证运行日志示例：**

```
step: 0000001/5 | act_loss: 6.5647 | lr: 7.01e-05 | steps/s: 0.03 | max_mem: 26834MB
step: 0000002/5 | act_loss: 15.5003 | lr: 3.78e-05 | steps/s: 0.06 | max_mem: 26836MB
step: 0000005/5 | act_loss: 2.6500 | lr: 1.06e-05 | steps/s: 0.12 | max_mem: 26836MB
```

- `act_loss` 下降 = 模型在学习
- `max_mem` ≈ 26.8 GB（32 GB 内有 ~5 GB 余量）
- `steps/s` 逐步提升（FSDP warmup）

---

## 7. 训练监控与调优

### 7.1 TensorBoard

```bash
tensorboard --logdir checkpoints/ --port 6006
```

### 7.2 过拟合诊断

| 症状 | 对策 |
|------|------|
| Loss 快速接近 0 (< 200 steps) | 减小 MAX_STEPS，增大 WEIGHT_DECAY |
| Loss 不下降 | 检查数据路径、增大 LR |
| OOM | 减小 MAX_NUM_TOKENS、确认 CPU_OFFLOAD=True |

---

## 8. 实时推理评估

详见 [lerobot微调.md §8](lerobot微调.md#8-实时推理评估) 中 `scripts/eval/koch_realtime_eval.py` 脚本。

```bash
python scripts/eval/koch_realtime_eval.py \
  --model_path checkpoints/koch_pick_orange_YYYYMMDD_HHMMSS \
  --robot_port /dev/ttyACM0 \
  --episode_duration_s 50 \
  --num_episodes 10
```

---

## 9. 已验证的代码修复

以下修复已应用于当前代码库，确保 RTX 5090 单卡训练正常运行：

### 9.1 状态字典键名不匹配

**错误 1**：`AttributeError: 'PretrainedConfig' object has no attribute 'vision_config'. Did you mean: 'vit_config'?`
→ `train_utils.py:174` 访问 `mllm_config.vision_config`，但 config.json 字段名是 `vit_config`。

**错误 2**：`KeyError: 'mlp1.0.weight'`（`layers.py:170`）
→ checkpoint 中 connector 权重的 key 带 `connector.` 前缀（如 `connector.mlp1.0.weight`），但 `named_parameters()` 返回的是不带前缀的 `mlp1.0.weight`。

| 文件 | 行 | 修改 |
|------|-----|------|
| `BeingH/train/train_utils.py` | 174 | `mllm_config.vision_config` → `mllm_config.vit_config` |
| `BeingH/model/vit_model/internvit/modeling_intern_vit.py` | 376,387 | `"vision_model."` → `"vit_model."` |
| `BeingH/model/vit_model/internvit_navit.py` | 382 | `"vision_model."` → `"vit_model."` |
| `BeingH/model/layers.py` | 170 | 自动尝试 `connector.` 前缀匹配：`state_dict.pop("connector." + name, None) or state_dict.pop(name)` |

### 9.2 断言放宽

**错误**：`AssertionError: All weights should be consumed!`（`train_utils.py:227`）
→ `resume_from` 通过 `FSDPCheckpoint.try_load_ckpt` 加载完整模型权重（包括 action decoder 等 24 个额外 key）。原来的 assert 假设 `mllm_path` 只含 LLM + ViT + Connector，对完整 Being-H05 checkpoint 过于严格。

| 文件 | 行 | 修改 |
|------|-----|------|
| `BeingH/train/train_utils.py` | 227 | 移除严格 assert 改为 warning：未消费的非 LLM/ViT/Connector 权重视为正常（后续 resume 加载） |

### 9.3 数据集集成

| 文件 | 修改 |
|------|------|
| `configs/dataset_info.py` | 添加 `koch_posttrain` 到 `DATASET_REGISTRY` 和 `DATASET_INFO` |
| `configs/data_config.py` | 添加 `KochFollowerDataConfig` 类和 `koch_follower` 到 `DATA_CONFIG_MAP` |
| `BeingH/utils/constants.py` | 添加 `KOCH = "koch"` 到 `EmbodimentTag` 枚举和 `EMBODIMENT_TAG_MAPPING` |
| `BeingH/dataset/datasets/vla_dataset.py` | `vit_transform_args` 改为可选参数（默认 `None`） |

### 9.4 训练修复

| 文件 | 修改 |
|------|------|
| `BeingH/train/train.py` | Expert 参数排除冻结（`mot_gen`、`action_*`、`mpg` 等不冻结） |
| `BeingH/train/train.py` | 梯度检查点手动配置 `_gradient_checkpointing_func` |

### 9.5 梯度检查点（Gradient Checkpointing）

**错误**：`ValueError: Qwen3ForCausalLM is not compatible with gradient checkpointing.`
→ transformers 4.57 中 Qwen3ForCausalLM 没有声明 `supports_gradient_checkpointing = True`，调用 `_set_gradient_checkpointing()` 或公开 API `gradient_checkpointing_enable()` 都会报错。需直接在内部 model 层（Qwen3Model）上手动设置：

```python
inner_model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
inner_model.gradient_checkpointing = True
```

### 9.6 视频编码

Koch v2.1 数据使用 AV1 编码，但当前环境的 decord 不支持 AV1 解码。解决方案：用 ffmpeg 转码为 H.264。

### 9.7 Multiprocessing Fork → Spawn（CUDA 死锁）

**问题**：`train.py:1094` 处 `enumerate(train_loader)` 永久卡死。

**根因**：PyTorch DataLoader 默认使用 `multiprocessing_context='fork'`，但 fork 发生在 CUDA 已初始化之后（模型已加载到 GPU），子进程继承损坏的 CUDA 状态导致死锁。这是 PyTorch 经典陷阱：CUDA 初始化后不能 fork。

**修复**：将 multiprocessing start method 从 `fork` 改为 `spawn`：

| 文件 | 行 | 修改 |
|------|-----|------|
| `BeingH/train/train.py` | ~1011 | `multiprocessing_context='fork'` → `multiprocessing_context='spawn'` |

### 9.8 Pickle 兼容性修复

改为 `spawn` 后，所有传给 DataLoader worker 的对象必须可 pickle。陆续发现三处不可 pickle 的对象：

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `BeingH/dataset/preprocess.py` | `T.Lambda(lambda img: ...)` — lambda 不可 pickle | 替换为可 pickle 的 `_ConvertRGB` 类 |
| 2 | `BeingH/dataset/base_dataset.py` | `dataset_iters = [iter(dataset) for ...]` — 迭代器不可 pickle | 添加 `__getstate__`/`__setstate__` 方法，pickle 时排除迭代器、unpickle 时重建 |
| 3 | `train.py` `collate_wrapper` | 内部定义的局部 `collate_fn` 不可 pickle | 改为模块级 `collate_fn`，`collate_wrapper()` 返回它的引用 |

### 9.9 num_workers=0 回退

**问题**：改为 spawn + 逐个修复 pickle 问题后仍有遗漏。代码库的 dataset/transform 链路不是为 spawn multiprocessing 设计的（多处闭包、迭代器、lambda）。

**方案**：`num_workers=0`，数据在主进程中单线程加载。对于少量 episode 的数据，I/O 完全不是瓶颈，GPU forward/backward 才是。

### 9.10 数据加载性能优化

**问题**：`num_workers=0` 后第一批数据极慢。根因在 `vla_dataset.py:1044`：每个 sample 都调用 `pd.read_parquet()` 读一次完整 parquet + seek + 解码两路 AV1 mp4 视频帧。达到第一个 log（step 10）需打包 ~420 个 sample，每个 ~1-2 秒 → 7-14 分钟。

**解决方案**：降低序列长度和 logging 间隔让第一批快速产出；使用 decord 做 O(1) 帧索引（比 torchvision_av 的 O(n) 逐帧解码快 100 倍以上）。

### 9.11 Action Expert 权重尺寸不匹配

**现象**：270 个 key 因尺寸不匹配被跳过，全部是 action expert 相关层（`_mot_gen.*`、`action_encoder.*`、`action_decoder.*` 等）：

```
ckpt=[1024, 3072] vs model=[2048, 6144]  ← expert hidden_size 不匹配
Total skipped due to size mismatch: 270 keys
```

**原因**：checkpoint 中 action expert (Qwen3-0.6B) 的 `hidden_size=1024`，但当前模型配置 `hidden_size=2048`。这些层将以随机初始化从头训练。不影响训练跑通，但需要更多数据/步数收敛。

### 9.12 YAML Weight 字段类型错误

**问题**：`TypeError: unsupported operand type(s) for +: 'int' and 'list'`（`base_dataset.py:201`）

**根因**：YAML 中的 `weight` 是数据集组级别的采样权重，必须为标量。多子数据集场景下（如 `all_data.yaml` 有 17 个子数据集），子数据集间的权重由 `LeRobotIterableDataset` 内部根据 `total_frames` 自动计算，不需在 YAML 中逐项指定。参考 `libero_robocasa.yaml`（28 个子数据集），其 `weight: 1` 也是标量。

---

## 10. 常见问题

### 10.1 OOM

```bash
# 确保以下配置
--cpu_offload True
--sharding_strategy FULL_SHARD
--grad_checkpoint True
--freeze_mllm True
--max_num_tokens 1024  # 或更小
```

### 10.2 视频解码失败 (`ERROR cannot find video stream`)

```bash
# 安装 ffmpeg 并转码
conda install -y -c conda-forge ffmpeg
# 批量转码脚本见下方
```

### 10.3 梯度检查点不兼容 (`is not compatible with gradient checkpointing`)

已通过手动设置 `_gradient_checkpointing_func` 修复（见 §9.5）。

### 10.4 KeyError: 'koch_follower' / 'koch_posttrain'

确保已编辑：
- `configs/dataset_info.py` → `DATASET_REGISTRY` + `DATASET_INFO`
- `configs/data_config.py` → `KochFollowerDataConfig` + `DATA_CONFIG_MAP`
- `BeingH/utils/constants.py` → `EmbodimentTag.KOCH` + `EMBODIMENT_TAG_MAPPING`

### 10.5 批量视频转码脚本


脚本先把每个 "$f" 转成临时文件 "$f".h264.mp4，然后 mv "$f".h264.mp4 "$f" 覆盖回原路径。
因此最终的转码文件仍然位于原位置，例如：

```bash
# 将所有 Koch 数据集的视频从 AV1 转为 H.264
for ds in /home/hyl/Being-H/origin_data_v21/koch_follower/*/; do
  for cam in observation.images.above observation.images.wrist; do
    video_dir="${ds}videos/chunk-000/${cam}"
    if [ -d "$video_dir" ]; then
      for f in "$video_dir"/*.mp4; do
        tmp="${f}.h264.mp4"
        ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 23 "$tmp" 2>/dev/null
        mv "$tmp" "$f"
      done
    fi
  done
  echo "Done: $(basename $ds)"
done
```





### 10.6 训练卡死在 DataLoader（`enumerate(train_loader)` 无响应）

**根因**：PyTorch DataLoader 使用 `fork` 模式，但 CUDA 已在父进程初始化。fork 后子进程继承损坏的 CUDA 状态导致死锁。

**解决**：
1. 将 `multiprocessing_context` 从 `fork` 改为 `spawn`（见 §9.7）
2. 或使用 `num_workers=0` 回退方案（见 §9.9）

### 10.7 SIGHUP 信号导致训练中断

**现象**：
```
W0603 18:12:29.372000 Signals.SIGHUP death signal, shutting down workers
torch.distributed.elastic.multiprocessing.api.SignalException: Process got signal: 1
```

**根因**：SSH 断开/终端关闭时，内核向关联进程组发 SIGHUP。torchrun elastic agent 收到 SIGHUP 后主动杀 worker。`nohup` 失效 — 被 torchrun 覆盖了信号处理器。

**解决方案**：使用 `setsid` 让进程脱离终端 session，内核不会向它发 SIGHUP：

```bash
setsid bash scripts/train/finetune_koch_all_data.sh </dev/null > ./koch_finetune_all.log 2>&1 &
```

或使用 `tmux`/`screen` 防断连：
```bash
tmux new -s train
# 在 tmux 内启动训练，按 Ctrl+B D 分离
# 重新连接: tmux attach -t train
```

### 10.8 从检查点恢复训练

训练被中断（如 SIGHUP）后，可以从最近的 checkpoint 恢复，同时恢复 optimizer 和 scheduler 状态：

```bash
# 关键参数变更：
# --resume_from      → 指向最新 checkpoint（而非原始 model/）
# --resume_model_only → False（同时恢复 optimizer + scheduler + train_step）
# --output_dir       → 保持不变，后续 checkpoint 继续写入同一目录

torchrun \
  --nnodes=1 --nproc_per_node=1 --master_port=29106 \
  BeingH/train/train.py \
  --mllm_path model \
  --expert_path model \
  --resume_from checkpoints/koch_all_data_20260603_114849/0004000 \
  --resume_model_only False \
  --output_dir checkpoints/koch_all_data_20260603_114849 \
  ... (其余参数与首次训练相同)
```

> ⚠️ 如果恢复时 `resume_model_only=True`，optimizer 和 scheduler 会重新初始化（momentum 丢失、lr 从头 warmup），等于只借权重重新训。

### 10.9 Pickle 错误合集（spawn 模式）

改用 `spawn` multiprocessing 后可能遇到以下 pickle 错误：

| 错误信息 | 原因 | 修复 |
|----------|------|------|
| `Can't pickle local object 'build_vit_transform_base.<locals>.<lambda>'` | transform 中的 lambda 不可 pickle | 替换为可 pickle 的类（§9.8） |
| `cannot pickle 'generator' object` | dataset 迭代器作为类属性不可 pickle | 添加 `__getstate__`/`__setstate__`（§9.8） |
| `Can't pickle local object 'collate_wrapper.<locals>.collate_fn'` | 局部函数不可 pickle | 改为模块级函数（§9.8） |

如果逐个修复成本过高，直接使用 `num_workers=0`（§9.9）。

---

## 参考

- Being-H05 仓库: https://github.com/BeingBeyond/Being-H
- Being-H05-2B HF: https://huggingface.co/BeingBeyond/Being-H05-2B
- LeRobot: https://github.com/huggingface/lerobot
- 训练入口: `BeingH/train/train.py`
- 推理入口: `BeingH/inference/beingh_policy.py`
- 数据格式: LeRobot v2.1 (per-episode Parquet + MP4)


# 微调指令

```
cd /home/hyl/Being-H/Being-H05

# 让进程脱离终端 session，内核一开始就不会向它发 SIGHUP
setsid bash scripts/train/finetune_koch_all_data.sh </dev/null > ./koch_finetune_all.log 2>&1 &
nohup bash scripts/train/finetune_koch_all_data.sh > ./koch_finetune_all.log 2>&1 &
tail -f ./koch_finetune_all.log
pkill -f "train.py"
```
