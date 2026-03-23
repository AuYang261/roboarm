# 机械臂视觉目标检测 VLM 微调 — 完整复现指南

> 面向本科生的零基础复现教程
> 目标：让完全不了解大模型微调的同学也能一步步跑通整个流程

---

## 目录

1. [项目概述](#1-项目概述)
2. [你需要了解的基本概念](#2-你需要了解的基本概念)
3. [硬件与系统要求](#3-硬件与系统要求)
4. [目录结构说明](#4-目录结构说明)
5. [Step 1：安装基础工具](#step-1安装基础工具)
6. [Step 2：创建 Python 虚拟环境并安装 PyTorch](#step-2创建-python-虚拟环境并安装-pytorch)
7. [Step 3：安装 ms-swift 及其他依赖](#step-3安装-ms-swift-及其他依赖)
8. [Step 4：准备图片和 YOLO 模型](#step-4准备图片和-yolo-模型)
9. [Step 5：生成训练数据](#step-5生成训练数据)
10. [Step 6：数据集拆分](#step-6数据集拆分)
11. [Step 7：下载基础模型](#step-7下载基础模型)
12. [Step 8：启动训练](#step-8启动训练)
13. [Step 9：合并 LoRA 权重](#step-9合并-lora-权重)
14. [Step 10：部署推理服务](#step-10部署推理服务)
15. [Step 11：测试推理效果](#step-11测试推理效果)
16. [常见问题与解决方案](#常见问题与解决方案)
17. [数据格式详解](#数据格式详解)

---

## 1. 项目概述

### 我们要做什么？

训练一个**能看图、能听指令、能定位物体**的 AI 模型，用来控制机械臂抓取积木。

具体流程：
```
用户发图片 + 语音指令（如"抓最近的蓝色积木"）
         ↓
    VLM 模型分析
         ↓
输出边界框坐标（物体在图片中的位置和大小）
         ↓
  机械臂根据坐标运动抓取
```

### 输出格式

模型输出一个 JSON，描述目标物体的位置：

```json
{
    "id": 1,
    "failed": false,
    "class_name": "蓝色积木",
    "box_center_x": 0.821,
    "box_center_y": 0.131,
    "box_width": 0.142,
    "box_height": 0.508
}
```

坐标是**归一化坐标**（0~1 之间），其中：
- `box_center_x/y`：目标物体中心点坐标（左上角为原点，向右/向下为正方向）
- `box_width/height`：物体边界框的宽和高

---

## 2. 你需要了解的基本概念

> 如果你已熟悉这些概念，可以跳过本节。

### 什么是 VLM？

VLM（Vision-Language Model，视觉语言模型）是能同时处理**图片 + 文字**的 AI 模型。
本项目使用的是阿里巴巴开源的 **Qwen3.5-9B**（90 亿参数）。

### 什么是微调（Fine-tuning）？

预训练的大模型很通用，但不一定擅长特定任务（比如识别积木位置）。
微调就是用你自己的数据，在预训练模型基础上继续训练，让它学会做你的特定任务。

### 什么是 LoRA？

LoRA（Low-Rank Adaptation）是一种**高效微调方法**：
- 不修改原模型的全部参数（参数量太大，显存和计算量都撑不住）
- 只在原模型旁边附加少量新参数（"适配器"），训练这些新参数
- 训练完成后，可以把 LoRA 参数"合并"回原模型，得到一个完整的微调模型

类比：LoRA 就像给原模型"贴了一层薄薄的知识贴片"，而不是重写整本教科书。

### 什么是 YOLO？

YOLO（You Only Look Once）是一个目标检测模型，能快速识别图片中物体的位置和类别。
本项目用预训练好的 YOLO OBB（旋转目标检测）模型自动标注训练图片。

### 什么是 ms-swift？

[ms-swift](https://github.com/modelscope/ms-swift) 是阿里 ModelScope 开源的大模型微调框架，封装了训练、评估等流程，只需写几行命令就能启动微调。

---

## 3. 硬件与系统要求

| 项目 | 本项目配置 | 最低要求 |
|------|-----------|---------|
| GPU | NVIDIA RTX 5090（32GB 显存） | 24GB 显存（如 RTX 3090/4090/A10） |
| 内存 | 建议 64GB+ | 32GB |
| 磁盘 | 约 100GB 空闲空间 | 80GB |
| 操作系统 | Ubuntu Linux | 任意 Linux 发行版 |
| CUDA Driver | 12.9 | ≥12.1 |
| Python | 3.11 | 3.10 ~ 3.12 |

> **注意**：如果你的 GPU 显存少于 24GB，需要将 `train.sh` 中的 `--lora_rank` 减小（如改为 8），或使用 `--quantization_bit 4`（4bit 量化训练）。

---

## 4. 目录结构说明

```
llm_tuning/
├── .venv/                      # Python 虚拟环境（运行所有脚本都要先激活它）
├── pic/                        # 手工标注的验证集图片（18张）
├── original/                   # 自动采集的训练图片（101张）
├── original_preview/           # YOLO 标注可视化图片（供人工检查质量）（运行脚本生成）
├── annotated/                  # 外接矩形可视化图片（运行脚本生成）
├── yolo/
│   └── best.pt                 # 预训练的 YOLO OBB 模型（识别积木）
├── annotate_dataset.py         # 【脚本1】用 YOLO 标注图片 → 生成训练数据
├── draw_json_preview.py        # 【脚本1.1】生成YOLO 标注可视化图片original_preview/
├── split_dataset.py            # 【脚本2】拆分为训练集/测试集
├── train.sh                    # 【脚本3】启动训练
├── resume_train.sh             # 【脚本4】从断点恢复训练
├── merge_lora_peft.py          # 【脚本5】合并 LoRA 权重到基础模型
├── merge_lora.sh               # 【脚本6】启动合并脚本merge_lora_peft.py
├── serve_vllm.sh               # 【脚本7】启动推理服务
├── kill_vllm.sh                # 【脚本8】停止推理服务
├── eval_vis.py                 # 【脚本9】可视化评估
├── val.jsonl                   # 验证集（37条，来自 pic/ 手工标注）（运行脚本生成）
├── train_split.jsonl           # 训练集（862条，original/ 的 80%）（运行脚本生成）
├── test_split.jsonl            # 测试集（212条，original/ 的 20%）（运行脚本生成）
└── output/                     # （运行脚本生成）
    ├── train.log               # 训练日志
    ├── qwen3.5-9b-graspdet/    # 训练产生的 checkpoint
    └── merged-qwen3.5-9b-graspdet/  # 合并后的完整模型
```

---

## Step 1：安装基础工具

### 1.1 安装 uv（Python 包管理器）

`uv` 是一个比 pip 快很多的 Python 包管理工具。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

验证安装：
```bash
uv --version
# 输出类似：uv 0.7.x
```

### 1.2 克隆/获取项目代码

如果是已有项目目录，进入目录：
```bash
cd /home/xjy/llm_tuning
```

---

## Step 2：创建 Python 虚拟环境并安装 PyTorch

### 2.1 创建虚拟环境

```bash
# 在项目目录下创建 .venv 虚拟环境（Python 3.11）
uv venv .venv --python 3.11

# 激活虚拟环境
source .venv/bin/activate
```

激活成功后，命令行提示符前面会出现 `(.venv)`：
```
(.venv) xjy@machine:~/llm_tuning$
```

> **重要**：每次新开终端都需要重新激活虚拟环境！

### 2.2 安装 PyTorch

根据你的 CUDA 版本选择安装命令：

**CUDA 12.1 ~ 12.9（RTX 5090 或其他新卡）：**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 如果网速很慢，可以先设置代理（如果有的话）：
> ```bash
> export https_proxy=http://127.0.0.1:7890
> export http_proxy=http://127.0.0.1:7890
> ```

### 2.3 验证 PyTorch 安装

```bash
python -c "import torch; print(torch.__version__); print('GPU可用:', torch.cuda.is_available()); print('GPU名称:', torch.cuda.get_device_name(0))"
```

预期输出：
```
2.10.0+cu128
GPU可用: True
GPU名称: NVIDIA GeForce RTX 5090
```

---

## Step 3：安装 ms-swift 及其他依赖

```bash
# 安装 ms-swift（包含所有可选依赖）
uv pip install "ms-swift[all]"

# 安装其他依赖（图像处理、YOLO等）
uv pip install ultralytics opencv-python safetensors
```

> `ms-swift[all]` 会安装 transformers、accelerate、peft 等一系列依赖，过程可能需要几分钟。

验证安装：
```bash
swift --version
# 输出类似：ms-swift 4.0.2
```

---

## Step 4：准备图片和 YOLO 模型

### 4.1 图片准备

本项目的图片是用相机拍摄的机械臂工作台图片，包含红/蓝/黄三色积木。

目录结构：
- `pic/`：18 张手工精标注的图片（用作验证集，已有对应标注数据）
- `original/`：101 张新采集的图片（将用 YOLO 自动标注）

如果你要自己采集数据，建议：
- 在不同光线条件下拍摄
- 每张图包含 2~5 个积木
- 分辨率不限，但建议不低于 640×480

### 4.2 YOLO 模型

训练好的 YOLO OBB 模型已放置在 `yolo/best.pt`，是用验证集pic/中的数据训练的，能识别：
- `red_block`（红色积木）
- `blue_block`（蓝色积木）
- `yellow_block`（黄色积木）

> 如果你换了其他类别的物体，需要重新训练 YOLO 模型，这里不展开介绍。

---

## Step 5：生成训练数据

这一步用 YOLO 模型自动标注 `original/` 中的 101 张图片，生成 VLM 训练数据，输出到 `merged_train.jsonl`。

```bash
source .venv/bin/activate
python annotate_dataset.py --input-jsonl /dev/null
# 若没有val.jsonl可以用如下命令用YOLO标注生成
# 我是用人工标签生成的，也可以自己写个脚本（参考annotate_dataset.py）用人工标签生成，差别不大
python annotate_dataset.py --original-dir pic/ --output val.jsonl
```

> `--input-jsonl /dev/null` 表示不读入任何旧数据，只处理 `original/` 目录下的图片。输出文件默认为 `merged_train.jsonl`（供下一步拆分用）。

### 脚本做了什么？

1. 对每张图片运行 YOLO 推理，得到积木的**旋转边界框**（含角度）
2. 将旋转框转换为**轴对齐矩形框**（只有上下左右，没有旋转），坐标归一化到 0~1
3. 对每张图片生成最多 **11 种不同的自然语言指令**（见下表），每种指令配上对应标注
4. 把标注可视化图片保存到 `annotated/`（供人工检查质量）
5. 输出 `merged_train.jsonl`（1074条，仅含 `original/` 图片数据）

**11 种指令类型：**

| 指令示例 | 类型 |
|---------|------|
| "抓取蓝色积木" | 颜色 |
| "抓取红色积木" | 颜色 |
| "抓取黄色积木" | 颜色 |
| "抓取最近的积木" | 距离（y坐标最大=最近机械臂） |
| "抓取最远的积木" | 距离（y坐标最小） |
| "抓取最左边的积木" | 方向 |
| "抓取最右边的积木" | 方向 |
| "抓取最近的蓝色积木" | 颜色+距离 |
| "抓取最近的红色积木" | 颜色+距离 |
| "抓取最右边的蓝色积木" | 颜色+方向（需≥2个蓝色才生成） |
| "抓取最左边的红色积木" | 颜色+方向（需≥2个红色才生成） |

### 预期输出

```
找到 101 张图片。
加载 YOLO 模型: yolo/best.pt
  处理: img_0001.jpg → 检测到 4 个目标，生成 10 条训练样本
  处理: img_0002.jpg → 检测到 3 个目标，生成 9 条训练样本
  ...
=== 完成 ===
  处理图片: 101 张（0 张无检测结果）
  新生成样本: 1074 条
  原有样本: 0 条
  合并输出: merged_train.jsonl（共 1074 条）
  标注图片: annotated/
```

### 可视化检查

打开 `annotated/` 目录看几张图，确认标注框正确框住了积木：
- 实线框 = YOLO OBB 旋转框
- 细线框 = 对应的轴对齐外接矩形（这是训练数据中实际使用的框）

如果发现很多漏检或误检，可以调整置信度：
```bash
python annotate_dataset.py --conf 0.7   # 提高置信度，减少误检
python annotate_dataset.py --conf 0.5   # 降低置信度，减少漏检
```

---

## Step 6：数据集拆分

把 `merged_train.jsonl` 按图片随机划分为训练集和测试集。`val.jsonl` 来自 `pic/` 手工标注数据，已单独存在，不参与此步骤。

```bash
python split_dataset.py
```

**划分规则：**
- 验证集（`val.jsonl`）：来自 `pic/` 的 18 张手工精标注图片，共 37 条，已存在，**不由此脚本生成**
- 训练集（`train_split.jsonl`）：`merged_train.jsonl` 中 80% 的图片（81张，862条）
- 测试集（`test_split.jsonl`）：`merged_train.jsonl` 中 20% 的图片（20张，212条）

> **为什么按图片而不是按样本划分？**
> 每张图会生成 9-11 条样本（11种指令）。如果按样本随机划分，可能同一张图的部分样本在训练集，另一部分在测试集——模型见过这张图，测试结果就不可靠了。按图片分组划分可以避免这个问题。

预期输出：
```
验证集  val.jsonl                :    0 条  (0 张图)   ← merged_train.jsonl 中无 pic/ 数据，正常
训练集  train_split.jsonl        :  862 条  (81 张图)
测试集  test_split.jsonl         :  212 条  (20 张图)
```

---

## Step 7：下载基础模型

Qwen3.5-9B 模型约 18GB，需要提前下载。

### 方式一：ModelScope（国内推荐，速度快）

```bash
# 安装 modelscope
uv pip install modelscope

# 下载模型（会存到 ~/.cache/modelscope/）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"
```

### 方式二：Hugging Face

```bash
# 建议用镜像站
export HF_ENDPOINT=https://hf-mirror.com

pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/.cache/huggingface/hub/Qwen3.5-9B
```

### 验证下载

```bash
ls ~/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B/
# 应该看到很多 model-xxxx-of-xxxx.safetensors 文件
```

> 如果用 Hugging Face 下载，需要修改 `merge_lora_peft.py` 中的 `MODEL_DIR` 路径。

---

## Step 8：启动训练

### 8.1 确认训练脚本内容

查看 [train.sh](train.sh)，关键参数解释：

```bash
swift sft \
    --model Qwen/Qwen3.5-9B \          # 基础模型名称（会自动从缓存加载）
    --train_type lora \                 # 使用 LoRA 微调
    --dataset train_split.jsonl \       # 训练集
    --val_dataset val.jsonl \           # 验证集（每100步评估一次）
    --output_dir output/qwen3.5-9b-graspdet \  # 输出目录
    --num_train_epochs 10 \             # 训练 10 个 epoch
    --per_device_train_batch_size 1 \   # 每步处理1个样本（显存限制）
    --gradient_accumulation_steps 16 \  # 累积16步再更新，等效 batch_size=16
    --learning_rate 1e-4 \             # 学习率
    --lora_rank 16 \                   # LoRA 秩（越大参数越多，越慢）
    --lora_alpha 32 \                  # LoRA 缩放系数（通常是 rank 的 2 倍）
    --target_modules all-linear \      # 对所有线性层应用 LoRA
    --max_length 4096 \                # 最大序列长度（含图片 token）
    --warmup_ratio 0.1 \              # 前 10% 步数做 warmup（学习率从0升到目标值）
    --save_steps 100 \                # 每 100 步保存一次 checkpoint
    --eval_steps 100 \                # 每 100 步评估一次
    --logging_steps 10 \              # 每 10 步打印一次 loss
    --bf16 true \                     # 用 bfloat16 精度（节省显存）
    --gradient_checkpointing true \   # 梯度检查点（节省显存，略微降低速度）
    --attn_impl sdpa \                # 使用 PyTorch 内置 SDPA，不需要 flash-attn
    --dataloader_num_workers 4        # 4 个 worker 并行加载数据
```

### 8.2 启动训练

```bash
cd /home/xjy/llm_tuning
bash train.sh
```

训练日志会同时输出到终端和 `output/train.log`。

### 8.3 训练过程监控

**观察 loss 曲线：**
```bash
# 实时查看训练日志
tail -f output/train.log

# 或者过滤只看 loss 行
grep "loss" output/train.log | tail -20
```

正常 loss 变化趋势（参考）：
```
step 10:  train_loss=2.5
step 50:  train_loss=1.2
step 100: train_loss=0.5
step 200: train_loss=0.1
step 540: train_loss=0.002  ← 本项目最终结果
```

**GPU 显存监控：**
```bash
# 新开一个终端运行
watch -n 2 nvidia-smi
```

### 8.4 如果训练中断了（恢复训练）

```bash
# 查看已有的 checkpoint
ls output/qwen3.5-9b-graspdet/v1-*/

# 修改 resume_train.sh 中的 checkpoint 路径，然后运行
bash resume_train.sh
```

### 8.5 训练时间参考

| 配置 | 时间 |
|------|------|
| RTX 5090，862条样本，10 epochs | 约 5 小时（34s/step，共540步） |
| RTX 4090，同配置 | 约 7~8 小时 |

> **提示**：可以后台运行训练，关掉终端也不会中断：
> ```bash
> nohup bash train.sh &
> tail -f output/train.log  # 查看进度
> ```

---

## Step 9：合并 LoRA 权重

训练完成后，LoRA 参数保存在 `output/qwen3.5-9b-graspdet/` 下的 checkpoint 目录中，是单独的"差量"文件，还不是完整的可部署模型。需要把 LoRA 参数合并回基础模型。

### 9.1 修改合并脚本中的路径

打开 [merge_lora_peft.py](merge_lora_peft.py)，找到第 15-17 行，修改为你的实际路径：

```python
# 修改这里的 checkpoint 路径（选择 val_loss 最低的那个）
ADAPTER   = BASE_DIR / "output/qwen3.5-9b-graspdet/v1-xxxxxxxx-xxxxxx/checkpoint-100"
# 修改为 Qwen3.5-9B 模型的实际缓存路径
MODEL_DIR = Path("/home/xjy/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B")
# 合并后的输出路径（可保持默认）
OUT_DIR   = BASE_DIR / "output/merged-qwen3.5-9b-graspdet"
```

**如何确定用哪个 checkpoint？**
```bash
# 查看训练日志中 best_model_checkpoint 一行
grep "best_model" output/train.log | tail -1
# 或者选 val_loss 最低的 checkpoint
grep "eval_loss" output/train.log
```

### 9.2 运行合并

```bash
bash merge_lora.sh
```

预期输出：
```
LoRA rank=16, alpha=32, scale=2.0000
LoRA modules: 248
Delta keys: 248
  Saved model-00001-of-00005.safetensors  (xxx tensors, 48 merged so far)
  Saved model-00002-of-00005.safetensors  ...
  ...
Done. Merged 248/248 LoRA deltas → output/merged-qwen3.5-9b-graspdet
```

合并后，`output/merged-qwen3.5-9b-graspdet/` 就是一个完整的可部署模型目录。

---

## Step 10：部署推理服务

用 vLLM 启动一个兼容 OpenAI API 格式的推理服务：

```bash
bash serve_vllm.sh
```

服务启动后（约需 1~2 分钟加载模型），监听 `http://localhost:8001`。

**检查服务是否启动成功：**
```bash
# 查看日志
tail -f output/vllm_serve.log

# 看到如下内容说明启动成功：
# INFO: Application startup complete.
# INFO: Uvicorn running on http://0.0.0.0:8001
```

**停止服务：**
```bash
bash kill_vllm.sh
```

---

## Step 11：测试推理效果

### 11.1 用 curl 测试单条推理

把你的图片转为 base64，然后发送请求：

```bash
# 把图片转为 base64
IMG_B64=$(base64 -w 0 path/to/your/image.jpg)

# 发送推理请求
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"output/merged-qwen3.5-9b-graspdet\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$IMG_B64\"}},
        {\"type\": \"text\", \"text\": \"你是一个智能机械臂控制系统。请根据指令输出目标物体的边界框JSON。\n用户指令：抓取蓝色积木\"}
      ]
    }]
  }"
```

### 11.2 用 Python 测试

```python
import base64
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")

with open("path/to/image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="output/merged-qwen3.5-9b-graspdet",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": "抓取蓝色积木"}
        ]
    }]
)

output = response.choices[0].message.content
print("模型输出：", output)

# 解析 JSON（模型可能在 JSON 前输出 <think>...</think> 推理链）
import re
json_match = re.search(r'\{.*\}', output, re.DOTALL)
if json_match:
    result = json.loads(json_match.group())
    print("解析结果：", result)
```

### 11.3 可视化评估

```bash
python eval_vis.py
# 结果图片保存到 output/vis/
```

---

## 常见问题与解决方案

### Q1: `torch.cuda.is_available()` 返回 False

**可能原因：**
1. CUDA 驱动未安装或版本不匹配
2. 虚拟环境内的 PyTorch 和系统 CUDA 版本不匹配

**解决方法：**
```bash
# 检查驱动版本
nvidia-smi

# 根据驱动版本选择正确的 PyTorch wheel
# CUDA 12.1~12.9 → cu128
# CUDA 11.x      → cu118
```

### Q2: 训练时显存 OOM（Out of Memory）

**解决方法（按显存从小到大排列）：**

```bash
# 方法1：已在 train.sh 中设置，确认这行存在
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 方法2：减小 lora_rank
--lora_rank 8  # 从 16 改为 8

# 方法3：使用 4bit 量化（QLoRA）
--quantization_bit 4

# 方法4：减少 max_length
--max_length 2048  # 从 4096 改为 2048
```

### Q3: 下载模型速度很慢

```bash
# 设置代理（如果有）
export https_proxy=http://127.0.0.1:7890

# 或者用 ModelScope 镜像（国内速度快）
export MODELSCOPE_CACHE=/home/xjy/.cache/modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"
```

### Q4: `swift: command not found`

```bash
# 确保激活了虚拟环境
source .venv/bin/activate
# 或者用完整路径
.venv/bin/swift sft ...
```

### Q5: 训练 loss 不降

可能原因：
1. 学习率太大 → 尝试 `--learning_rate 5e-5`
2. 数据质量问题 → 检查 `annotated/` 中的可视化图片，看标注是否正确
3. epoch 数太少 → 增加 `--num_train_epochs 15`

### Q6: vLLM 启动失败

**问题1：** `layer_type_validation` 导入错误
这是 transformers 版本不兼容问题，需要在 vLLM 源码中添加 fallback：
```python
# 找到 .venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/qwen3_5.py
# 在 from transformers... 导入处加 try/except
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import layer_type_validation
except ImportError:
    def layer_type_validation(*args, **kwargs):
        pass
```

**问题2：** 第一次推理 Triton OOM
参考 vLLM PR #36599，在 `qwen3_next.py` 的 `_warmup_prefill_kernels` 方法中添加 OOM 保护。

### Q7: 模型输出格式不对（`<think>` 块）

Qwen3.5 默认会输出推理链：
```
<think>
用户想要蓝色积木...分析图片...
</think>
{"id": 1, "failed": false, ...}
```

用正则表达式过滤：
```python
import re
text = model_output
# 去掉 <think>...</think> 块
text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
# 解析 JSON
result = json.loads(text)
```

---

## 数据格式详解

### 训练数据格式（JSONL）

每行一个 JSON，格式如下：

```json
{
    "messages": [
        {
            "role": "user",
            "content": "<image>\n你是一个智能机械臂的视觉控制中枢...\n用户指令：「抓取蓝色积木」"
        },
        {
            "role": "assistant",
            "content": "{\"id\": 1, \"failed\": false, \"class_name\": \"蓝色积木\", \"box_center_x\": 0.512, \"box_center_y\": 0.345, \"box_width\": 0.142, \"box_height\": 0.208}"
        }
    ],
    "images": ["original/img_0001.jpg"]
}
```

字段说明：
- `messages`：对话历史，user 是用户输入（含 `<image>` 占位符），assistant 是期望的模型输出
- `images`：对应的图片路径列表（相对路径）

### 边界框坐标说明

```
图片坐标系：
(0,0) ─────────────── (1,0)
  │                     │
  │    ┌─────────┐      │
  │    │  box    │      │
  │    │  (cx,cy)│      │
  │    └─────────┘      │
  │                     │
(0,1) ─────────────── (1,1)

cx = box_center_x（中心点x，0=左边缘，1=右边缘）
cy = box_center_y（中心点y，0=上边缘，1=下边缘）
w  = box_width（框的宽度占图片宽度的比例）
h  = box_height（框的高度占图片高度的比例）

实际像素坐标换算：
  pixel_cx = cx * image_width
  pixel_cy = cy * image_height
  pixel_x1 = (cx - w/2) * image_width   # 左边界
  pixel_y1 = (cy - h/2) * image_height  # 上边界
  pixel_x2 = (cx + w/2) * image_width   # 右边界
  pixel_y2 = (cy + h/2) * image_height  # 下边界
```

---

## 整体流程回顾

```
1. 准备环境
   └── uv + Python 3.11 + PyTorch + ms-swift

2. 准备数据
   ├── pic/（手工标注验证集）
   └── original/（新采集图片）

3. 自动标注
   └── annotate_dataset.py
       ├── YOLO OBB 推理 → 检测积木位置
       ├── OBB → 轴对齐 bbox → 归一化坐标
       ├── 11种指令 → 11条训练样本/图
       └── merged_train.jsonl（1074条）

4. 拆分数据集
   ├── val.jsonl（37条，pic/ 手工标注，独立来源）
   └── split_dataset.py（对 merged_train.jsonl 按图片随机划分）
       ├── train_split.jsonl（862条）
       └── test_split.jsonl（212条）

5. 下载基础模型
   └── Qwen/Qwen3.5-9B（~18GB）

6. LoRA 微调训练
   └── train.sh → swift sft
       └── output/qwen3.5-9b-graspdet/checkpoint-xxx/

7. 合并 LoRA 权重
   └── merge_lora_peft.py
       └── output/merged-qwen3.5-9b-graspdet/

8. 部署服务
   └── serve_vllm.sh → http://localhost:8001

9. 使用模型
   └── 发送图片 + 指令 → 获取边界框 JSON
```

---

*如有问题，欢迎提 issue 或联系项目维护者。*
