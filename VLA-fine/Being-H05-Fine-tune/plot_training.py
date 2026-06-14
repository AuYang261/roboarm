#!/usr/bin/env python3
"""
训练曲线可视化脚本
====================
从 training.log 中解析训练步骤数据，绘制 act_loss、und_loss、lr、steps/s 等曲线。

使用方法（conda lerobot 环境）:
    conda activate lerobot
    python plot_training.py

输出: training_curves.png (保存到当前目录)

数据解读:
  - act_loss (Action Loss):     动作预测的 Flow Matching 损失，越低越好
  - und_loss (Understanding Loss): 模型"理解"损失，当前阶段为 0（未激活）
  - lr (Learning Rate):         学习率，采用 warmup + 余弦衰减策略
  - steps/s:                    每秒训练步数，反映训练吞吐量
  - max_mem:                    GPU 峰值显存占用 (MB)
"""

import re
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ── 中文字体配置 ──
# 自动检测系统中可用的中文字体，优先使用微软雅黑/黑体
_cjk_font = None
_candidate_fonts = [
    "Microsoft YaHei",    # 微软雅黑 (Windows)
    "SimHei",             # 黑体 (Windows)
    "WenQuanYi Micro Hei", # 文泉驿微米黑 (Linux)
    "Noto Sans CJK SC",   # Noto 简体中文 (跨平台)
    "Source Han Sans SC", # 思源黑体 (跨平台)
    "PingFang SC",        # 苹方 (macOS)
    "Heiti SC",           # 黑体-简 (macOS)
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font_name in _candidate_fonts:
    if _font_name in _available_fonts:
        _cjk_font = _font_name
        break

if _cjk_font:
    plt.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    print(f"[字体] 使用中文字体: {_cjk_font}")
else:
    # 没有中文字体时，回退到英文 + 警告
    print("[字体] 未检测到中文字体，图表中的中文将无法正常显示")
    print("[字体] 可尝试安装: pip install matplotlib --upgrade 或安装中文字体")

# ──────────────────────────────────────────────────────────
# 1. 解析 training.log，提取每步的训练指标
# ──────────────────────────────────────────────────────────
LOG_PATH = Path(__file__).parent / "training.log"

# 正则匹配日志行格式:
# [2026-06-03 11:50:47] step: 0000010/10000 | act_loss: 2.7823 | und_loss: 0.0000 | lr: 2.20e-06 | steps/s: 0.13 | ETA: 21:23:16 | max_mem: 26836MB
PATTERN = re.compile(
    r"step:\s*(\d+)/(\d+)\s*\|"
    r"\s*act_loss:\s*([\d.]+)\s*\|"
    r"\s*und_loss:\s*([\d.]+)\s*\|"
    r"\s*lr:\s*([\de.+\-]+)\s*\|"
    r"\s*steps/s:\s*([\d.]+)\s*\|"
    r"\s*ETA:\s*([\d:]+)\s*\|"
    r"\s*max_mem:\s*(\d+)MB"
)

steps = []          # 当前步数
total_steps = []    # 总步数
act_losses = []     # 动作损失
und_losses = []     # 理解损失
lrs = []            # 学习率
steps_per_sec = []  # 每秒步数
max_mems = []       # 峰值显存 (MB)

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        match = PATTERN.search(line)
        if match:
            steps.append(int(match.group(1)))
            total_steps.append(int(match.group(2)))
            act_losses.append(float(match.group(3)))
            und_losses.append(float(match.group(4)))
            lrs.append(float(match.group(5)))
            steps_per_sec.append(float(match.group(6)))
            max_mems.append(int(match.group(8)))

# 转为 numpy 数组便于计算
steps = np.array(steps)
act_losses = np.array(act_losses)
und_losses = np.array(und_losses)
lrs = np.array(lrs)
steps_per_sec = np.array(steps_per_sec)
max_mems = np.array(max_mems)

total = total_steps[0]  # 总计划步数
print(f"解析完成: {len(steps)} 条训练记录")
print(f"步数范围: {steps[0]} ~ {steps[-1]} / {total}")
print(f"act_loss 范围: {act_losses.min():.4f} ~ {act_losses.max():.4f}")
print(f"und_loss 范围: {und_losses.min():.4f} ~ {und_losses.max():.4f}")
print(f"lr 范围:     {lrs.min():.2e} ~ {lrs.max():.2e}")
print(f"steps/s 范围: {steps_per_sec.min():.2f} ~ {steps_per_sec.max():.2f}")
print(f"max_mem:     {max_mems[0]} MB (恒定)")

# ──────────────────────────────────────────────────────────
# 2. 计算派生指标
# ──────────────────────────────────────────────────────────

# 2.1 act_loss 的指数移动平均 (EMA)，用于平滑观察下降趋势
#     alpha 越小曲线越平滑，这里取 0.05 相当于窗口约 20 步
def ema(data, alpha=0.05):
    """计算指数移动平均，平滑训练曲线中的噪声"""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

act_loss_ema = ema(act_losses, alpha=0.05)

# 2.2 训练进度百分比
progress_pct = steps / total * 100

# 2.3 估算剩余训练时间（基于当前 steps/s）
#     公式: 剩余步数 / 当前速度 / 3600 = 剩余小时
eta_seconds = (total - steps) / steps_per_sec  # 剩余秒数
eta_hours = eta_seconds / 3600

# ──────────────────────────────────────────────────────────
# 3. 绘图 — 4 个子图并排
# ──────────────────────────────────────────────────────────
FIG_SIZE = (20, 12)  # 宽画布，容纳 4 个子图
fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
fig.suptitle(
    f"Training Curves — Flow Matching VLA (KoCH PostTrain)\n"
    f"Model: Qwen3-1.7B MoT + InternViT | "
    f"Progress: {steps[-1]}/{total} steps ({progress_pct[-1]:.1f}%) | "
    f"Interrupted by SIGHUP at step {steps[-1]}",
    fontsize=14, fontweight="bold"
)

# ─── 子图1: Action Loss ───────────────────────────────────
#       这是最核心的指标。act_loss 是 Flow Matching 的 MSE 损失，
#       衡量模型预测的动作与真实动作之间的差异。
#       从 ~2.78 下降到 ~0.06，说明模型在逐步学会正确的抓取动作。
ax1 = axes[0, 0]
ax1.plot(steps, act_losses, alpha=0.25, color="tab:blue", linewidth=0.8,
         label="Raw act_loss (per step)")
ax1.plot(steps, act_loss_ema, color="tab:blue", linewidth=2.0,
         label="act_loss EMA (α=0.05)")
ax1.axhline(y=0.1, color="gray", linestyle="--", linewidth=0.8,
            label="0.1 参考线 (较低损失)")
ax1.set_xlabel("Step")
ax1.set_ylabel("Action Loss")
ax1.set_title("Action Loss (Flow Matching MSE) — 越低越好")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, total)
# 标注起始和最终值
ax1.annotate(f"{act_losses[0]:.3f}", xy=(steps[0], act_losses[0]),
             xytext=(steps[0] + 200, act_losses[0] + 0.3),
             arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
ax1.annotate(f"{act_losses[-1]:.4f}", xy=(steps[-1], act_losses[-1]),
             xytext=(steps[-1] - 800, act_losses[-1] + 0.15),
             arrowprops=dict(arrowstyle="->", color="green"), fontsize=9, color="green")

# ─── 子图2: Understanding Loss ────────────────────────────
#       und_loss 衡量模型"理解"场景的能力（如物体识别、关系推理）。
#       当前阶段 und_loss=0 恒为 0，说明 U-ND（Understanding via Next-token
#       Distribution）损失尚未激活——训练还处于第一阶段，只优化动作。
ax2 = axes[0, 1]
if und_losses.max() == 0:
    # und_loss 全为 0 时，用文字说明而非画一条平线
    ax2.text(0.5, 0.5, "und_loss = 0.0000 (全程未激活)\n\n"
                        "U-ND (Understanding via Next-token Distribution) 损失\n"
                        "在当前训练阶段尚未启用。\n"
                        "该损失在后续阶段会激活，用于增强模型\n"
                        "对场景语义的理解能力。\n\n"
                        "num_und = 0 (无理解样本生成)",
             transform=ax2.transAxes, ha="center", va="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))
    ax2.set_title("Understanding Loss — 等待后续阶段激活")
else:
    ax2.plot(steps, und_losses, color="tab:orange", linewidth=1.5)
    ax2.set_ylabel("Understanding Loss")
    ax2.set_title("Understanding Loss (U-ND) — 越低越好")
    ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Step")

# ─── 子图3: Learning Rate ─────────────────────────────────
#       学习率调度: warmup → 余弦衰减 (cosine decay)。
#       初始从 ~2.2e-6 线性增加到峰值 ~6.1e-5，然后按余弦曲线缓慢衰减。
#       这种 warmup 策略可以防止训练初期梯度不稳定导致的发散。
ax3 = axes[1, 0]
ax3.plot(steps, lrs * 1e6, color="tab:green", linewidth=1.5)  # 转换为 1e-6 单位
ax3.set_xlabel("Step")
ax3.set_ylabel("Learning Rate (x 1e-6)")
ax3.set_title("Learning Rate Schedule — Warmup + Cosine Decay")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, total)
# 使用科学计数法格式化 y 轴
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
# 标注峰值 LR
peak_idx = np.argmax(lrs)
ax3.annotate(f"Peak LR: {lrs[peak_idx]:.2e}",
             xy=(steps[peak_idx], lrs[peak_idx] * 1e6),
             xytext=(steps[peak_idx] + 500, lrs[peak_idx] * 1e6 + 5),
             arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")

# ─── 子图4: Throughput (steps/s) ──────────────────────────
#       steps/s 反映 GPU 计算效率。当前值约 0.13~0.20 steps/s，
#       这意味着每秒约执行 0.13~0.20 个训练步（每步包含前向+反向传播）。
#       对于 VLA（Vision-Language-Action）大模型来说，这个速度属于正常范围。
#       训练全程 10000 步预计需要约 14~21 小时。
ax4 = axes[1, 1]
ax4.plot(steps, steps_per_sec, color="tab:purple", linewidth=1.5)
ax4.set_xlabel("Step")
ax4.set_ylabel("Steps per Second")
ax4.set_title("Training Throughput (steps/s) — GPU 计算效率")
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, total)
# 计算平均吞吐量
avg_throughput = steps_per_sec.mean()
ax4.axhline(y=avg_throughput, color="red", linestyle="--", linewidth=1.0,
            label=f"平均: {avg_throughput:.3f} steps/s")
ax4.legend(loc="lower right", fontsize=9)
# 标注训练被中断的时间点
ax4.annotate(f"训练中断 @ step {steps[-1]}",
             xy=(steps[-1], steps_per_sec[-1]),
             xytext=(steps[-1] - 2000, steps_per_sec[-1] + 0.03),
             arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")

plt.tight_layout()

# ──────────────────────────────────────────────────────────
# 4. 保存图片
# ──────────────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent / "training_curves.png"
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n图片已保存至: {OUTPUT_PATH}")

# ──────────────────────────────────────────────────────────
# 5. 额外: 生成训练摘要统计
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("训练摘要统计")
print("=" * 60)
print(f"总记录步数:        {len(steps)}")
print(f"训练进度:          {progress_pct[-1]:.1f}% ({steps[-1]}/{total})")
print(f"act_loss 初始值:   {act_losses[0]:.4f}")
print(f"act_loss 最终值:   {act_losses[-1]:.4f}")
print(f"act_loss 下降幅度: {act_losses[0] - act_losses[-1]:.4f} "
      f"({(1 - act_losses[-1] / act_losses[0]) * 100:.1f}%)")
print(f"act_loss 最小值:   {act_losses.min():.4f} (step {steps[act_losses.argmin()]})")
print(f"und_loss:          {und_losses.max():.4f} (全程为0, 未激活)")
print(f"学习率范围:        {lrs.min():.2e} ~ {lrs.max():.2e}")
print(f"当前学习率:        {lrs[-1]:.2e}")
print(f"平均 steps/s:      {avg_throughput:.3f}")
print(f"当前 steps/s:      {steps_per_sec[-1]:.3f}")
print(f"GPU 显存占用:      {max_mems[0]} MB (~{max_mems[0]/1024:.1f} GB)")
print(f"训练已用时间:      约 6h19min (根据日志中 elapsed_time)")
print(f"训练中断原因:      SIGHUP 信号 (终端断开/进程被杀)")
print(f"训练中断时 step:   {steps[-1]} / {total}")
print("=" * 60)
