"""
推理验证脚本：对 val.jsonl 中的图片运行模型推理，画出预测框与GT框，保存对比图
用法：python eval_vis.py
"""

import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
ADAPTER = BASE_DIR / "output/qwen3.5-9b-graspdet/v1-20260318-114022/checkpoint-100"
VAL_JSONL = BASE_DIR / "val.jsonl"
OUT_DIR = BASE_DIR / "output/vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 加载模型 ────────────────────────────────────────────────────────────────
print("Loading model...")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from swift import TransformersEngine, RequestConfig, InferRequest

model_id = "Qwen/Qwen3.5-9B"
engine = TransformersEngine(model_id, adapters=[str(ADAPTER)])
print("Model loaded.")


# ── 解析工具 ─────────────────────────────────────────────────────────────────
def parse_box(text: str):
    """从模型输出中提取 JSON，返回 box dict 或 None"""
    # 去掉 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 提取第一个 JSON 对象或数组
    for pattern in (r"\{.*?\}", r"\[.*?\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
                if isinstance(obj, list):
                    obj = obj[0]
                if not obj.get("failed", False):
                    return obj
            except Exception:
                pass
    return None


# ── 绘图工具 ─────────────────────────────────────────────────────────────────
COLOR_PRED = (80, 80, 255)   # BGR 红：预测框
COLOR_GT   = (80, 200, 80)   # BGR 绿：GT框
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(img, img_w, img_h, box, color, label=""):
    cx = box["box_center_x"] * img_w
    cy = box["box_center_y"] * img_h
    bw = box["box_width"] * img_w
    bh = box["box_height"] * img_h
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1 + 3, y1 - 6), FONT, 0.5, color, 1, cv2.LINE_AA)


# ── 提取用户指令 ──────────────────────────────────────────────────────────────
def extract_instruction(user_content: str) -> str:
    m = re.search(r'用户指令[：:]\s*["\"](.+?)["\"]', user_content)
    return m.group(1) if m else "?"


# ── 主循环 ────────────────────────────────────────────────────────────────────
samples = [json.loads(l) for l in open(VAL_JSONL)]
print(f"Total val samples: {len(samples)}")

# 每张图只推一条（取每张图第一个出现的样本），避免重复
seen_images = {}
for s in samples:
    img_path = s["images"][0]
    if img_path not in seen_images:
        seen_images[img_path] = s

print(f"Unique images: {len(seen_images)}, running inference...")

req_config = RequestConfig(max_tokens=8192, temperature=0)

total = len(seen_images)
results = []
for idx, (img_path, sample) in enumerate(seen_images.items(), 1):
    print(f"[{idx}/{total}] {Path(img_path).name} ...", flush=True)
    user_content = sample["messages"][0]["content"]
    gt_content = sample["messages"][1]["content"]
    instruction = extract_instruction(user_content)

    # 推理
    infer_req = InferRequest(
        messages=[{"role": "user", "content": user_content}],
        images=[str(BASE_DIR / img_path)],
    )
    resp = engine.infer([infer_req], request_config=req_config)
    pred_text = resp[0].choices[0].message.content

    pred_box = parse_box(pred_text)
    gt_box = parse_box(gt_content)

    status = "OK" if pred_box else "FAIL"
    print(f"  -> {status}  pred={pred_box}  gt={gt_box}", flush=True)

    if idx <= 3:
        print(f"  raw: {repr(pred_text[:300])}", flush=True)

    results.append(
        {
            "img_path": img_path,
            "instruction": instruction,
            "pred_text": pred_text,
            "pred_box": pred_box,
            "gt_box": gt_box,
        }
    )

print(f"Inference done. Drawing boxes...")

n_ok, n_fail = 0, 0
for r in results:
    img_full = str(BASE_DIR / r["img_path"])
    img = cv2.imread(img_full)
    h, w = img.shape[:2]

    gt_box   = r["gt_box"]
    pred_box = r["pred_box"]

    if gt_box:
        draw_box(img, w, h, gt_box, COLOR_GT, f"GT:{gt_box.get('class_name','')}")
    if pred_box:
        draw_box(img, w, h, pred_box, COLOR_PRED, f"Pred:{pred_box.get('class_name','')}")
        n_ok += 1
    else:
        cv2.putText(img, "PARSE FAILED", (10, 30), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        n_fail += 1

    # 底部黑条 + 指令文字
    cv2.rectangle(img, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(img, r["instruction"], (5, h - 10), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 用指令作为文件名（截断+清理非法字符）
    safe_instr = re.sub(r'[\\/:*?"<>|]', "_", r["instruction"])[:60]
    img_stem = Path(r["img_path"]).stem
    out_path = OUT_DIR / f"{img_stem}__{safe_instr}_vis.jpg"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        out_path.write_bytes(buf.tobytes())

print(f"\n=== 结果 ===")
print(f"成功解析框: {n_ok}/{len(results)}")
print(f"解析失败:   {n_fail}/{len(results)}")
print(f"输出目录:   {OUT_DIR}")
print(f"红框=预测，绿框=GT")
