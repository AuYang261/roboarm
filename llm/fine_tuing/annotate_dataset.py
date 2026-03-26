#!/usr/bin/env python3
"""
用 YOLO OBB 模型标注 original/ 下的图片，生成训练数据，并与原有 train.jsonl 合并。

功能：
  1. 对 original/ 中每张图片运行 YOLO OBB 推理
  2. 将旋转框(xywhr)转换为轴对齐归一化 bbox
  3. 对每张图片应用多种自然语言指令（颜色/位置/距离），生成训练样本
  4. 在 annotated/ 下输出含标注框的可视化图片供人工检查
  5. 与原有 train.jsonl 合并，输出 merged_train.jsonl

用法：
    python annotate_dataset.py
    python annotate_dataset.py --conf 0.7 --input-jsonl train.jsonl --output merged_train.jsonl
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

# YOLO 类名 → 中文类名（用于 class_name 字段）
CLASS_NAME_ZH = {
    "blue_block": "蓝色积木",
    "red_block": "红色积木",
    "yellow_block": "黄色积木",
}

# 各类别的可视化颜色 (BGR)
CLASS_COLOR = {
    "blue_block": (200, 100, 0),
    "red_block": (0, 50, 220),
    "yellow_block": (0, 200, 220),
}
DEFAULT_COLOR = (0, 200, 0)

# 固定系统 prompt（与 train.jsonl 保持一致）
SYSTEM_PROMPT = """你是一个智能机械臂的视觉控制中枢。你的任务是根据用户的自然语言指令，分析图片中的场景，并计算出机械臂需要抓取的目标精确位置，用方框表示。

确保方框边界能完全框住该物体，特别是长条形的物体（如长条积木），要体现出其长度。

坐标系定义：
- 图片左上角为原点 (0, 0)。
- 向右为 x 轴 (0 -> 1)。
- 向下为 y 轴 (0 -> 1)。

用户指令可能包含：
- 颜色/形状特征（如："抓红色的积木"、"抓圆形的棋子"）
- 相对距离描述（如："抓离机械臂最近的"，通常机械臂底座位于图片下方(y坐标较大)，需根据图片内容判断）
- 空间位置描述（如："抓最右边的"、"抓左上角的"、"抓中间的"）。
- 用户的输入是通过语音转文字得到的，可能存在一定的语义不清晰或歧义，请结合图片内容进行合理推断

请按照以下步骤进行推理（Thinking Process），并在最后给出最终坐标：
1. **分析指令**：提取用户想要抓取的目标物体的关键特征（颜色、类别）和限制条件（位置、距离）。
2. **扫描图像**：在心中列出所有符合特征的候选物体及其大致坐标。
3. **逻辑筛选**：根据限制条件（如"最右边"即x坐标最大，"最近"即距离特定点欧氏距离最小）从候选中选出唯一目标，如果有多个符合条件的物体，选择任意一个即可。
4. **确定坐标**：给出能完全框住该目标的精确边界框。

输出格式要求：
请输出一个纯净的JSON数组，不要包含任何Markdown格式（如 ```json）、代码块标记或额外的解释性文字。

数组中的每个对象应包含以下字段：
- "id"：整数，唯一标识符。
- "failed"：布尔值，表示是否未找到符合条件的目标物体。如果未找到，设置为 true，其他字段置为任意值。否则，设置为 false，并填写以下字段：
- "class_name"：字符串，物体的具体类别，必须包含颜色和形状信息（例如："红色方形积木"、"蓝色长条积木"、"黑色螺丝"）。
- "box_center_x"：物体边界框中心点的x坐标，范围0-1之间，保留3位小数。
- "box_center_y"：物体边界框中心点的y坐标，范围0-1之间，保留3位小数。
- "box_width"：物体边界框的宽度，范围0-1之间，保留3位小数。
- "box_height"：物体边界框的高度，范围0-1之间，保留3位小数。

示例 1：
用户指令："抓取最右边的蓝色积木"
模型回复：
{
    "id": 1,
    "failed": false,
    "class_name": "蓝色积木",
    "box_center_x": 0.821,
    "box_center_y": 0.131,
    "box_width": 0.142,
    "box_height": 0.508
}

示例 2：
用户指令："抓红色的方块"
模型回复：
{
    "id": 2,
    "failed": false,
    "class_name": "红色积木",
    "box_center_x": 0.385,
    "box_center_y": 0.445,
    "box_width": 0.105,
    "box_height": 0.108
}

示例 3：
用户指令："抓最左边的黄色积木"
模型回复：
{
    "failed": true,
}

现在，请根据图片和以下指令执行任务：
用户指令："{instruction}\""""


def build_user_content(instruction: str) -> str:
    return "<image>\n" + SYSTEM_PROMPT.replace("{instruction}", instruction)


# ──────────────────────────────────────────────
# 检测结果数据结构
# ──────────────────────────────────────────────

@dataclass
class Detection:
    """单个 YOLO OBB 检测结果（像素坐标 + 归一化 bbox）。"""
    px_cx: float   # 像素中心 x
    px_cy: float   # 像素中心 y
    px_w: float    # 像素宽
    px_h: float    # 像素高
    angle_rad: float  # 旋转角（弧度）
    score: float
    class_name: str   # 英文，如 "blue_block"

    # 归一化轴对齐 bbox（由 OBB 外接矩形计算得到）
    norm_cx: float = 0.0
    norm_cy: float = 0.0
    norm_w: float = 0.0
    norm_h: float = 0.0

    @property
    def class_name_zh(self) -> str:
        return CLASS_NAME_ZH.get(self.class_name, self.class_name)


# ──────────────────────────────────────────────
# OBB → 轴对齐归一化 bbox
# ─────────────���────────────────────────────────

def obb_to_aabb_normalized(
    cx: float, cy: float, w: float, h: float, r_rad: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """将 OBB (xywhr，像素) 转换为轴对齐归一化 bbox (cx, cy, w, h)。"""
    pts = cv2.boxPoints(((cx, cy), (w, h), float(np.rad2deg(r_rad))))
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    # 裁剪到图像范围内
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(img_w), x_max)
    y_max = min(float(img_h), y_max)
    norm_cx = float((x_min + x_max) / 2) / img_w
    norm_cy = float((y_min + y_max) / 2) / img_h
    norm_w = float(x_max - x_min) / img_w
    norm_h = float(y_max - y_min) / img_h
    return round(norm_cx, 3), round(norm_cy, 3), round(norm_w, 3), round(norm_h, 3)


# ──────────────────────────────────────────────
# YOLO 推理
# ──────────────────────────────────────────────

def detect_image(
    model: YOLO, image_path: Path, conf_thres: float = 0.6, iou_thres: float = 0.45
) -> tuple[list[Detection], tuple[int, int]]:
    """对单张图片运行 YOLO OBB 推理，返回检测列表和图像尺寸 (w, h)。"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    img_h, img_w = img.shape[:2]

    results = model(img, conf=conf_thres, iou=iou_thres, verbose=False)[0]
    obb = results.obb
    if obb is None or len(obb) == 0:
        return [], (img_w, img_h)

    xywhr = obb.xywhr.cpu().numpy()   # shape (N, 5)
    scores = obb.conf.cpu().numpy()
    class_ids = obb.cls.cpu().numpy().astype(int)

    detections = []
    for (cx, cy, w, h, r), score, cls_id in zip(xywhr, scores, class_ids):
        cls_name = model.names[cls_id]
        norm_cx, norm_cy, norm_w, norm_h = obb_to_aabb_normalized(
            cx, cy, w, h, r, img_w, img_h
        )
        det = Detection(
            px_cx=float(cx), px_cy=float(cy),
            px_w=float(w), px_h=float(h),
            angle_rad=float(r),
            score=float(score),
            class_name=cls_name,
            norm_cx=norm_cx, norm_cy=norm_cy,
            norm_w=norm_w, norm_h=norm_h,
        )
        detections.append(det)
    return detections, (img_w, img_h)


# ──────────────────────────────────────────────
# 自然语言指令函数
#
# 每个函数接收检测列表，返回 (instruction_text, target_detection) 或 None。
# 若条件不满足（如图中没有蓝色积木），返回 None，该指令不生成训练样本。
# ──────────────────────────────────────────────

def get_blue_block_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取蓝色积木。"""
    candidates = [d for d in detections if d.class_name == "blue_block"]
    if not candidates:
        return None
    target = random.choice(candidates)
    instruction = random.choice(["抓取蓝色积木", "抓蓝色的积木", "拿蓝色积木"])
    return instruction, target


def get_red_block_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取红色积木。"""
    candidates = [d for d in detections if d.class_name == "red_block"]
    if not candidates:
        return None
    target = random.choice(candidates)
    instruction = random.choice(["抓取红色积木", "抓红色的积木", "拿红色积木"])
    return instruction, target


def get_yellow_block_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取黄色积木。"""
    candidates = [d for d in detections if d.class_name == "yellow_block"]
    if not candidates:
        return None
    target = random.choice(candidates)
    instruction = random.choice(["抓取黄色积木", "抓黄色的积木", "拿黄色积木"])
    return instruction, target


def get_nearest_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取离机械臂最近的积木（y坐标最大，即图片最下方）。"""
    if not detections:
        return None
    target = max(detections, key=lambda d: d.norm_cy)
    instruction = random.choice([
        "抓取最近的积木", "抓离机械臂最近的积木", "抓最近的那个"
    ])
    return instruction, target


def get_farthest_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取离机械臂最远的积木（y坐标最小，即图片最上方）。"""
    if not detections:
        return None
    target = min(detections, key=lambda d: d.norm_cy)
    instruction = random.choice([
        "抓取最远的积木", "抓离机械臂最远的积木", "抓最远的那个"
    ])
    return instruction, target


def get_leftmost_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取最左边的积木。"""
    if not detections:
        return None
    target = min(detections, key=lambda d: d.norm_cx)
    instruction = random.choice(["抓取最左边的积木", "抓最左边的那个", "拿最左侧的积木"])
    return instruction, target


def get_rightmost_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取最右边的积木。"""
    if not detections:
        return None
    target = max(detections, key=lambda d: d.norm_cx)
    instruction = random.choice(["抓取最右边的积木", "抓最右边的那个", "拿最右侧的积木"])
    return instruction, target


def get_nearest_blue_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取最近的蓝色积木。"""
    candidates = [d for d in detections if d.class_name == "blue_block"]
    if not candidates:
        return None
    target = max(candidates, key=lambda d: d.norm_cy)
    instruction = random.choice(["抓取最近的蓝色积木", "抓离机械臂最近的蓝色积木"])
    return instruction, target


def get_nearest_red_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取最近的红色积木。"""
    candidates = [d for d in detections if d.class_name == "red_block"]
    if not candidates:
        return None
    target = max(candidates, key=lambda d: d.norm_cy)
    instruction = random.choice(["抓取最近的红色积木", "抓离机械臂最近的红色积木"])
    return instruction, target


def get_rightmost_blue_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取最右边的蓝色积木。"""
    candidates = [d for d in detections if d.class_name == "blue_block"]
    if len(candidates) < 2:
        return None  # 只有一个时无需区分方向，跳过
    target = max(candidates, key=lambda d: d.norm_cx)
    instruction = random.choice(["抓取最右边的蓝色积木", "拿最右侧的蓝色积木"])
    return instruction, target


def get_leftmost_red_instruction(
    detections: list[Detection],
) -> tuple[str, Detection] | None:
    """抓取最左边的红色积木。"""
    candidates = [d for d in detections if d.class_name == "red_block"]
    if len(candidates) < 2:
        return None
    target = min(candidates, key=lambda d: d.norm_cx)
    instruction = random.choice(["抓取最左边的红色积木", "拿最左侧的红色积木"])
    return instruction, target


# 所有指令函数列表，脚本会依次对每张图片尝试所有指令
INSTRUCTION_FUNCTIONS = [
    get_blue_block_instruction,
    get_red_block_instruction,
    get_yellow_block_instruction,
    get_nearest_instruction,
    get_farthest_instruction,
    get_leftmost_instruction,
    get_rightmost_instruction,
    get_nearest_blue_instruction,
    get_nearest_red_instruction,
    get_rightmost_blue_instruction,
    get_leftmost_red_instruction,
]


# ──────────────────────────────────────────────
# 训练样本生成
# ──────────────────────────────────────────────

def make_training_sample(
    image_rel_path: str, instruction: str, target: Detection
) -> dict:
    """生成单条 train.jsonl 格式的训练样本。"""
    answer = {
        "id": 1,
        "failed": False,
        "class_name": target.class_name_zh,
        "box_center_x": target.norm_cx,
        "box_center_y": target.norm_cy,
        "box_width": target.norm_w,
        "box_height": target.norm_h,
    }
    return {
        "messages": [
            {
                "role": "user",
                "content": build_user_content(instruction),
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(", ", ": ")),
            },
        ],
        "images": [image_rel_path],
    }


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def draw_annotated_image(
    image_path: Path,
    detections: list[Detection],
    out_path: Path,
) -> None:
    """绘制含 OBB 旋转框 + 轴对齐外接框的标注图片，保存到 out_path。"""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    img_h, img_w = img.shape[:2]

    for det in detections:
        color = CLASS_COLOR.get(det.class_name, DEFAULT_COLOR)

        # 1. 画 YOLO OBB 旋转框（实线）
        obb_pts = cv2.boxPoints(
            ((det.px_cx, det.px_cy), (det.px_w, det.px_h), float(np.rad2deg(det.angle_rad)))
        ).astype(np.int32)
        cv2.drawContours(img, [obb_pts], 0, color, 2)

        # 2. 画轴对齐外接框（虚线效果：用矩形）
        x1 = int((det.norm_cx - det.norm_w / 2) * img_w)
        y1 = int((det.norm_cy - det.norm_h / 2) * img_h)
        x2 = int((det.norm_cx + det.norm_w / 2) * img_w)
        y2 = int((det.norm_cy + det.norm_h / 2) * img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # 3. 标签（类名 + 置信度）
        label = f"{det.class_name_zh} {det.score:.2f}"
        cv2.putText(
            img, label,
            (int(det.px_cx - det.px_w / 2), max(0, int(det.px_cy - det.px_h / 2) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 imencode 避免路径含中文问题
    ok, buf = cv2.imencode(out_path.suffix or ".png", img)
    if ok:
        out_path.write_bytes(buf.tobytes())


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO 标注 + 训练数据生成")
    parser.add_argument("--original-dir", default="original", help="原始图片目录")
    parser.add_argument("--model-path", default="yolo/best.pt", help="YOLO 模型路径")
    parser.add_argument("--conf", type=float, default=0.6, help="YOLO 置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IOU 阈值")
    parser.add_argument("--input-jsonl", default="train.jsonl", help="原有训练数据")
    parser.add_argument("--output", default="merged_train.jsonl", help="合并后输出文件")
    parser.add_argument("--annotated-dir", default="annotated", help="标注可视化图片输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（指令措辞随机选择）")
    args = parser.parse_args()

    random.seed(args.seed)

    original_dir = Path(args.original_dir)
    annotated_dir = Path(args.annotated_dir)
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"错误：模型文件不存在 {model_path}", file=sys.stderr)
        sys.exit(1)
    if not original_dir.exists():
        print(f"错误：图片目录不存在 {original_dir}", file=sys.stderr)
        sys.exit(1)

    # 找所有图片
    image_paths = sorted(
        p for p in original_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    print(f"找到 {len(image_paths)} 张图片。")

    # 加载 YOLO 模型
    print(f"加载 YOLO 模型: {model_path}")
    model = YOLO(str(model_path))
    print(f"模型类别: {model.names}")

    # 处理每张图片
    new_samples: list[dict] = []
    stats = {"images": 0, "no_det": 0, "samples": 0}

    for img_path in image_paths:
        print(f"  处理: {img_path.name}", end="")
        try:
            detections, (img_w, img_h) = detect_image(
                model, img_path, conf_thres=args.conf, iou_thres=args.iou
            )
        except Exception as e:
            print(f" [错误: {e}]")
            continue

        stats["images"] += 1

        if not detections:
            print(f" → 无检测结果，跳过")
            stats["no_det"] += 1
            continue

        print(f" → 检测到 {len(detections)} 个目标", end="")

        # 可视化标注图
        out_annotated = annotated_dir / img_path.name
        draw_annotated_image(img_path, detections, out_annotated)

        # 相对路径（用于 jsonl 中的 images 字段）
        img_rel = str(original_dir / img_path.name)

        # 依次尝试所有指令函数
        count_for_this_image = 0
        for fn in INSTRUCTION_FUNCTIONS:
            result = fn(detections)
            if result is None:
                continue
            instruction, target = result
            sample = make_training_sample(img_rel, instruction, target)
            new_samples.append(sample)
            count_for_this_image += 1

        print(f"，生成 {count_for_this_image} 条训练样本")
        stats["samples"] += count_for_this_image

    # 读取原有 train.jsonl
    existing_samples: list[dict] = []
    input_path = Path(args.input_jsonl)
    if input_path.exists():
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_samples.append(json.loads(line))
        print(f"\n原有训练数据: {len(existing_samples)} 条")
    else:
        print(f"\n未找到原有训练数据文件 {input_path}，仅使用新生成数据。")

    # 合并并写出
    all_samples = existing_samples + new_samples
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n=== 完成 ===")
    print(f"  处理图片: {stats['images']} 张（{stats['no_det']} 张无检测结果）")
    print(f"  新生成样本: {stats['samples']} 条")
    print(f"  原有样本: {len(existing_samples)} 条")
    print(f"  合并输出: {output_path}（共 {len(all_samples)} 条）")
    print(f"  标注图片: {annotated_dir}/")
    print(f"\n注意：新样本的 thinking_process 字段为空，")
    print(f"请随后运行 generate_thinking.py --input {output_path} 补全推理过程。")


if __name__ == "__main__":
    main()
