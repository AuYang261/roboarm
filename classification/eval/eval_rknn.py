"""Evaluate RKNN model on the test split and save predictions.

Usage (from repo root):
    python eval_rknn.py --weights ./models/best-screw.rknn --data ./dataset
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from rknn.api import RKNN
except ImportError:
    print("Error: RKNN.api not found. Please install RKNN Toolkit Lite2.")
    sys.exit(1)


def read_classes(classes_path: Path) -> List[str]:
    """Read class names from classes.txt file."""
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    names: List[str] = []
    with classes_path.open("r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            if ":" in ln:
                _, name = ln.split(":", 1)
                names.append(name.strip())
            else:
                names.append(ln)
    return names


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_rknn_model(model_path: Path) -> RKNN:
    """Load RKNN model and initialize runtime."""
    rknn_lite = RKNN()
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f'Load RKNN model failed with error code {ret}')
    print('--> Init runtime environment')
    # Use NPU core 0; adjust if needed
    ret = rknn_lite.init_runtime(core_mask=RKNN.NPU_CORE_0)
    if ret != 0:
        raise RuntimeError(f'Init runtime environment failed with error code {ret}')
    return rknn_lite


def preprocess_image(image: np.ndarray, input_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess image for RKNN model: BGR to RGB, resize, add batch dimension."""
    # Convert BGR to RGB (assuming input image is BGR from cv2.imread)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    # Add batch dimension: shape becomes (1, height, width, 3)
    img_batch = np.expand_dims(img_resized, axis=0)
    # Note: normalization is handled by model conversion config; keep uint8.
    return img_batch


def postprocess_rknn_output(output: np.ndarray, conf_thres: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess RKNN output assuming shape [1, N, 7] where each row is
    [cx, cy, w, h, angle_rad, conf, cls].

    Returns:
        boxes: [M, 5] array of (cx, cy, w, h, angle_rad)
        scores: [M] array of confidence scores
        class_ids: [M] array of integer class IDs
    """
    # output shape: (1, N, 7)
    detections = output[0]  # [N, 7]
    # Filter by confidence
    mask = detections[:, 5] >= conf_thres
    detections = detections[mask]
    if len(detections) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes = detections[:, :5]      # cx, cy, w, h, angle
    scores = detections[:, 5]      # confidence
    class_ids = detections[:, 6].astype(int)  # class id
    return boxes, scores, class_ids


def obb_to_polygon(cx: float, cy: float, w: float, h: float, angle: float) -> List[Tuple[float, float]]:
    """
    Convert oriented bounding box (center format) to 4-corner polygon.
    Coordinates are normalized [0,1] relative to image dimensions.
    Angle in radians, counter‑clockwise.
    Returns list of four (x, y) tuples.
    """
    # Calculate half dimensions
    half_w = w / 2.0
    half_h = h / 2.0

    # Corners relative to center before rotation
    corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ])

    # Rotation matrix
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Rotate corners and translate to center
    rotated = np.dot(corners, rot.T)
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    # Convert to list of tuples
    polygon = [(float(x), float(y)) for x, y in rotated]
    return polygon


def polygon_to_aabb(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute axis‑aligned bounding box from polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def iou_aabb(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU for two axis‑aligned bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def read_gt_labels(label_path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Read ground‑truth labels in polygon format:
    each line: cls x1 y1 x2 y2 x3 y3 x4 y4
    Returns list of (class_id, polygon) where polygon is list of 4 (x, y) tuples.
    """
    items = []
    if not label_path.exists():
        return items
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                continue  # skip malformed lines
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            polygon = [
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7])
            ]
            items.append((cls, polygon))
    return items


def evaluate_image(
    gt: List[Tuple[int, List[Tuple[float, float]]]],
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_class_ids: np.ndarray,
    iou_thresh: float = 0.5
) -> Tuple[float, int, int, int]:
    """
    Evaluate predictions against ground truth.
    Converts predicted OBB to polygons, then to AABB for IoU calculation.

    Returns:
        mean_iou: average IoU of true positives
        tp: number of true positives
        fp: number of false positives
        fn: number of false negatives
    """
    if len(pred_boxes) == 0:
        # No predictions: all GT are false negatives
        return 0.0, 0, 0, len(gt)

    # Convert predictions to polygons and then to AABBs
    pred_polygons = []
    for box in pred_boxes:
        cx, cy, w, h, angle = box
        poly = obb_to_polygon(cx, cy, w, h, angle)
        pred_polygons.append(poly)

    gt_aabbs = [polygon_to_aabb(poly) for _, poly in gt]
    pred_aabbs = [polygon_to_aabb(poly) for poly in pred_polygons]

    used_gt = [False] * len(gt)
    ious = []
    tp = 0
    fp = 0

    # Match predictions to ground truth
    for pred_idx, (cls_pred, pred_aabb) in enumerate(zip(pred_class_ids, pred_aabbs)):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, (cls_gt, gt_aabb) in enumerate(zip([c for c, _ in gt], gt_aabbs)):
            if used_gt[gt_idx] or cls_gt != cls_pred:
                continue
            iou = iou_aabb(pred_aabb, gt_aabb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            used_gt[best_gt_idx] = True
            tp += 1
            ious.append(best_iou)
        else:
            fp += 1

    fn = used_gt.count(False)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return mean_iou, tp, fp, fn


def draw_annotations(
    image_path: Path,
    out_image_path: Path,
    gt: List[Tuple[int, List[Tuple[float, float]]]],
    pred_boxes: np.ndarray,
    pred_class_ids: np.ndarray,
    class_names: List[str]
) -> None:
    """Draw ground truth (green) and predictions (red) on image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Draw ground truth polygons in green
    for cls, poly in gt:
        pts = [(int(x * w), int(y * h)) for x, y in poly]
        draw.polygon(pts, outline=(0, 255, 0), width=2)

    # Draw predicted polygons in red
    for box, cls_id in zip(pred_boxes, pred_class_ids):
        cx, cy, w_box, h_box, angle = box
        poly = obb_to_polygon(cx, cy, w_box, h_box, angle)
        pts = [(int(x * w), int(y * h)) for x, y in poly]
        draw.polygon(pts, outline=(255, 0, 0), width=2)
        # Optionally add class label
        if cls_id < len(class_names):
            label = class_names[cls_id]
            draw.text((pts[0][0], pts[0][1] - 10), label, fill=(255, 0, 0))

    img.save(out_image_path)


def save_predictions(
    out_labels_dir: Path,
    stem: str,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_class_ids: np.ndarray
) -> None:
    """Save predictions in YOLO OBB format: cls cx cy w h angle conf."""
    label_file = out_labels_dir / f"{stem}.txt"
    ensure_dir(label_file.parent)
    with label_file.open("w", encoding="utf-8") as f:
        for box, score, cls_id in zip(pred_boxes, pred_scores, pred_class_ids):
            cx, cy, w, h, angle = box
            # cls_id, cx, cy, w, h, angle, confidence
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f} {score:.4f}\n")


def save_result_txt(
    result_path: Path,
    classes: List[str],
    total_images: int,
    total_tp: int,
    total_fp: int,
    total_fn: int,
    iou_list: List[float],
    iou_thresh: float = 0.5,
    per_image_stats: List[str] = None
) -> None:
    """Write evaluation summary to result.txt."""
    ensure_dir(result_path.parent)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0

    with result_path.open("w", encoding="utf-8") as f:
        f.write("RKNN Model Evaluation on Test Split\n")
        f.write(f"Classes: {', '.join(classes)}\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"True Positives (TP): {total_tp}\n")
        f.write(f"False Positives (FP): {total_fp}\n")
        f.write(f"False Negatives (FN): {total_fn}\n")
        f.write(f"Precision @ IoU={iou_thresh}: {precision:.4f}\n")
        f.write(f"Recall @ IoU={iou_thresh}: {recall:.4f}\n")
        f.write(f"Mean IoU (AABB approximation): {mean_iou:.4f}\n")
        if per_image_stats:
            f.write("\nPer-image statistics:\n")
            for line in per_image_stats:
                f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RKNN model on test images")
    parser.add_argument("--weights", default="./models/best-screw.rknn", help="Path to RKNN model file")
    parser.add_argument("--data", default="./dataset", help="Dataset root containing images/ and labels/")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (square)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (note: RKNN may have built‑in NMS)")
    parser.add_argument("--eval_iou", type=float, default=0.5, help="IoU threshold for evaluation matching")
    parser.add_argument("--out", default="./eval_rknn", help="Output directory")
    parser.add_argument("--classes", default="./classes.txt", help="Path to classes.txt")
    args = parser.parse_args()

    data_root = Path(args.data)
    classes = read_classes(Path(args.classes))

    out_root = Path(args.out)
    out_images = out_root / "images" / "test"
    out_labels = out_root / "labels" / "test"
    test_images_dir = data_root / "images" / "test"
    test_labels_dir = data_root / "labels" / "test"
    result_txt = out_root / "result.txt"

    ensure_dir(out_root)
    ensure_dir(out_images)
    ensure_dir(out_labels)

    # Load RKNN model
    print(f"Loading RKNN model from {args.weights}")
    model = load_rknn_model(Path(args.weights))

    # Collect evaluation statistics
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_list = []
    per_image_lines = []

    # Iterate over test images
    image_paths = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")))
    print(f"Found {len(image_paths)} test images")

    for img_path in image_paths:
        stem = img_path.stem
        gt_path = test_labels_dir / f"{stem}.txt"
        gt_items = read_gt_labels(gt_path)

        # Load and preprocess image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Could not read {img_path}, skipping")
            continue

        input_tensor = preprocess_image(img_bgr, (args.imgsz, args.imgsz))

        # Inference
        outputs = model.inference(inputs=[input_tensor])
        if not outputs or len(outputs) == 0:
            print(f"Warning: No output from model for {stem}")
            pred_boxes = np.array([])
            pred_scores = np.array([])
            pred_class_ids = np.array([])
        else:
            # Assume first output is detection tensor
            raw_output = outputs[0]  # shape (1, N, 7)
            pred_boxes, pred_scores, pred_class_ids = postprocess_rknn_output(
                raw_output, conf_thres=args.conf
            )

        # Evaluate
        mean_iou, tp, fp, fn = evaluate_image(
            gt_items, pred_boxes, pred_scores, pred_class_ids, iou_thresh=args.eval_iou
        )
        total_images += 1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        iou_list.append(mean_iou)

        # Save predictions
        save_predictions(out_labels, stem, pred_boxes, pred_scores, pred_class_ids)

        # Draw annotated image
        out_img_path = out_images / f"{stem}.jpg"
        draw_annotations(
            img_path, out_img_path, gt_items, pred_boxes, pred_class_ids, classes
        )

        print(f"{stem}: TP={tp}, FP={fp}, FN={fn}, IoU={mean_iou:.3f}")
        per_image_lines.append(f"{stem}: TP={tp}, FP={fp}, FN={fn}, IoU={mean_iou:.3f}, detections={len(pred_boxes)}")

    # Save overall results
    save_result_txt(
        result_txt, classes, total_images, total_tp, total_fp, total_fn, iou_list, args.eval_iou, per_image_lines
    )

    print(f"\nEvaluation complete. Results saved to {result_txt}")
    print(f"  Precision: {total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0:.4f}")
    print(f"  Recall:    {total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0:.4f}")

    # Cleanup
    model.release()


if __name__ == "__main__":
    main()