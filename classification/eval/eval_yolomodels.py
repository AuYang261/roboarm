"""Evaluate best-screw.pt on the test split and save predictions.

Usage (from repo root):
    python eval_best_screw.py --weights ./best-screw.pt --data ./dataset
"""

import argparse
from pathlib import Path
from typing import List

import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

def read_classes(classes_path: Path) -> List[str]:
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
    path.mkdir(parents=True, exist_ok=True)


def format_metric(metrics, attr: str, default: float = 0.0) -> float:
    try:
        return getattr(metrics.box, attr)
    except Exception:
        return default


def save_result_txt(result_path: Path, classes: List[str], metrics, per_image_stats: List[str], metrics_error: str | None = None) -> None:
    ensure_dir(result_path.parent)
    precision = format_metric(metrics, "p") if metrics else 0.0
    recall = format_metric(metrics, "r") if metrics else 0.0
    # 根据 precision 求 precision 的 平均值 和 recall 的 平均值 类型是个 array([...])
    prec_mean = np.average(precision)
    prec_recall = np.average(recall)
    
    map50 = float(format_metric(metrics, "map50")) if metrics else 0.0
    map50_95 = float(format_metric(metrics, "map")) if metrics else 0.0

    with result_path.open("w", encoding="utf-8") as f:
        f.write("Test evaluation using best-screw.pt\n")
        f.write(f"Classes: {', '.join(classes)}\n")
        if metrics:
            # f.write(f"Metrics: {str(metrics.box)}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Precision Mean: {prec_mean}\n")
            f.write(f"Recall Mean: {prec_recall}\n")
            f.write(f"mAP@0.5: {map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {map50_95:.4f}\n")
        else:
            f.write("Metrics: not available (evaluation failed)\n")
            if metrics_error:
                f.write(f"Reason: {metrics_error}\n")
        f.write("\nPer-image detections:\n")
        for line in per_image_stats:
            f.write(line + "\n")


def save_predictions(model: YOLO, test_dir: Path, out_images_dir: Path, out_labels_dir: Path,
                     imgsz: int, conf: float, iou: float) -> List[str]:
    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)
    per_image_lines = []

    results = model.predict(
        source=str(test_dir),
        stream=True,
        save=False,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        split="test",
        verbose=False,
    )

    for result in results:
        image_path = Path(result.path)
        stem = image_path.stem

        # save annotated image
        plotted = result.plot()  # BGR ndarray
        pil_img = Image.fromarray(plotted[:, :, ::-1])  # convert to RGB for PIL
        out_img_path = out_images_dir / f"{stem}.jpg"
        pil_img.save(out_img_path)

        # save predicted labels (YOLO format + confidence)
        label_file = out_labels_dir / f"{stem}.txt"
        ensure_dir(label_file.parent)
        with label_file.open("w", encoding="utf-8") as lf:
            # Prefer OBB outputs if available; normalize to YOLO format
            boxes_len = 0
            obb = getattr(result, "obb", None)
            with open("./tem.txt", "w", encoding="utf-8") as f:
                f.write(str(image_path))
                f.write(str(obb))
                f.write("\n\n")
            if obb is not None and len(obb) > 0 and hasattr(obb, "data"):
                h, w_img = result.orig_shape[:2]
                obb_data = obb.data.cpu().numpy()  # [cx, cy, w, h, theta, conf, cls]
                for cx, cy, bw, bh, theta, conf_score, cls_id in obb_data:
                    lf.write(
                        f"{int(cls_id)} {cx / w_img:.6f} {cy / h:.6f} {bw / w_img:.6f} {bh / h:.6f} {theta:.6f} {float(conf_score):.4f}\n"
                    )
                boxes_len = obb_data.shape[0]
            else:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xywhn = boxes.xywhn.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for (x_center, y_center, w, h), cls_id, conf_score in zip(xywhn, cls_ids, confs):
                        lf.write(
                            f"{int(cls_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {float(conf_score):.4f}\n"
                        )
                    boxes_len = len(boxes)
                else:
                    boxes_len = 0

        det_count = boxes_len
        per_image_lines.append(f"{stem}: {det_count} detections saved")

    return per_image_lines


def main():
    parser = argparse.ArgumentParser(description="Evaluate best-screw.pt on test images")
    parser.add_argument("--weights", default="./classification/object_detect/runs/best-screw.pt", help="Path to model weights")
    parser.add_argument("--data", default="./classification/dataset_process/eval_dataset", help="Dataset root containing images/ and labels/")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--out", default="./classification/dataset_process/eval_dataset/eval", help="Output directory")
    parser.add_argument("--classes", default="./classification/dataset_process/eval_dataset/classes.txt", help="Path to classes.txt")
    args = parser.parse_args()

    data_root = Path(args.data)
    classes = read_classes(Path(args.classes))

    out_root = Path(args.out)
    out_images = out_root / "images" / "test"
    out_labels = out_root / "labels" / "test"
    test_images_dir = data_root / "images" / "test"
    result_txt = out_root / "result.txt"
    ensure_dir(out_root)

    # prepare a lightweight data yaml to satisfy Ultralytics validator for OBB
    names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    # Use forward slashes and quote path to avoid YAML parse issues on Windows
    path_str = str(data_root).replace("\\", "/")
    data_yaml = f"""path: "{path_str}"
task: obb
train: images/train
val: images/val
test: images/test
names:
{names_block}
"""
    yaml_path = out_root / "eval_data.yaml"
    yaml_path.write_text(data_yaml, encoding="utf-8")

    model = YOLO(args.weights)

    metrics = None
    metrics_error = None
    try:
        # run evaluation on the test split to collect metrics
        metrics = model.val(
            data=str(yaml_path),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
    except Exception as e:  # OBB validator may fail if labels are axis-aligned
        metrics_error = str(e)
        print(f"Warning: evaluation metrics unavailable: {metrics_error}")

    # save predictions and labeled images
    per_image_lines = save_predictions(
        model=model,
        test_dir=test_images_dir,
        out_images_dir=out_images,
        out_labels_dir=out_labels,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )

    save_result_txt(result_txt, classes, metrics, per_image_lines, metrics_error)
    if metrics:
        print(f"Evaluation complete. Metrics written to {result_txt}.")
    else:
        print(f"Detections saved, but metrics unavailable. See {result_txt} for details.")


if __name__ == "__main__":
    main()

