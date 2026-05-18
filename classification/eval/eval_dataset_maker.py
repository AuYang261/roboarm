

# 将当前文件夹下所有文件夹中对应的同名.png和.json文件（递归查找），
# 将所有数据放入test集，不划分train和val，并放入对应文件夹中重命名，文件组织格式如下

# yolo 标签为 class_index x1 y1 x2 y2 x3 y3 x4 y4
# gemini 标签格式为 class_index x_center y_center width height angle


import os
import shutil
import random
from pathlib import Path
import json
from PIL import Image
import argparse
import numpy as np
from typing import List
import cv2



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_label_file(label_path: Path) -> List[tuple]:
    items = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 9:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            items.append((cls, coords))
    return items


def draw_quad(img: np.ndarray, coords: List[float], color=(0, 255, 0), thickness=2):
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
    pts = pts.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def process_split(images_dir: Path, labels_dir: Path, out_dir: Path, class_names: List[str] | None = None) -> None:
    ensure_dir(out_dir)
    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        img_path = find_image(images_dir, stem)
        if img_path is None:
            print(f"[WARN] image not found for label {label_path}")
            continue

        items = load_label_file(label_path)
        if not items:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] failed to read image {img_path}")
            continue

        h, w = img.shape[:2]
        for cls, coords in items:
            # denormalize
            denorm = []
            for i, v in enumerate(coords):
                if i % 2 == 0:  # x
                    denorm.append(v * w)
                else:  # y
                    denorm.append(v * h)
            draw_quad(img, denorm, color=(0, 255, 0), thickness=2)
            if class_names and 0 <= cls < len(class_names):
                text = class_names[cls]
            else:
                text = str(cls)
            # put text near first point
            p0 = (int(denorm[0]), int(denorm[1]))
            cv2.putText(img, text, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        out_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path), img)



def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_classes_file(path: Path):
    """Parse classes.txt which may be in formats:
    - 'index: name' per line
    - 'name' per line
    Returns (class_map, class_list) where class_map maps label string to index,
    and class_list is ordered by index.
    """
    if not path.exists():
        return None, None
    lines = []
    with path.open('r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)

    class_map = {}
    class_list = []
    if all(':' in ln for ln in lines):
        # index: name format
        tmp = []
        for ln in lines:
            idx_str, name = ln.split(':', 1)
            idx = int(idx_str.strip())
            name = name.strip()
            tmp.append((idx, name))
        tmp.sort(key=lambda x: x[0])
        class_list = [name for _, name in tmp]
        class_map = {name: idx for idx, name in tmp}
    else:
        # names per line
        class_list = [ln for ln in lines]
        class_map = {name: i for i, name in enumerate(class_list)}
    return class_map, class_list


def labelme_json_to_yolo(json_path, image_path):
    """Convert LabelMe JSON (polygons) to YOLO OBB 4-corner lines.
    Returns list of tuples: (class, x1, y1, x2, y2, x3, y3, x4, y4)
    - All coordinates are normalized by image size to [0, 1]
    - Corners are ordered consistently (clockwise) based on PCA axes
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img = Image.open(image_path)
    img_w, img_h = img.size

    lines = []
    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label', '0')
        points = shape.get('points', [])
        if not points:
            continue
        pts = np.array(points, dtype=np.float32)
        if pts.shape[0] < 2:
            continue

        # center of polygon
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        pts0 = pts - np.array([cx, cy], dtype=np.float32)

        # PCA to find principal axis and orthogonal axis
        cov = np.cov(pts0.T)
        vals, vecs = np.linalg.eigh(cov)
        if np.allclose(vals, 0):
            # degenerate polygon -> fallback to axis-aligned bbox corners
            x_min = float(pts[:, 0].min())
            x_max = float(pts[:, 0].max())
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            p1 = np.array([x_max, y_min], dtype=np.float32)
            p2 = np.array([x_max, y_max], dtype=np.float32)
            p3 = np.array([x_min, y_max], dtype=np.float32)
            p4 = np.array([x_min, y_min], dtype=np.float32)
        else:
            order = np.argsort(vals)[::-1]
            u = vecs[:, order[0]]
            u = u / (np.linalg.norm(u) + 1e-8)
            v = np.array([-u[1], u[0]], dtype=np.float32)
            proj_u = pts0 @ u
            proj_v = pts0 @ v
            a = float((proj_u.max() - proj_u.min()) / 2.0)
            b = float((proj_v.max() - proj_v.min()) / 2.0)
            center = np.array([cx, cy], dtype=np.float32)
            # consistent clockwise corners
            p1 = center + a * u + b * v  # top-right (approx.)
            p2 = center + a * u - b * v  # bottom-right
            p3 = center - a * u - b * v  # bottom-left
            p4 = center - a * u + b * v  # top-left

        # normalize to [0,1]
        x1, y1 = p1[0] / img_w, p1[1] / img_h
        x2, y2 = p2[0] / img_w, p2[1] / img_h
        x3, y3 = p3[0] / img_w, p3[1] / img_h
        x4, y4 = p4[0] / img_w, p4[1] / img_h

        lines.append((label, x1, y1, x2, y2, x3, y3, x4, y4))

    return lines

def labelme_json_to_gemini(json_path, image_path):
    """Convert LabelMe JSON (polygons) to YOLO OBB lines.
    Returns list of tuples: (class, x_center, y_center, w, h, angle)
    - x_center, y_center, w, h are normalized by image size
    - angle is in radians, range approximately [-pi/2, pi/2]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img = Image.open(image_path)
    img_w, img_h = img.size

    lines = []
    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label', '0')
        points = shape.get('points', [])
        if not points:
            continue
        pts = np.array(points, dtype=np.float32)
        if pts.shape[0] < 2:
            continue

        # center of polygon
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        pts0 = pts - np.array([cx, cy], dtype=np.float32)

        # PCA to find principal axis
        cov = np.cov(pts0.T)
        vals, vecs = np.linalg.eigh(cov)
        if np.allclose(vals, 0):
            # degenerate polygon -> fallback to axis-aligned bbox
            x_min = float(pts[:, 0].min())
            x_max = float(pts[:, 0].max())
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            w_abs = x_max - x_min
            h_abs = y_max - y_min
            angle = 0.0
        else:
            order = np.argsort(vals)[::-1]
            u = vecs[:, order[0]]
            u = u / (np.linalg.norm(u) + 1e-8)
            v = np.array([-u[1], u[0]], dtype=np.float32)
            proj_u = pts0 @ u
            proj_v = pts0 @ v
            w_abs = float(proj_u.max() - proj_u.min())
            h_abs = float(proj_v.max() - proj_v.min())
            angle = float(np.arctan2(u[1], u[0]))
            # normalize angle to [-pi/2, pi/2]
            if angle < -np.pi/2:
                angle += np.pi
            elif angle > np.pi/2:
                angle -= np.pi

        x_center = cx / img_w
        y_center = cy / img_h
        w = w_abs / img_w
        h = h_abs / img_h

        lines.append((label, x_center, y_center, w, h, angle))

    return lines


def build_class_map(all_json_paths):
    labels = set()
    for jp in all_json_paths:
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for s in data.get('shapes', []):
                labels.add(s.get('label', '0'))
        except Exception:
            continue
    class_list = sorted(labels)
    class_map = {c: i for i, c in enumerate(class_list)}
    return class_map, class_list


def process_dataset(src_dir='.', dst_dir='dataset', img_ext='png', seed=42, classes_file: str | None = None):
    src = Path(src_dir)
    dst = Path(dst_dir)

    # gather all json paths for classes recursively
    all_jsons = []
    file_pairs = []  # list of (json_path, img_path, relative_parent?) maybe we need relative path for grouping?

    # recursively find all json files under src_dir
    for json_path in src.rglob('*.json'):
        # corresponding image file should be in same directory with same stem
        img_path = json_path.with_suffix('.' + img_ext)
        if img_path.exists():
            all_jsons.append(str(json_path))
            file_pairs.append((json_path, img_path))

    # file_pairs now contains all valid (json, image) pairs

    # build class map: prefer provided classes.txt if present
    create_dir_if_not_exists(dst)
    cls_path_candidates = []
    if classes_file:
        cls_path_candidates.append(Path(classes_file))
    cls_path_candidates.append(dst / 'classes.txt')
    cls_path_candidates.append(Path('./classes.txt'))

    class_map = None
    class_list = None
    for p in cls_path_candidates:
        cm, cl = parse_classes_file(p)
        if cm:
            class_map, class_list = cm, cl
            break

    if class_map is None or class_list is None:
        # fallback: derive from JSONs and write classes.txt
        class_map, class_list = build_class_map(all_jsons)
        with open(dst / 'classes.txt', 'w', encoding='utf-8') as f:
            for i, c in enumerate(class_list):
                f.write(f"{i}: {c}\n")

    # create output dirs (only test)
    for kind in [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test',
        'labels_gemini/train', 'labels_gemini/val', 'labels_gemini/test']:
        create_dir_if_not_exists(dst / kind)

    random.seed(seed)

    # counter for test split only
    test_counter = 1

    # process all file pairs as test set
    for json_path, img_path in file_pairs:
        # ensure files exist (already checked, but double-check)
        if not json_path.exists() or not img_path.exists():
            continue

        idx = test_counter
        dst_img_name = f'image{idx:06d}.' + img_ext
        dst_label_name = f'image{idx:06d}.txt'

        shutil.copy2(img_path, dst / 'images/test' / dst_img_name)

        # convert json to yolo and map labels to indices per classes.txt
        yolo_items = labelme_json_to_yolo(str(json_path), str(img_path))
        with open(dst / 'labels/test' / dst_label_name, 'w', encoding='utf-8') as lf:
            for (label, x1, y1, x2, y2, x3, y3, x4, y4) in yolo_items:
                cls_idx = class_map.get(label, 0)
                lf.write(
                    f"{cls_idx} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
                )
        # convert json to gemini format and write
        gemini_items = labelme_json_to_gemini(str(json_path), str(img_path))
        with open(dst / 'labels_gemini/test' / dst_label_name, 'w', encoding='utf-8') as gf:
            for (label, xc, yc, w, h, angle) in gemini_items:
                cls_idx = class_map.get(label, 0)
                gf.write(
                    f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {angle:.6f}\n"
                )

        test_counter += 1


def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON to YOLO txt (all data as test set)')
    parser.add_argument('--src', default='./classification/dataset_process/dataset/original', help='Source root containing json and image files (recursive)')
    parser.add_argument('--dst', default='./classification/dataset_process/eval_dataset', help='Destination dataset folder')
    parser.add_argument('--ext', default='png', help='Image extension (png/jpg)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--classes', default="./classification/dataset_process/eval_dataset/classes.txt", help='Optional path to classes.txt to use for indices')
    
    parser.add_argument("--img", default="./classification/dataset_process/eval_dataset/images/test", help="Images directory")
    parser.add_argument("--label", default="./classification/dataset_process/eval_dataset/labels/test", help="Labels directory (quad format)")
    parser.add_argument("--out", default="./classification/dataset_process/eval_dataset/test_img", help="Output directory for annotated images")
    args = parser.parse_args()

    process_dataset(src_dir=args.src, dst_dir=args.dst, img_ext=args.ext, seed=args.seed, classes_file=args.classes)

    images_dir = Path(args.img)
    labels_dir = Path(args.label)
    out_dir = Path(args.out)

    class_names = None
    if args.classes:
        class_path = Path(args.classes)
        if class_path.exists():
            with class_path.open("r", encoding="utf-8") as f:
                class_names = [ln.strip() for ln in f if ln.strip()]
    process_split(images_dir, labels_dir, out_dir, class_names)

if __name__ == '__main__':
    main()
            




