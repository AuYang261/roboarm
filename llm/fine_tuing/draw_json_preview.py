import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

COLOR = {
    "red_block":    (0,   50,  220),
    "blue_block":   (200, 100, 0),
    "yellow_block": (0,   200, 220),
}

original_dir = Path("original")
preview_dir  = Path("original_preview")
preview_dir.mkdir(exist_ok=True)

json_files = sorted(original_dir.glob("*.json"))
for json_path in json_files:
    img_path = json_path.with_suffix(".png")
    if not img_path.exists():
        print(f"图片不存在: {img_path}, 跳过")
        continue

    img = cv2.imread(str(img_path))
    data = json.loads(json_path.read_text())

    for shape in data["shapes"]:
        label  = shape["label"]
        points = np.array(shape["points"], dtype=np.float32).astype(np.int32)
        color  = COLOR.get(label, (0, 255, 0))

        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
        cv2.putText(
            img, label,
            (points[0][0], max(0, points[0][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

    save_path = preview_dir / img_path.name
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imsave(str(save_path), img_rgb)

print(f"完成，生成 {len(json_files)} 张预览图 -> original_preview/")
