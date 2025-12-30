import yaml
import cv2
import numpy as np
import time
import os
import sys


'''
conda create -n rknn_lite python=3.10
conda activate rknn_lite
wd
sudo apt remove cmake
sudo apt install -y build-essential git libprotobuf-dev protobuf-compiler libssl-dev
# 安装对应版本的 cmake https://ashe27.github.io/2024/11/30/linux-build-cmake/ cmake version 3.28.1

pip install --upgrade pip setuptools wheel
pip install -r ./calibration/arm64_requirements_cp310.txt

pip install "onnxoptimizer==0.3.8" "onnxruntime>=1.16.0"
# pip install ./calibration/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install ./calibration/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
'''


import yaml
import cv2
import numpy as np
import time
import os
import sys

# 使用 rknn_lite Lite（适用于板端推理）
from rknnlite.api import RKNNLite

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera


def preprocess_frame(frame, input_size=(640, 640)):
    """将 BGR 图像预处理为模型输入格式（RGB + resize + 归一化）"""
    # 转为 RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize 到模型输入尺寸
    img_resized = cv2.resize(img_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    # 归一化到 [0, 1] 或 [0, 255] —— 根据你转换 rknn_lite 时的 config 决定
    # 假设训练时是 /255.0，则这里也 /255.0；如果 rknn_lite config 用了 mean=[0,0,0], std=[255,255,255]，则输入应为 uint8
    # 这里我们保持 uint8，让 rknn_lite 内部做 (x - mean) / std
    img = np.expand_dims(img_resized, 0)
    return img


def postprocess_rknn_output(output, conf_thres=0.8, iou_thres=0.45, input_size=(640, 640)):
    """
    后处理 rknn_lite 输出
    假设 output 是 [1, N, 7]: x, y, w, h, angle(rad), conf, cls
    """
    detections = output[0]  # [N, 7]
    # 过滤低置信度
    mask = detections[:, 5] >= conf_thres
    detections = detections[mask]

    if len(detections) == 0:
        return []

    boxes = detections[:, :5]  # x, y, w, h, angle
    scores = detections[:, 5]
    class_ids = detections[:, 6].astype(int)

    # 可选：添加 NMS（此处省略，YOLOv11 OBB 的 NMS 较复杂，若模型已内置可跳过）
    # 简单起见，先不过滤重复框

    return boxes, scores, class_ids


def draw_box(frame, u, v, w, h, angle_deg, label):
    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
    box_points = np.int64(box_points)
    cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (int(u - w / 2), int(v - h / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

def main():
    config_yaml = yaml.safe_load(
        open(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config.yaml",
            ),
            encoding="utf-8",
        )
    )
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        config_yaml.get("classification_YOLO_model_path", ["best.rknn_lite"])[0]
    )
    default_conf_thres = config_yaml.get("default_conf_thres", 0.8)

    # Load rknn_lite model
    rknn_lite = RKNNLite()
    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    
    
    # Open camera
    camera = Camera(color=True, depth=False)

    while True:
        frames = camera.get_frames()
        if frames.get("color") is None:
            print("Failed to grab frame")
            continue

        frame = frames["color"]
        start_time = time.time()

        # Preprocess
        input_frame = preprocess_frame(frame, input_size=(1280, 1280))  # 根据你的模型调整
        # Inference
        # print(input_frame.shape)
        outputs = rknn_lite.inference(inputs=[input_frame])
        # outputs 是 list of numpy arrays，通常只有一个输出
        if len(outputs) == 0:
            continue
        raw_output = outputs[0]  # shape: [1, N, 7] 或类似

        # Postprocess
        try:
            boxes, scores, class_ids = postprocess_rknn_output(
                raw_output, conf_thres=default_conf_thres
            )
        except Exception as e:
            print(f"Postprocess error: {e}")
            boxes, scores, class_ids = [], [], []

        annotated_frame = frame.copy()
        class_names = ["TapScrew", "PanScrew", "ShortPanScrew", "MiniPanScrew", "HexBolt", "HexNut", "BlackHexNut"]  # ⚠️ 替换为你的实际类别名！
        # 更好的方式：从 config.yaml 读取
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x, y, w, h, angle_rad = box
            angle_deg = np.rad2deg(angle_rad)
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
            draw_box(
                annotated_frame,
                x,
                y,
                w,
                h,
                angle_deg,
                f"{class_name}: {score:.2f}",
            )

        end_time = time.time()
        print(f"Inference + Postprocess time: {end_time - start_time:.3f} s")
        fps = 1 / (end_time - start_time)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        
        # 将窗口最大化 
        cv2.imshow("rknn_lite YOLOv11 OBB Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    camera.close()
    cv2.destroyAllWindows()
    rknn_lite.release()


if __name__ == "__main__":
    main()