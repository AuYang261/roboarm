
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import argparse
import sys

def detect_objects_in_frame(model, frame, conf_thres=0.8, iou_thres=0.45):
    results = model(frame, conf=conf_thres, iou=iou_thres)[0]
    detections = results.obb.xywhr.cpu().numpy()  # xywhr format
    scores = results.obb.conf.cpu().numpy()
    class_ids = results.obb.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]
    return [
        ((u, v, w, h, r), score, class_id, class_name)
        for (u, v, w, h, r), score, class_id, class_name in zip(
            detections, scores, class_ids, class_names
        )
    ]


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


def load_model(model_path, device=""):
    model = YOLO(model_path)
    if device:
        model.to(device)
    return model



# 从 ip_camera_client:port_camera_client 获取视频流进行目标检测，将结果通过 ip_model_server:port_model_server 提供服务
def web_model_server(model_path, host : str, port : int, conf_thres :float, iou_thres : float):

    # Load model
    model = load_model(model_path)
    
    url = f"http://{host}:{port}/rgb_stream"

    # Open video capture
    print(f"Connecting to {url}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Error: Unable to open video stream: {url}")
        return

    # 连续读取失败计数
    err_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # stream may be temporarily unavailable; retry shortly
            time.sleep(0.05)
            err_count += 1
            if err_count > 30:
                print("Error: Unable to read from rgb stream")
                break
            continue
        err_count = 0

        start_time = time.time()

        # Perform inference
        results = detect_objects_in_frame(
            # model, frame, conf_thres=args.conf_thres, iou_thres=args.iou_thres
            model, frame, conf_thres=conf_thres, iou_thres=iou_thres
        )
        annotated_frame = frame
        for (x, y, w, h, r), score, class_id, class_name in results:
            draw_box(
                annotated_frame,
                x,
                y,
                w,
                h,
                np.rad2deg(r),
                f"{class_name}: {score:.2f}",
            )
        end_time = time.time()
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
        
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv8")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "runs", "best.pt"),
        help="Path to the YOLOv8/v11 model",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.8,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou-thres", 
        type=float, 
        default=0.45, 
        help="IoU threshold for NMS"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="192.168.0.5",
        help="Host for the model server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8083,
        help="Port for the model server",
    )
    args = parser.parse_args()

    web_model_server(
        model_path=args.model,

        host=args.host,
        port=args.port,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
    

if __name__ == "__main__":
    main()