# Description: 调用目标识别和机械臂控制，实现抓取功能。
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_control import Arm
import numpy as np
from classification.object_detect.detect import (
    detect_objects_in_frame,
    load_model,
    draw_box,
)
from camera.camera_api import Camera
import cv2
import time
import concurrent.futures


def main():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    model_paths = [
        os.path.join(
            os.path.dirname(__file__), "object_detect", "runs", "积木方块", "best.pt"
        ),
        os.path.join(
            os.path.dirname(__file__),
            "object_detect",
            "runs",
            "积木螺丝国际象棋",
            "best.pt",
        ),
    ]

    arm = Arm(port="COM3")
    arm.move_to_home(gripper_angle_deg=80)
    cam = Camera(ip="127.0.0.1", color=True, depth=False)
    models = [load_model(model_path) for model_path in model_paths]
    future = None

    while True:
        start_time = time.time()
        try:
            frames = cam.get_frames()
            if frames is None:
                continue
            frame = frames.get("color")
            if frame is None:
                continue

            if future is None or future.done():
                detections = []
                for model in models:
                    detections.extend(
                        detect_objects_in_frame(model, frame, conf_thres=0.5)
                    )
            else:
                detections = detections
            if len(detections) == 0 and (future is None or future.done()):
                # 移到旁边以免挡住视野
                future = executor.submit(
                    arm.move_to,
                    [0.1, 0.0, 0.12],
                    80,
                )
            for (u, v, w, h, r), score, class_id, class_name in detections:
                angle_deg = np.rad2deg(r)
                if future is None or future.done():
                    # 将图像坐标转换为机械臂坐标系
                    target_x, target_y = arm.pixel2pos(u, v)
                    box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
                    # 计算较长边的两顶点
                    if np.linalg.norm(box_points[0] - box_points[1]) > np.linalg.norm(
                        box_points[1] - box_points[2]
                    ):
                        box_points = (
                            [box_points[0], box_points[1]]
                            if box_points[0][0] < box_points[1][0]
                            else [box_points[1], box_points[0]]
                        )
                    else:
                        box_points = (
                            [box_points[1], box_points[2]]
                            if box_points[1][0] < box_points[2][0]
                            else [box_points[2], box_points[1]]
                        )
                    # gripper_angle_rad 沿着物体长边方向，在[-pi/2, pi/2]范围内
                    gripper_angle_rad = np.pi / 2 + np.arctan2(
                        box_points[1][1] - box_points[0][1],
                        box_points[1][0] - box_points[0][0],
                    )
                    if gripper_angle_rad > np.pi / 2:
                        gripper_angle_rad -= np.pi

                    # 夹爪向外偏移一些，避免刚好顶到物体
                    offset = 0.00
                    future = executor.submit(
                        arm.catch,
                        target_x + offset * np.cos(gripper_angle_rad),
                        target_y + offset * np.sin(-gripper_angle_rad),
                        gripper_angle_rad,
                        [0.2, -0.08],
                        0.073,
                        0.1,
                    )
                draw_box(frame, u, v, w, h, angle_deg, f"{class_name}: {score:.2f}")

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # 按Esc键退出
                break

        except KeyboardInterrupt:
            print("Exiting...")

    arm.move_to_home(gripper_angle_deg=80)
    arm.disconnect_arm()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
