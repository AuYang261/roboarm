# Description: 调用目标识别和机械臂控制，实现抓取功能。
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_base import Arm
from config_getter import get_config_value
import numpy as np
from object_detect.detect import (
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
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    default_gripper_aside_pos = get_config_value("default_gripper_aside_pos")
    default_conf_thres = get_config_value("default_conf_thres")
    class_pos = get_config_value("class_pos")
    place_distance_threshold = get_config_value("place_distance_threshold")
    offset = get_config_value("catch_offset")

    arm = Arm()
    arm.move_to_home(gripper_open_0to1=1)
    cam = Camera(color=True, depth=False)
    models = [load_model(model_path) for model_path in model_paths]
    detections = []
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
                        detect_objects_in_frame(
                            model, frame, conf_thres=default_conf_thres
                        )
                    )
            for (u, v, w, h, r), score, class_id, class_name in detections:
                angle_deg = np.rad2deg(r)
                if future is None or future.done():
                    # 将图像坐标转换为机械臂坐标系
                    target_x, target_y = arm.pixel2pos(u, v)
                    gripper_angle_rad = arm.gripper_angle_by_longer(
                        u, v, w, h, angle_deg
                    )
                    if (
                        np.linalg.norm(
                            np.array(class_pos.get(class_name, [-0.2, 0.0]))
                            - np.array([target_x, target_y])
                        )
                        < place_distance_threshold
                    ):
                        print(
                            f"Object {class_name} is too close to place position, skipping catch."
                        )
                    else:
                        future = executor.submit(
                            arm.catch_and_place,
                            # 夹爪向外偏移一些，避免刚好顶到物体
                            target_x + offset * np.cos(gripper_angle_rad),
                            target_y + offset * np.sin(-gripper_angle_rad),
                            gripper_angle_rad,
                            class_pos.get(class_name, [-0.2, 0.0]),
                        )
                draw_box(frame, u, v, w, h, angle_deg, f"{class_name}: {score:.2f}")

            if future is None or future.done():
                # 移到旁边以免挡住视野
                future = executor.submit(
                    arm.move_to,
                    default_gripper_aside_pos,
                    1,
                )
            end_time = time.time()
            if end_time - start_time == 0:
                fps = 0.0
            else:
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

    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    arm.disconnect_arm()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
