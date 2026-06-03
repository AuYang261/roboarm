# Description: 调用目标识别和机械臂控制，实现抓取功能。
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_base import Arm
from utils.config_getter import get_config_value
import numpy as np
import threading
from object_detect.detect import (
    detect_objects_in_frame,
    load_model,
    draw_box,
)
from camera.camera_api import Camera
import time
import copy
from utils.cv2_display import show_image, poll_key, destroy_all_windows


def classify_and_grasp_objects(arm, frame, models, conf_thres: float) -> list[dict]:
    """对一帧图像运行 YOLO 检测，逐个抓取识别到的物体。

    Args:
        arm: Arm() 实例。
        frame: BGR 图像 (numpy array)。
        models: YOLO 模型列表。
        conf_thres: 置信度阈值。

    Returns:
        [(u, v, w, h, r), score, class_id, class_name, grasp_success]
        每个检测结果附加 grasp_success 字段。
    """
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    place_distance_threshold = get_config_value(
        "place_distance_threshold", default=0, raise_if_missing=False
    )
    offset = get_config_value("catch_offset")

    detections = []
    for model in models:
        detections.extend(detect_objects_in_frame(model, frame, conf_thres=conf_thres))

    results = []
    for (u, v, w, h, r), score, class_id, class_name in detections:
        angle_deg = np.rad2deg(r)
        target_x, target_y = arm.pixel2pos(u, v)
        gripper_angle_rad = arm.gripper_angle_by_longer(u, v, w, h, angle_deg)

        class_place_pos_data = place_pos.get(class_name)
        if class_place_pos_data is None or "pos" not in class_place_pos_data:
            class_place_pos = [target_x, target_y]
        else:
            class_place_pos = copy.deepcopy(class_place_pos_data["pos"])
            for i, v_ref in enumerate(class_place_pos):
                if v_ref == "x":
                    class_place_pos[i] = target_x
                elif v_ref == "-x":
                    class_place_pos[i] = -target_x
                elif v_ref == "y":
                    class_place_pos[i] = target_y
                elif v_ref == "-y":
                    class_place_pos[i] = -target_y

        if (
            place_distance_threshold > 0
            and np.linalg.norm(
                np.array(class_place_pos) - np.array([target_x, target_y])
            )
            < place_distance_threshold
        ):
            grasp_success = None  # skipped
        else:
            grasp_success = arm.catch_and_place(
                target_x + offset * np.cos(gripper_angle_rad),
                target_y + offset * np.sin(-gripper_angle_rad),
                gripper_angle_rad,
                class_place_pos,
            )

        results.append(((u, v, w, h, r), score, class_id, class_name, grasp_success))

    return results


def main():
    model_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    default_gripper_aside_pos = get_config_value(
        "default_gripper_aside_pos", raise_if_missing=False
    )
    default_conf_thres = get_config_value("default_conf_thres")

    arm = Arm()
    arm.move_to_home(gripper_open_0to1=1)
    cam = Camera(color=True, depth=False)
    models = [load_model(model_path) for model_path in model_paths]

    latest_results: list = []
    results_lock = threading.Lock()
    cam_lock = threading.Lock()
    stop_event = threading.Event()

    def detection_loop():
        while not stop_event.is_set():
            with cam_lock:
                frames = cam.get_frames()
            if frames is None:
                continue
            frame = frames.get("color")
            if frame is None:
                continue

            results = classify_and_grasp_objects(arm, frame, models, default_conf_thres)

            with results_lock:
                latest_results[:] = results

            if default_gripper_aside_pos:
                arm.move_to(default_gripper_aside_pos, 1)

    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    try:
        while not stop_event.is_set():
            with cam_lock:
                frames = cam.get_frames()
            if frames is None:
                continue
            frame = frames.get("color")
            if frame is None:
                continue

            with results_lock:
                results_copy = list(latest_results)

            for (u, v, w, h, r), score, _, class_name, grasp_success in results_copy:
                angle_deg = np.rad2deg(r)
                if grasp_success is None:
                    status = "SKIP"
                elif grasp_success:
                    status = "OK"
                else:
                    status = "FAIL"
                draw_box(
                    frame, u, v, w, h, angle_deg, f"{class_name}: {score:.2f} {status}"
                )

            show_image("Detections", frame)
            if poll_key(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        stop_event.set()

    detection_thread.join(timeout=2)
    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    arm.disconnect_arm()
    cam.close()
    destroy_all_windows()


if __name__ == "__main__":
    main()
