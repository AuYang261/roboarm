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
from camera.orb_camera import open_camera, get_frames, close_camera
import cv2
import time
import concurrent.futures


def main():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    model_path = os.path.join(
        os.path.dirname(__file__), "object_detect", "runs", "best.pt"
    )

    arm = Arm(port="COM3")
    arm.move_to_home(gripper_angle_deg=80)
    cam = open_camera(color=True, depth=False)
    model = load_model(model_path)

    while True:
        start_time = time.time()
        futures = []
        try:
            frames = get_frames(cam)
            if frames is None:
                continue
            frame = frames.get("color")
            if frame is None:
                continue

            detections = detect_objects_in_frame(model, frame)
            for (x, y, w, h, r), score, class_id, class_name in detections:
                futures.append(executor.submit(arm.catch, x, y, r))
                draw_box(frame, x, y, w, h, np.rad2deg(r), f"{class_name}: {score:.2f}")

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
        for f in futures:
            f.result()  # 等待所有任务完成

    arm.move_to_home(gripper_angle_deg=80)
    arm.disconnect_arm()
    close_camera(cam)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
