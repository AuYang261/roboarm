# Description: 控制usb相机，获取图片用来构建数据集
import cv2
import numpy as np
import os
import sys

from requests import get

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from camera.orb_camera import open_camera, get_frames, close_camera


def main():
    print("Opening camera...")
    cap = open_camera(True, False)
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "original"
    )
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    img_count = 0
    while True:
        frames = get_frames(cap)
        if frames.get("color") is None:
            print("Failed to grab frame")
            continue
        cv2.imshow("Camera", frames["color"])
        key = cv2.waitKey(1)
        if key == ord("c") or key == ord(" "):  # 按 'c'/空格 键拍照
            while True:
                img_name = os.path.join(dataset_path, f"img_{img_count:03d}.png")
                if not os.path.exists(img_name):
                    break
                img_count += 1
            cv2.imwrite(img_name, frames["color"])
            print(f"Captured {img_name}")
            img_count += 1
        elif key == ord("q"):  # 按 'q' 键退出
            print("Exiting...")
            break
    close_camera(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
