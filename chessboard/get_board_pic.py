# Description: 控制usb相机，并通过鼠标点击标定棋盘格的四个角点
from calendar import c
import cv2
import numpy as np
import os
import sys

from polars import col
from requests import get

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.camera_api import Camera


POINTS = []


def get_4_corners(cap: Camera):
    global POINTS

    # 定义鼠标事件回调函数
    def mouse_callback(event, x, y, flags, param):
        global POINTS
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            print(f"Left button clicked at ({x}, {y})")
            POINTS.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
            print(f"Deleted last point")
            if POINTS:
                POINTS.pop()

    # 创建窗口并绑定鼠标回调函数
    window_name = "Camera"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        frames = cap.get_frames()
        color_frame = frames.get("color")
        if color_frame is None:
            print("Failed to grab frame")
            continue

        # Connect POINTS in sequence
        for i in range(len(POINTS)):
            cv2.line(color_frame, POINTS[i - 1], POINTS[i], (0, 0, 255), 2)
        # Draw points
        for point in POINTS:
            cv2.circle(color_frame, point, 5, (0, 0, 255), -1)
        cv2.putText(
            color_frame,
            "left click to select 4 corners of the chessboard, right click to undo",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            color_frame,
            "press 'q' to quit and save points",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        # Display the resulting frame
        cv2.imshow("USB Camera", color_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # save POINTS
    with open("points.txt", "w") as f:
        for point in POINTS:
            f.write(f"{point[0]},{point[1]}\n")


def main():
    global POINTS
    print("Opening camera...")
    cap = Camera(color=True, depth=False)
    if not os.path.exists("points.txt"):
        get_4_corners(cap)
    else:
        with open("points.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                x, y = map(int, line.strip().split(","))
                POINTS.append((x, y))
    while len(POINTS) != 4:
        print("Please select exactly 4 points.")
        get_4_corners(cap)
    # 按照顺时针顺序排列角点
    POINTS = sorted(POINTS, key=lambda x: (x[1], x[0]))
    if POINTS[0][0] > POINTS[1][0]:
        POINTS[0], POINTS[1] = POINTS[1], POINTS[0]
    if POINTS[2][0] < POINTS[3][0]:
        POINTS[2], POINTS[3] = POINTS[3], POINTS[2]
    print(f"Loaded points: {POINTS}")
    while (color_frame := cap.get_frames().get("color")) is None:
        print("Failed to grab frame")
    width, height = color_frame.shape[1], color_frame.shape[0]
    # 将角点往外扩展几个像素
    padding = 30
    POINTS_PADDED = []
    POINTS_PADDED.append(
        (max(POINTS[0][0] - padding, 0), max(POINTS[0][1] - padding, 0))
    )
    POINTS_PADDED.append(
        (min(POINTS[1][0] + padding, width), max(POINTS[1][1] - padding, 0))
    )
    POINTS_PADDED.append(
        (max(POINTS[2][0] + padding, 0), min(POINTS[2][1] + padding, height))
    )
    POINTS_PADDED.append(
        (min(POINTS[3][0] - padding, width), min(POINTS[3][1] + padding, height))
    )
    print(f"Padded points: {POINTS_PADDED}")
    # 根据4角点计算仿射变换矩阵
    pts1 = np.float32(np.array(POINTS_PADDED))
    pts2 = np.float32(np.array([(0, 0), (width, 0), (width, height), (0, height)]))
    M = cv2.getPerspectiveTransform(pts1, pts2)

    while True:
        color_frame = cap.get_frames().get("color")
        if color_frame is None:
            print("Failed to grab frame")
            continue

        # 仿射变换
        frame = cv2.warpPerspective(color_frame, M, (width, height))

        cv2.imshow("USB Camera", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.close()


if __name__ == "__main__":
    main()
