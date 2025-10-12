


import cv2
import threading
import time
import requests
import numpy as np

def start_mjpeg_client(host: str, port: int):
    url = f"http://{host}:{port}/video_feed"
    print(f"Connecting to {url}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return

    cv2.namedWindow('Web Camera Client', cv2.WINDOW_NORMAL)
    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame")
            break

        cv2.imshow('Web Camera Client', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_camera_client(host: str, port: int, video_type: str = "mjpeg"):
    if video_type == "mjpeg":
        start_mjpeg_client(host, port)
    else:
        print(f"Unsupported video type: {video_type}")

def main():
    host = "localhost"
    port = 8080
    camera_index = 4  # default camera index; change if needed

    start_camera_client(host, port, video_type="mjpeg")

if __name__ == '__main__':
    main()
