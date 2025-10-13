


import cv2
import threading
import time
import requests
import numpy as np
import argparse

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

        # 打印当前帧率 分辨率
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"FPS: {fps}, Resolution: {int(width)}x{int(height)}", end='\r')
        # cv2 上打印
        cv2.putText(frame, f"FPS: {fps}, Resolution: {int(width)}x{int(height)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
    
    # 读取 args 读取 host 和 port
    # python web_camera_client.py --host 127.0.0.1 --port 8081
    parser = argparse.ArgumentParser(description="Web Camera Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    args = parser.parse_args()

    host = args.host
    port = args.port
    # camera_index = 4  # default camera index; change if needed

    start_camera_client(host, port, video_type="mjpeg")

if __name__ == '__main__':
    main()
