
# python camera/orb_camera_client.py --host 127.0.0.1 --port 8083 --mode multi

import cv2
import threading
import time
import requests
import numpy as np
import argparse
import json
import signal

# Global stop event for graceful shutdown
stop_event = threading.Event()

def start_rgb_client(host: str, port: int):
    url = f"http://{host}:{port}/rgb_stream"
    print(f"Connecting to {url}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Unable to open rgb stream")
        return

    cv2.namedWindow('Web Camera Client RGB', cv2.WINDOW_NORMAL)
    print("Press 'q' to exit")
    
    err_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # stream may be temporarily unavailable; retry shortly
            
            err_count += 1
            if err_count > 30:
                print("Error: Unable to read from rgb stream")
                break
            continue
        err_count = 0

        # 打印当前帧率 分辨率
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"FPS: {fps}, Resolution: {int(width)}x{int(height)}", end='\r')
        # cv2 上打印
        cv2.putText(frame, f"FPS: {fps}, Resolution: {int(width)}x{int(height)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Web Camera Client RGB', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # 只摧毁当前窗口
    cv2.destroyWindow('Web Camera Client RGB')

def start_depth_client(host: str, port: int):
    url = f"http://{host}:{port}/depth_stream"
    print(f"Connecting to {url}")

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error: Unable to open depth stream")
        return

    cv2.namedWindow('Web Camera Client Depth', cv2.WINDOW_NORMAL)
    print("Press 'q' to exit")

    err_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            
            err_count += 1
            if err_count > 30:
                print("Error: Unable to read from depth stream")
                break
            continue
        err_count = 0
        # 打印当前帧率 分辨率
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"FPS: {fps}, Resolution: {int(width)}x{int(height)}", end='\r')
        # cv2 上打印
        cv2.putText(frame, f"FPS: {fps}, Resolution: {int(width)}x{int(height)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Web Camera Client Depth', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow('Web Camera Client Depth')

# 存在问题，传输速度过慢
def start_dict_client(host: str, port: int):
    url = f"http://{host}:{port}/dict_stream"
    print(f"Connecting to {url}")

    with requests.get(url, stream=True) as r:
        # 一次读一块，遇到边界就拆包
        boundary = b'--dictboundary'
        buffer = b''
        for chunk in r.iter_content(chunk_size=1024):
            if stop_event.is_set():
                break
            if not chunk:
                continue
            buffer += chunk
            # extract all complete parts
            while True:
                idx = buffer.find(boundary)
                if idx == -1:
                    break
                # part is data before this boundary
                part = buffer[:idx]
                buffer = buffer[idx + len(boundary):]
                part = part.strip()
                if not part or part == b'--':
                    continue
                # need header/body separator \r\n\r\n
                sep = b'\r\n\r\n'
                if sep not in part:
                    # incomplete, prepend back and wait for more data
                    buffer = part + boundary + buffer
                    break
                head, body = part.split(sep, 1)
                # body may have trailing CRLF
                body = body.strip()
                try:
                    text = body.decode('utf-8')
                except Exception as e:
                    print('Failed to decode body bytes:', e)
                    continue
                try:
                    json_data = json.loads(text)
                except json.JSONDecodeError as e:
                    print('JSON decode error:', e)
                    # print a short snippet for debugging
                    snippet = text[:200].replace('\n', '\\n')
                    print('Body snippet:', snippet)
                    continue
                dic = json_data
                print(f"Received dict of type {type(dic)} with {len(dic)} keys")

def _signal_handler(sig, frame):
    print('Received signal, shutting down...')
    stop_event.set()

signal.signal(signal.SIGINT, _signal_handler)
# 注册 ctrl +c 信号处理函数
signal.signal(signal.SIGTERM, _signal_handler)

def start_multi_client(host: str, port: int):
    # 多线程
    t1 = threading.Thread(target=start_rgb_client, args=(host, port))
    t2 = threading.Thread(target=start_depth_client, args=(host, port))
    
    t1.start()
    t2.start()
    
    # t1.join()
    # t2.join()

def start_usb_camera_client(host: str, port: int):
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


def main():
    
    # 读取 args 读取 host 和 port
    # python web_camera_client.py --host 127.0.0.1 --port 8081
    parser = argparse.ArgumentParser(description="Web Camera Client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8083, help="Server port")
    parser.add_argument("--mode", type=str, default="multi", help="Video source (default: rgb)")
    args = parser.parse_args()
    
    if args.mode == "none":
        pass
    elif args.mode == "dict":
        start_dict_client(args.host, args.port)
    elif args.mode == "rgb":
        start_rgb_client(args.host, args.port)
    elif args.mode == "depth":
        start_depth_client(args.host, args.port)
    elif args.mode == "multi":
        start_multi_client(args.host, args.port)
    elif args.mode == "usb":
        start_usb_camera_client(args.host, args.port)

if __name__ == '__main__':
    main()
