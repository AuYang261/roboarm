


import cv2
import time
import numpy as np
from flask import Flask, Response
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import argparse

ESC_KEY = 27


def start_camera_server(host: str, port: int, camera_index: int, mode: str = "RGB", video_type: str = "mjpeg"):
    
    sensor_type = OBSensorType.COLOR_SENSOR
    
    if mode == "DEPTH":
        print("todo: depth mode not implemented yet")
        return
        
    config = Config()
    pipeline = Pipeline()
    app = Flask(__name__)

    try:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        try:
            # 1280*720
            # color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)
    
    def generate_frames():
        while True:
            try:
                frames: FrameSet = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                
                # covert to RGB format
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("failed to convert frame to image")
                    continue
                
                frame = cv2.imencode('.jpg', color_image)[1].tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except KeyboardInterrupt:
                pass
        
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    pipeline.stop()

    
    

def main():
    # 读取 args 读取 host 和 port
    # python orb_net_device_server.py --host 0.0.0.0 --port 8081

    parser = argparse.ArgumentParser(description="ORB-NET Camera Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind the server")
    parser.add_argument("--camera_index", type=int, default=4, help="Camera index")
    args = parser.parse_args()

    host = args.host
    port = args.port
    camera_index = args.camera_index

    start_camera_server(host, port, camera_index, video_type="mjpeg")

if __name__ == '__main__':
    main()
