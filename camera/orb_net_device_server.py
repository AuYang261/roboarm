


import cv2
import time
import numpy as np
from flask import Flask, Response
from pyorbbecsdk import *
from utils import frame_to_bgr_image


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
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
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

    app.run(host=host, port=port)
    
    pipeline.stop()
    
    

def main():
    host = "localhost"
    port = 8080
    camera_index = 4  # default camera index; change if needed

    start_camera_server(host, port, camera_index, video_type="mjpeg")

if __name__ == '__main__':
    main()
