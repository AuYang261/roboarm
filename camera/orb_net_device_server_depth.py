


import cv2
import time
import numpy as np
from flask import Flask, Response
from pyorbbecsdk import *
from utils import frame_to_bgr_image
import argparse

ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


def start_camera_server(host: str, port: int, camera_index: int, mode: str = "DEPTH", video_type: str = "mjpeg"):
    
    sensor_type = OBSensorType.DEPTH_SENSOR

    config = Config()
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)
    
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
        last_print_time = time.time()
        while True:
            try:
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue
                depth_format = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()

                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)
                # Apply temporal filtering
                depth_data = temporal_filter.process(depth_data)

                center_y = int(height / 2)
                center_x = int(width / 2)
                center_distance = depth_data[center_y, center_x]

                current_time = time.time()
                if current_time - last_print_time >= PRINT_INTERVAL:
                    # print("center distance: ", center_distance)
                    last_print_time = current_time

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

                frame = cv2.imencode('.jpg', depth_image)[1].tobytes()
                
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
    # python orb_net_device_server.py --host 0.0.0.0 --port 8082
    
    parser = argparse.ArgumentParser(description="ORB-NET Camera Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8082, help="Port to bind the server")
    parser.add_argument("--camera_index", type=int, default=4, help="Camera index")
    args = parser.parse_args()

    host = args.host
    port = args.port
    camera_index = args.camera_index

    start_camera_server(host, port, camera_index, video_type="mjpeg")

if __name__ == '__main__':
    main()
