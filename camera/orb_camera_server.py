# python camera/orb_camera_server.py --host 127.0.0.1 --port 8083 --mode multi

import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    Pipeline,
    OBSensorType,
    FrameSet,
    Context,
    OBFormat,
    OBError,
    OBAlignMode,
    OBFrameType,
)

from utils import frame_to_bgr_image
import argparse
import time, json, threading
from flask import Flask, Response

# cached frames for better visualization
cached_frames = {
    "color": None,
    "depth": None,
    "left_ir": None,
    "right_ir": None,
    "ir": None,
}


def setup_camera():
    """Setup camera and stream configuration"""
    pipeline = Pipeline()
    config = Config()
    device = pipeline.get_device()

    # Try to enable all possible sensors
    video_sensors = [
        OBSensorType.COLOR_SENSOR,
        OBSensorType.DEPTH_SENSOR,
        OBSensorType.IR_SENSOR,
        OBSensorType.LEFT_IR_SENSOR,
        OBSensorType.RIGHT_IR_SENSOR,
    ]
    sensor_list = device.get_sensor_list()
    for sensor in range(len(sensor_list)):
        try:
            sensor_type = sensor_list[sensor].get_type()
            if sensor_type in video_sensors:
                config.enable_stream(sensor_type)
        except:
            continue
        
    pipeline.start(config)
    return pipeline


def setup_imu():
    """Setup IMU configuration"""
    pipeline = Pipeline()
    config = Config()
    config.enable_accel_stream()
    config.enable_gyro_stream()
    pipeline.start(config)
    return pipeline

def hw_d2c_align_stream_config(pipeline: Pipeline):
    """
    Gets the stream configuration for the pipeline.

    Args:
        pipeline (Pipeline): The pipeline object.

    Returns:
        Config: The stream configuration.
    """
    config = Config()
    try:
        # Get the list of color stream profiles
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        assert profile_list is not None
        
        # Iterate through the color stream profiles
        for i in range(len(profile_list)):
            color_profile = profile_list[i]
            
            # Check if the color format is RGB
            if color_profile.get_format() != OBFormat.RGB:
                continue
            
            # Get the list of hardware aligned depth-to-color profiles
            hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            if len(hw_d2c_profile_list) == 0:
                continue
            
            # Get the first hardware aligned depth-to-color profile
            hw_d2c_profile = hw_d2c_profile_list[0]
            print("hw_d2c_profile: ", hw_d2c_profile)
            
            # Enable the depth and color streams
            config.enable_stream(hw_d2c_profile)
            config.enable_stream(color_profile)
            
            # Set the alignment mode to hardware alignment
            config.set_align_mode(OBAlignMode.HW_MODE)
            return config
    except Exception as e:
        print(e)
        return None
    return None

def process_d2c(color_frame, depth_frame, min_depth=20, max_depth=10000) -> dict:

    if not color_frame or not depth_frame:
        # print("process_d2c: missing color_frame or depth_frame")
        return None
    depth_format = depth_frame.get_format()
    if depth_format != OBFormat.Y16:
        print("depth format is not Y16")
        return None

    # Convert the color frame to a BGR image
    color_image = frame_to_bgr_image(color_frame)
    if color_image is None:
        print("Failed to convert frame to image")
        return None

    # Get the depth data
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        (depth_frame.get_height(), depth_frame.get_width()))
    
    # Convert depth data to float32 and apply depth scale
    depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
    
    # Apply custom depth range, clip depth data
    depth_data = np.clip(depth_data, min_depth, max_depth)
    
    # Normalize depth data for display
    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)

    return {"color": color_image, "depth": depth_image}

def get_depth_data(depth_frame, min_depth=20, max_depth=10000):
    if not depth_frame:
        # print("process_d2c: missing color_frame or depth_frame")
        return None
    depth_format = depth_frame.get_format()
    if depth_format != OBFormat.Y16:
        print("depth format is not Y16")
        return None

    # Get the depth data
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        (depth_frame.get_height(), depth_frame.get_width()))
    
    # Convert depth data to float32 and apply depth scale
    depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
    
    # Apply custom depth range, clip depth data
    depth_data = np.clip(depth_data, min_depth, max_depth)
    
    return depth_data

def process_color(frame):
    """Process color image"""
    frame = frame if frame else cached_frames["color"]
    cached_frames["color"] = frame
    return frame_to_bgr_image(frame) if frame else None


def process_depth(frame):
    """Process depth image"""
    frame = frame if frame else cached_frames["depth"]
    cached_frames["depth"] = frame
    if not frame:
        return None
    try:
        depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape(frame.get_height(), frame.get_width())
        depth_image = cv2.normalize(
            depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    except ValueError:
        return None


def process_ir(ir_frame):
    """Process IR frame (left or right) to RGB image"""
    if ir_frame is None:
        return None
    ir_frame = ir_frame.as_video_frame()
    ir_data = np.asanyarray(ir_frame.get_data())
    width = ir_frame.get_width()
    height = ir_frame.get_height()
    ir_format = ir_frame.get_format()

    if ir_format == OBFormat.Y8:
        ir_data = np.resize(ir_data, (height, width, 1))
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
    elif ir_format == OBFormat.MJPG:
        ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
        data_type = np.uint8
        image_dtype = cv2.CV_8UC1
        max_data = 255
        if ir_data is None:
            print("decode mjpeg failed")
            return None
        ir_data = np.resize(ir_data, (height, width, 1))
    else:
        ir_data = np.frombuffer(ir_data, dtype=np.uint16)
        data_type = np.uint16
        image_dtype = cv2.CV_16UC1
        max_data = 255
        ir_data = np.resize(ir_data, (height, width, 1))

    cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
    ir_data = ir_data.astype(data_type)
    return cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)


def get_imu_text(frame, name):
    """Format IMU data"""
    if not frame:
        return []
    return [
        f"{name} x: {frame.get_x():.2f}",
        f"{name} y: {frame.get_y():.2f}",
        f"{name} z: {frame.get_z():.2f}",
    ]

def create_display(frames, width=1280, height=720):
    """Create display window"""
    display = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = height // 2, width // 2

    # Process video frames
    if 'color' in frames and frames['color'] is not None:
        display[0:h, 0:w] = cv2.resize(frames['color'], (w, h))

    if 'depth' in frames and frames['depth'] is not None:
        display[0:h, w:] = cv2.resize(frames['depth'], (w, h))

    if 'ir' in frames and frames['ir'] is not None:
        display[h:, 0:w] = cv2.resize(frames['ir'], (w, h))

    # Display IMU data
    if 'imu' in frames:
        y_offset = h + 20
        for data_type in ['accel', 'gyro']:
            text_lines = get_imu_text(frames['imu'].get(data_type), data_type.title())
            for i, line in enumerate(text_lines):
                cv2.putText(display, line, (w + 10, y_offset + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 80

    return display


def display_stream():
    # Window settings
    WINDOW_NAME = "MultiStream Viewer"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    # Initialize camera
    pipeline = setup_camera()
    imu_pipeline = setup_imu()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    while True:
        # Get all frames
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue
        # Process different frame types
        processed_frames = {'color': process_color(frames.get_color_frame()),
                            'depth': process_depth(frames.get_depth_frame())}

        # Process IR image: try stereo IR first, fallback to mono if unavailable
        try:
            left = process_ir(frames.get_frame(OBFrameType.LEFT_IR_FRAME).as_video_frame())
            right = process_ir(frames.get_frame(OBFrameType.RIGHT_IR_FRAME).as_video_frame())
            if left is not None and right is not None:
                processed_frames['ir'] = np.hstack((left, right))
        except:
            ir_frame = frames.get_ir_frame()
            if ir_frame:
                processed_frames['ir'] = process_ir(ir_frame.as_video_frame())

        # Process IMU data
        imu_frames = imu_pipeline.wait_for_frames(100)
        if not imu_frames:
            continue
        accel = imu_frames.get_frame(OBFrameType.ACCEL_FRAME)
        gyro = imu_frames.get_frame(OBFrameType.GYRO_FRAME)
        if accel and gyro:
            processed_frames['imu'] = {
                'accel': accel.as_accel_frame(),
                'gyro': gyro.as_gyro_frame()
            }

        # create display
        display = create_display(processed_frames, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.imshow(WINDOW_NAME, display)

        # check exit key
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # q or ESC
            break

    pipeline.stop()
    cv2.destroyAllWindows()

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(
                frame, self.alpha, self.previous_frame, 1 - self.alpha, 0
            )
        self.previous_frame = result
        return result

# 存在问题，传输速度过慢
def start_dict_server(host, port):

    app = Flask(__name__)

    # Initialize camera
    pipeline = setup_camera()
    imu_pipeline = setup_imu()

    def dict_generator():
        idx = 0
        while True:
            # Get all frames
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue
            # Process different frame types
            processed_frames = {'color': process_color(frames.get_color_frame()),
                                'depth': process_depth(frames.get_depth_frame())}

            # Process IR image: try stereo IR first, fallback to mono if unavailable
            try:
                left = process_ir(frames.get_frame(OBFrameType.LEFT_IR_FRAME).as_video_frame())
                right = process_ir(frames.get_frame(OBFrameType.RIGHT_IR_FRAME).as_video_frame())
                if left is not None and right is not None:
                    processed_frames['ir'] = np.hstack((left, right))
            except:
                ir_frame = frames.get_ir_frame()
                if ir_frame:
                    processed_frames['ir'] = process_ir(ir_frame.as_video_frame())

            # Process IMU data
            imu_frames = imu_pipeline.wait_for_frames(100)
            if not imu_frames:
                continue
            accel = imu_frames.get_frame(OBFrameType.ACCEL_FRAME)
            gyro = imu_frames.get_frame(OBFrameType.GYRO_FRAME)
            if accel and gyro:
                processed_frames['imu'] = {
                    'accel': accel.as_accel_frame(),
                    'gyro': gyro.as_gyro_frame()
                }

            # create display
            # display = create_display(processed_frames, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            # cv2.imshow(WINDOW_NAME, display)

            idx += 1
            data = {
                    "ts": time.time(),
                    "idx": idx, 
                    "rgb": processed_frames.get("color").tolist(),
                    "depth": processed_frames.get("depth").tolist()
                }
            #  multipart 需要 \r\n 边界
            yield (b'--dictboundary\r\n'
                b'Content-Type: application/json\r\n\r\n'
                + json.dumps(data).encode() +
                b'\r\n')

    @app.route('/dict_stream')
    def dict_stream():
        return Response(dict_generator(),
                        mimetype='multipart/x-mixed-replace; boundary=dictboundary')

    app.run(host=host, port=port, threaded=True, use_reloader=False)

    pipeline.stop()

def start_rgb_server(host, port):
    
    sensor_type = OBSensorType.COLOR_SENSOR
     
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
        
    @app.route('/rgb_stream')
    def rgb_stream():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    pipeline.stop()

def start_depth_server(host, port):
    
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
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(
                1280, 720, OBFormat.RGB, 30
            )
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

                # 计算中心点距离
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
                
                # 返回一段符合 MJPEG（multipart/x-mixed-replace）协议的数据片段
                # 代表“一个 JPEG 帧”，由 Flask 的 Response 直接写入 HTTP 响应流
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

global_processed_frames = None
global_ir_frame = None
global_imu_data = None
metadata_lock = threading.Lock()

def frame_generator(pipeline, imu_pipeline):
    idx = 0
    while True:
        # Get all frames
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue
        # Process different frame types 将图像放在 processed_frames 中
        processed_frames = {'color': process_color(frames.get_color_frame()),
                            'depth': process_depth(frames.get_depth_frame())}

        # Process IR image: try stereo IR first, fallback to mono if unavailable
        try:
            left = process_ir(frames.get_frame(OBFrameType.LEFT_IR_FRAME).as_video_frame())
            right = process_ir(frames.get_frame(OBFrameType.RIGHT_IR_FRAME).as_video_frame())
            if left is not None and right is not None:
                processed_frames['ir'] = np.hstack((left, right))
        except:
            ir_frame = frames.get_ir_frame()
            if ir_frame:
                processed_frames['ir'] = process_ir(ir_frame.as_video_frame())

        # Process IMU data
        imu_frames = imu_pipeline.wait_for_frames(100)
        if not imu_frames:
            continue
        accel = imu_frames.get_frame(OBFrameType.ACCEL_FRAME)
        gyro = imu_frames.get_frame(OBFrameType.GYRO_FRAME)
        if accel and gyro:
            processed_frames['imu'] = {
                'accel': accel.as_accel_frame(),
                'gyro': gyro.as_gyro_frame()
            }
        
        # 获取 点云 pointcloud
        

        # 更新 共享数据
        with metadata_lock:
            global global_processed_frames
            global global_ir_frame
            global_processed_frames = processed_frames
            global_ir_frame = ir_frame

def start_multi_server(host, port):
    # Initialize camera
    pipeline = setup_camera()
    imu_pipeline = setup_imu()
    
    app = Flask(__name__)
    
    def rgb_generator():
        while True:
            with metadata_lock:
                if global_processed_frames is None or global_processed_frames.get("color") is None:
                    continue
                color_image = global_processed_frames.get("color")
            frame = cv2.imencode('.jpg', color_image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    def depth_generator():
        while True:
            with metadata_lock:
                if global_processed_frames is None or global_processed_frames.get("depth") is None:
                    continue
                depth_image = global_processed_frames.get("depth")
            frame = cv2.imencode('.jpg', depth_image)[1].tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    @app.route('/rgb_stream')
    def rgb_stream():
        return Response(rgb_generator(),
                        mimetype='multipart/x-mixed-replace; boundary=dictboundary')
        
    @app.route('/depth_stream')
    def depth_stream():
        return Response(depth_generator(),
                        mimetype='multipart/x-mixed-replace; boundary=dictboundary')

    threading.Thread(target=frame_generator, args=(pipeline, imu_pipeline), daemon=True).start()
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    
    pipeline.stop()

def start_usb_camera_server(host: str, port: int, camera_index: int):
    from flask import Flask, Response

    app = Flask(__name__)
    cap = cv2.VideoCapture(camera_index)

    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host=host, port=port)

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="ORB-NET Camera Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument(
        "--port", type=int, default=8083, help="Port to bind the server"
    )
    parser.add_argument("--idx", type=int, default=4, help="camera index")
    parser.add_argument("--mode", type=str, default="multi", help="Video source (default: rgb)")
    args = parser.parse_args()
    if args.mode == "none":
        display_stream()
    elif args.mode == "dict":
        start_dict_server(args.host, args.port)
    elif args.mode == "rgb":
        start_rgb_server(args.host, args.port)
    elif args.mode == "depth":
        start_depth_server(args.host, args.port)
    elif args.mode == "multi":
        start_multi_server(args.host, args.port)
    elif args.mode == "usb":
        start_usb_camera_server(args.host, args.port, args.idx)


if __name__ == "__main__":
    main()
