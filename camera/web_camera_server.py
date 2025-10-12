


import cv2


def start_mjpeg_server(host: str, port: int, camera_index: int):
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

def start_camera_server(host: str, port: int, camera_index: int, video_type: str = "mjpeg"):
    if video_type == "mjpeg":
        start_mjpeg_server(host, port, camera_index)
    else:
        print(f"Unsupported video type: {video_type}")

def main():
    host = "localhost"
    port = 8080
    camera_index = 4  # default camera index; change if needed

    start_camera_server(host, port, camera_index, video_type="mjpeg")

if __name__ == '__main__':
    main()
