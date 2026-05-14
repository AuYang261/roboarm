import atexit
import os
import threading
from pathlib import Path
from typing import Callable, Any

import cv2
from flask import Flask, Response

from utils.config_getter import get_config_value

_HEADLESS: bool | None = True
_WINDOW_READY: set[str] = set()
_HEADLESS_SERVER_LOCK = threading.Lock()
_HEADLESS_SERVER_STARTED = False
_HEADLESS_TEMPLATE_PATH = Path(__file__).with_name("cv2_display.html")
_HEADLESS_SERVER_HOST = "0.0.0.0"
_HEADLESS_SERVER_PORT = get_config_value(
    "cv2_headless_port", 8765, raise_if_missing=False
)
_HEADLESS_JPEG_QUALITY = 90
_HEADLESS_LATEST_FRAMES: dict[str, bytes] = {}
_HEADLESS_FRAME_EVENTS: dict[str, threading.Event] = {}


def is_headless() -> bool:
    global _HEADLESS
    if _HEADLESS is not None:
        return _HEADLESS
    try:
        test_window = "__cv2_headless_probe__"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        _HEADLESS = False
    except cv2.error:
        _HEADLESS = True
    return _HEADLESS


def _frame_event(window_name: str) -> threading.Event:
    with _HEADLESS_SERVER_LOCK:
        return _HEADLESS_FRAME_EVENTS.setdefault(window_name, threading.Event())


def _headless_index_html() -> str:
    return _HEADLESS_TEMPLATE_PATH.read_text(encoding="utf-8")


def _run_headless_server() -> None:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return _headless_index_html()

    @app.get("/windows")
    def windows():
        with _HEADLESS_SERVER_LOCK:
            window_names = sorted(_HEADLESS_LATEST_FRAMES)
        return {"windows": window_names}

    @app.get("/stream/<path:window_name>")
    def stream(window_name: str):
        event = _frame_event(window_name)

        def generate():
            last_frame = None
            while True:
                if event.wait(timeout=30):
                    event.clear()
                frame = _HEADLESS_LATEST_FRAMES.get(window_name)
                if frame is None:
                    continue
                if frame == last_frame:
                    continue
                last_frame = frame
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    app.run(
        host=_HEADLESS_SERVER_HOST,
        port=_HEADLESS_SERVER_PORT,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


def _ensure_headless_server() -> None:
    global _HEADLESS_SERVER_STARTED
    if _HEADLESS_SERVER_STARTED:
        return
    with _HEADLESS_SERVER_LOCK:
        if _HEADLESS_SERVER_STARTED:
            return
        thread = threading.Thread(target=_run_headless_server, daemon=True)
        thread.start()
        _HEADLESS_SERVER_STARTED = True
        print(
            f"Headless OpenCV stream ready at http://{_HEADLESS_SERVER_HOST}:{_HEADLESS_SERVER_PORT}/ "
            f"(append ?window=<window_name>)"
        )


def _encode_headless_frame(image: Any) -> bytes | None:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), _HEADLESS_JPEG_QUALITY],
    )
    if not ok:
        return None
    return encoded.tobytes()


def _publish_headless_frame(window_name: str, image: Any) -> None:
    frame = _encode_headless_frame(image)
    if frame is None:
        return
    _HEADLESS_LATEST_FRAMES[window_name] = frame
    _frame_event(window_name).set()


def show_image(window_name: str, image: Any):
    if is_headless():
        _ensure_headless_server()
        _publish_headless_frame(window_name, image)
        return
    if window_name not in _WINDOW_READY:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        _WINDOW_READY.add(window_name)
    cv2.imshow(window_name, image)


def poll_key(delay: int = 1) -> int:
    if is_headless():
        return -1
    return cv2.waitKey(delay)


def set_mouse_callback(window_name: str, callback: Callable[..., Any]):
    if is_headless():
        return
    if window_name not in _WINDOW_READY:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        _WINDOW_READY.add(window_name)
    cv2.setMouseCallback(window_name, callback)


def destroy_all_windows():
    if is_headless():
        return
    cv2.destroyAllWindows()


def destroy_window(window_name: str):
    if is_headless():
        return
    cv2.destroyWindow(window_name)
    _WINDOW_READY.discard(window_name)


atexit.register(destroy_all_windows)
