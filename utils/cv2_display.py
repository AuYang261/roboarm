import os
from typing import Callable, Any
import cv2


_IMSHOW_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "cv2.imshow.png"
)
_HEADLESS: bool | None = None
_WINDOW_READY: set[str] = set()


def is_headless() -> bool:
    global _HEADLESS
    if _HEADLESS is not None:
        return _HEADLESS
    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        _HEADLESS = True
        return _HEADLESS
    try:
        test_window = "__cv2_headless_probe__"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        _HEADLESS = False
    except cv2.error:
        _HEADLESS = True
    return _HEADLESS


def show_image(window_name: str, image: Any):
    if is_headless():
        cv2.imwrite(_IMSHOW_OUTPUT_PATH, image)
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
