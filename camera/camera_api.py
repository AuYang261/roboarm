import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
from utils.config_getter import get_config_value
from utils.cv2_display import show_image, poll_key, destroy_all_windows

class Camera:

    def __init__(
        self,
        color: bool = True,
        depth: bool = False,
    ):
        self.ip = get_config_value("camera_ip", "", False)
        self.port = get_config_value("camera_port", None, False)
        self.color = color
        self.depth = depth
        self.pipeline = None
        self.cap_rgb = None
        self.cap_depth = None
        if not self.ip:
            from camera import orb_camera

            self.orb_camera = orb_camera
            self.pipeline = self.orb_camera.open_camera(self.color, self.depth)
        else:
            if self.port is None:
                raise ValueError("未指定远程相机端口号")
            from camera import orb_camera_client

            self.orb_camera_client = orb_camera_client
            self.cap_rgb, self.cap_depth = self.orb_camera_client.open_orb_web_camera(
                self.ip, self.port, self.color, self.depth
            )

    def get_frames(self) -> dict[str, cv2.typing.MatLike | None]:
        if not self.ip:
            return self.orb_camera.get_frames(self.pipeline)
        else:
            frame_rgb = None
            frame_depth = None

            if self.cap_rgb is not None:
                ret, frame_rgb = self.cap_rgb.read()
            if self.cap_depth is not None:
                ret, frame_depth = self.cap_depth.read()
            return {"color": frame_rgb, "depth": frame_depth}

    def close(self):
        if not self.ip:
            self.orb_camera.close_camera(self.pipeline)
        else:
            self.orb_camera_client.close_orb_web_camera(self.cap_rgb, self.cap_depth)


def main():

    camera = Camera(color=True, depth=False)

    frames = camera.get_frames()
    frame_rgb = frames.get("color")
    frame_depth = frames.get("depth")

    show_image("Camera Client", frame_rgb if frame_rgb is not None else frame_depth)

    print("Press 'q' to exit")
    while True:
        key = poll_key(1)
        if key == ord('q'):
            break

    destroy_all_windows()


if __name__ == "__main__":
    main()
