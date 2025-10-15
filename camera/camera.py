import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera import orb_camera
from camera import orb_camera_client


class Camera:
    def __init__(self, ip: str = "", color: bool = True, depth: bool = False):
        self.ip = ip
        self.color = color
        self.depth = depth
        if ip == "":
            self.pipeline = orb_camera.open_camera(color, depth)
        else:
            pass

    def get_frames(self) -> dict:
        if self.ip == "":
            return orb_camera.get_frames(self.pipeline)
        else:
            return {"color": None, "depth": None}

    def close(self):
        if self.ip == "":
            orb_camera.close_camera(self.pipeline)
        else:
            pass
