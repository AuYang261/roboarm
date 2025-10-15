import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera import orb_camera
from camera import orb_camera_client

import cv2
import argparse


class Camera:
    def __init__(self, ip: str = "", port: int = 8083, color: bool = True, depth: bool = False):
        self.ip = ip
        self.port = port
        self.color = color
        self.depth = depth
        self.pipeline = None
        self.cap_rgb = None
        self.cap_depth = None
        if ip == "":
            self.pipeline = orb_camera.open_camera(color, depth)
        else:
            self.cap_rgb, self.cap_depth = orb_camera_client.open_orb_web_camera(ip, port, color, depth)

    def get_frames(self):
        if self.ip == "":
            return orb_camera.get_frames(self.pipeline)
        else:
            frame_rgb = None
            frame_depth = None
            
            if self.cap_rgb is not None:
                ret, frame_rgb = self.cap_rgb.read()
            if self.cap_depth is not None:
                ret, frame_depth = self.cap_depth.read()
            
            return frame_rgb, frame_depth

    def close(self):
        if self.ip == "":
            orb_camera.close_camera(self.pipeline)
        else:
            orb_camera_client.close_orb_web_camera(self.cap)



def main():

    camera = Camera(ip="127.0.0.1", port=8081, color=True, depth=False)

    frame_rgb, frame_depth = camera.get_frames()
    
    cv2.namedWindow('Camera Client', cv2.WINDOW_NORMAL)
    cv2.imshow('Camera Client', frame_rgb if frame_rgb is not None else frame_depth)
    
    
    
    print("Press 'q' to exit")
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()