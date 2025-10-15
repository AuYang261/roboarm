# Description: 按wasd/shift/space控制机械臂末端位置，esc退出
import sys
import os

from sympy import im

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_control import Arm
import numpy as np
from typing import Sequence, List, Union
import time
from pynput import keyboard
import threading

POS: list[float] = [0, 0.1, 0.1, 0, 0]  # x, y, z, gripper_angle_deg


def on_press(key):
    global POS
    try:
        if key.char == "w" and POS[1] < 0.3:
            POS[1] += 0.01
            print("Current position:", POS)
        elif key.char == "s" and POS[1] > 0:
            POS[1] -= 0.01
            print("Current position:", POS)
        elif key.char == "a" and POS[0] > -0.2:
            POS[0] -= 0.01
            print("Current position:", POS)
        elif key.char == "d" and POS[0] < 0.2:
            POS[0] += 0.01
            print("Current position:", POS)
        elif key.char == "z" and POS[3] < 90:
            POS[3] += 5
            print("Current position:", POS)
        elif key.char == "c" and POS[3] > 0:
            POS[3] -= 5
            print("Current position:", POS)
        elif key.char == "e":
            POS[4] = 0
            print("Current position:", POS)
        elif key.char == "q":
            POS[4] = 80
            print("Current position:", POS)
    except AttributeError:
        if key == keyboard.Key.shift and POS[2] > 0.05:
            POS[2] -= 0.01
            print("Current position:", POS)
        elif key == keyboard.Key.space and POS[2] < 0.2:
            POS[2] += 0.01
            print("Current position:", POS)
        elif key == keyboard.Key.esc:
            print("Exiting...")
            POS = []


def on_release(key):
    pass


def get_input():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def main():
    global POS
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    arm = Arm("COM3")
    POS[:3] = arm.move_to_home(gripper_angle_deg=80).pos  # type: ignore
    time.sleep(1)
    while True:
        try:
            if len(POS) == 0:
                arm.move_to_home(gripper_angle_deg=80)
                arm.disconnect_arm()
                exit(0)
            arm.move_to(
                POS[:3],
                gripper_angle_deg=POS[4],
                rot_rad=np.deg2rad(POS[3]),
                warning=False,
            )
            time.sleep(0.1)
        except Exception as e:
            print("Error:", e)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
