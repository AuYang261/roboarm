# Description: 按wasd/shift/space控制机械臂末端位置，esc退出
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_base import Arm
import numpy as np
import time
from pynput import keyboard
import threading

ROT_STEP_RAD = 0.1
MAX_ROT_RAD = np.pi / 2
TARGET_POSE: list[float] = [0, 0.1, 0.1, 0, 0]  # x, y, z, rot_rad, gripper_open_0to1


def on_press(key):
    global TARGET_POSE
    try:
        if key.char == "w" and TARGET_POSE[1] < 0.3:
            TARGET_POSE[1] += 0.01
            print("Current position:", TARGET_POSE)
        elif key.char == "s" and TARGET_POSE[1] > 0:
            TARGET_POSE[1] -= 0.01
            print("Current position:", TARGET_POSE)
        elif key.char == "a" and TARGET_POSE[0] > -0.2:
            TARGET_POSE[0] -= 0.01
            print("Current position:", TARGET_POSE)
        elif key.char == "d" and TARGET_POSE[0] < 0.2:
            TARGET_POSE[0] += 0.01
            print("Current position:", TARGET_POSE)
        elif key.char == "z" and TARGET_POSE[3] < MAX_ROT_RAD:
            TARGET_POSE[3] += ROT_STEP_RAD
            print("Current position:", TARGET_POSE)
        elif key.char == "c" and TARGET_POSE[3] > 0:
            TARGET_POSE[3] -= ROT_STEP_RAD
            print("Current position:", TARGET_POSE)
        elif key.char == "e":
            TARGET_POSE[4] = 0
            print("Current position:", TARGET_POSE)
        elif key.char == "q":
            TARGET_POSE[4] = 1
            print("Current position:", TARGET_POSE)
        elif key.char == "r":
            TARGET_POSE[:3] = np.random.uniform(
                [-0.2, 0, 0.07], [0.2, 0.3, 0.17]
            ).tolist()
            print("Current position:", TARGET_POSE)
    except AttributeError:
        if key == keyboard.Key.shift and TARGET_POSE[2] > 0.05:
            TARGET_POSE[2] -= 0.01
            print("Current position:", TARGET_POSE)
        elif key == keyboard.Key.space and TARGET_POSE[2] < 0.2:
            TARGET_POSE[2] += 0.01
            print("Current position:", TARGET_POSE)
        elif key == keyboard.Key.esc:
            print("Exiting...")
            TARGET_POSE = []


def on_release(key):
    pass


def get_input():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def main():
    global TARGET_POSE
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    arm = Arm()
    arm.move_to_home(gripper_open_0to1=1)
    pos = arm.get_arm_pos()
    if pos:
        TARGET_POSE[:3] = pos
    time.sleep(1)
    while True:
        try:
            if len(TARGET_POSE) != 5:
                arm.move_to_home(gripper_open_0to1=1)
                arm.disconnect_arm()
                return
            arm.move_to(
                TARGET_POSE[:3],
                gripper_open_0to1=TARGET_POSE[4],
                rot_rad=TARGET_POSE[3],
            )
        except Exception as e:
            print("Error:", e)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
