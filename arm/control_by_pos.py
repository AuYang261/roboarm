# Description: 按wasd/z/x/u/j/q/e控制机械臂末端位置，esc退出
import sys
import os
import select

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_base import Arm
import numpy as np
import time
import threading

ROT_STEP_RAD = 0.1
MAX_ROT_RAD = np.pi / 2
TARGET_POSE = np.array(
    [0, 0.1, 0.1, 0, 0, 0, 0]
)  # x, y, z, RZ, RY, RX, gripper_open_0to1


def handle_key(key):
    global TARGET_POSE
    if key == "w":
        TARGET_POSE[1] += 0.01
        print("Current position:", TARGET_POSE.round(2))
    elif key == "s":
        TARGET_POSE[1] -= 0.01
        print("Current position:", TARGET_POSE.round(2))
    elif key == "a":
        TARGET_POSE[0] -= 0.01
        print("Current position:", TARGET_POSE.round(2))
    elif key == "d":
        TARGET_POSE[0] += 0.01
        print("Current position:", TARGET_POSE.round(2))
    elif key == "z":
        TARGET_POSE[2] -= 0.01
        print("Current position:", TARGET_POSE.round(2))
    elif key == "x":
        TARGET_POSE[2] += 0.01
        print("Current position:", TARGET_POSE.round(2))
    elif key == "u":
        TARGET_POSE[3] += ROT_STEP_RAD
        print("Current position:", TARGET_POSE.round(2))
    elif key == "j":
        TARGET_POSE[3] -= ROT_STEP_RAD
        print("Current position:", TARGET_POSE.round(2))
    elif key == "i":
        TARGET_POSE[4] += ROT_STEP_RAD
        print("Current position:", TARGET_POSE.round(2))
    elif key == "k":
        TARGET_POSE[4] -= ROT_STEP_RAD
        print("Current position:", TARGET_POSE.round(2))
    elif key == "o":
        TARGET_POSE[5] += ROT_STEP_RAD
        print("Current position:", TARGET_POSE.round(2))
    elif key == "l":
        TARGET_POSE[5] -= ROT_STEP_RAD
        print("Current position:", TARGET_POSE.round(2))
    elif key == "e":
        TARGET_POSE[6] = 0
        print("Current position:", TARGET_POSE.round(2))
    elif key == "q":
        TARGET_POSE[6] = 1
        print("Current position:", TARGET_POSE.round(2))
    elif key == "r":
        TARGET_POSE[:3] = np.random.uniform([-0.2, 0, 0.07], [0.2, 0.3, 0.17]).tolist()
        print("Current position:", TARGET_POSE.round(2))
    elif key == "\x1b":
        print("Exiting...")
        TARGET_POSE = np.empty([0])


def read_key_posix(timeout=0.1):
    import importlib

    termios = importlib.import_module("termios")
    tty = importlib.import_module("tty")

    fd = sys.stdin.fileno()
    tcgetattr = getattr(termios, "tcgetattr")
    tcsetattr = getattr(termios, "tcsetattr")
    tcsadrain = getattr(termios, "TCSADRAIN")
    setraw = getattr(tty, "setraw")
    old_settings = tcgetattr(fd)
    try:
        setraw(fd)
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        key = sys.stdin.read(1)
        if key == "\x1b":
            readable, _, _ = select.select([sys.stdin], [], [], 0.01)
            if readable:
                key += sys.stdin.read(1)
                readable, _, _ = select.select([sys.stdin], [], [], 0.01)
                if readable:
                    key += sys.stdin.read(1)
        return key
    finally:
        tcsetattr(fd, tcsadrain, old_settings)


def read_key(timeout=0.1):
    if os.name == "nt":
        import msvcrt

        if not msvcrt.kbhit():
            time.sleep(timeout)
            return None
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            key += msvcrt.getwch()
        return key

    return read_key_posix(timeout)


def get_input():
    while True:
        key = read_key()
        if key is None:
            continue
        handle_key(key)
        if len(TARGET_POSE) != 7:
            return


def main():
    global TARGET_POSE
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    arm = Arm()
    arm.move_to_home(gripper_open_0to1=1)
    pos, rot = arm.get_arm_pose()
    if pos:
        TARGET_POSE[:3] = np.array(pos)
        # TARGET_POSE[2] += 0.05
    if rot:
        TARGET_POSE[3:6] = np.deg2rad(rot)
    while True:
        try:
            if len(TARGET_POSE) != 7:
                arm.move_to_home(gripper_open_0to1=1)
                arm.disconnect_arm()
                return
            arm.move_to(
                TARGET_POSE[:3].tolist(),
                gripper_open_0to1=TARGET_POSE[6],
                rot_rad=TARGET_POSE[3],
                euler_angles_deg_zyx=np.rad2deg(TARGET_POSE[3:6]).tolist(),
            )
        except Exception as e:
            print("Error:", e)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
