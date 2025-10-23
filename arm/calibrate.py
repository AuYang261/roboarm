import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_control import Arm


def main():
    calibration_path = os.path.join(os.path.dirname(__file__), "..", "calibration")
    if os.path.exists(os.path.join(calibration_path, "arm_offset.txt")):
        os.rename(
            os.path.join(calibration_path, "arm_offset.txt"),
            os.path.join(calibration_path, "arm_offset_old.txt"),
        )
    arm = Arm()
    arm.disable_torque()
    while True:
        print(arm.get_read_arm_angles())
    arm.disconnect_arm()

if __name__ == "__main__":
    main()
