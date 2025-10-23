import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_control import Arm


def main():
    arm = Arm()
    arm.disable_torque()
    while True:
        print(arm.get_read_arm_angles())
    arm.disconnect_arm()

if __name__ == "__main__":
    main()
