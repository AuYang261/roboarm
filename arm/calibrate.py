# Description: 标定完后，运行此脚本，此脚本会显示当前机械臂各关节的角度值
# 把机械臂放到零位位置(见docs/image1.png)，然后读取各关节角度，作为offset保存下来
# 单位度，夹爪角度不需要
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_control import Arm


def main():
    calibration_path = os.path.join(os.path.dirname(__file__), "..", "calibration")
    if os.path.exists(os.path.join(calibration_path, "arm_offset.txt")):
        with open(os.path.join(calibration_path, "arm_offset.txt"), "w") as f:
            for i in range(5):
                f.write("0\n")
    arm = Arm()
    arm.disable_torque()
    while True:
        print(arm.get_read_arm_angles())
    arm.disconnect_arm()

if __name__ == "__main__":
    main()
