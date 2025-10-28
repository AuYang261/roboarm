# Description: 标定完后，运行此脚本，此脚本会显示当前机械臂各关节的角度值
# 把机械臂放到零位位置(见docs/image1.png)，然后读取各关节角度，作为offset保存下来
# 单位度，夹爪角度不需要
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_control import Arm


def main():
    arm = Arm()
    arm.disable_torque()
    while True:
        print("=" * 10)
        try:
            for angle_deg in list(arm.arm.get_observation().values())[:-1]:
                print(f"  - {angle_deg:.2f}")
        except Exception as e:
            pass
    arm.disconnect_arm()


if __name__ == "__main__":
    main()
