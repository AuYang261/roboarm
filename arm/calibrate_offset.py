# Description: 设置零点，即获取一组数据，让不同本体上零点的物理状态相同
# 1. roboarm
#    运行此脚本（若没标定会先进入标定），此脚本会显示当前机械臂各关节的角度值
#    把机械臂放到零位位置(见docs/image1.png)，然后读取各关节角度，作为offset保存下来
#    单位度，夹爪角度不需要
# 2. piper
#    先通过.venv/lib/python3.10/site-packages/piper_sdk/demo/V2/piper_set_joint_zero.py脚本设置零点
#    然后运行此脚本
#   （暂不清楚零点不同的本体，指定相同的位姿是否会有相同表现，如果是则无需运行此脚本，待测试。目前是硬编码DEFAULT_DOWN_EULER_DEG）
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_base import Arm
import time
from config_getter import get_config_value


def main():
    arm_type = get_config_value("arm_type")
    if arm_type == "lerobo":
        from arm.lerobo_arm_control import LeroboArm

        arm: LeroboArm = Arm()  # type:ignore
        arm.disable_torque()
    else:
        arm = Arm()  # type:ignore
    while True:
        print("=" * 10)
        try:
            # 需要原始接口获取真实原始数据，自己封装的高层接口是已考虑offset的修正数据
            if arm_type == "lerobo":
                for angle_deg in list(arm.arm.get_observation().values())[:-1]:
                    print(f"  - {angle_deg:.2f}")
            # elif arm_type == "piper":
            #     for pose_i in arm.piper:
            #         print(f"  - {pose_i:.2f}")
            else:
                raise RuntimeError(f"Unknown arm_type {arm_type}")
        except Exception as e:
            pass
        time.sleep(0.1)
    arm.disconnect_arm()


if __name__ == "__main__":
    main()
