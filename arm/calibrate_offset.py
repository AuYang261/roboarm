# Description: 设置零点，即获取一组数据，让不同本体上零点的物理状态相同
# 1. roboarm
#    运行此脚本（若没标定会先进入标定），此脚本会显示当前机械臂各关节的角度值
#    把机械臂放到零位位置(见docs/image1.png)，然后读取各关节角度，作为offset保存下来
#    单位度，夹爪角度不需要
# 2. piper
#    通过.venv/lib/python3.10/site-packages/piper_sdk/demo/V2/piper_set_joint_zero.py脚本设置统一零点
#    要严格设置标准的零点位（J1零点校准刻度对齐为零点，J2J3当机械臂失能后自然垂下放置即为零点，J6轴线与J4共线状态下J5为零点）
#    以保证不同的本体零点相同，指定相同的位姿会有相同表现
#    piper机械臂下此脚本仅显示当前的x,y,z,RZ,RY,RX方便调试，非必需
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_base import Arm
import time
from utils.config_getter import get_config_value

def main():
    arm_type = get_config_value("arm_type")
    arm_backend = get_config_value("arm_backend", "real", raise_if_missing=False)
    if arm_type == "lerobo" and arm_backend == "sim":
        print("offset 标定仅适用于真机，arm_backend=sim 时无需也无法标定，已退出")
        return

    arm = Arm()
    arm.disable_torque()

    try:
        while True:
            print("=" * 10)
            try:
                # 需要原始接口获取真实原始数据，自己封装的高层接口是已考虑offset的修正数据
                if arm_type == "lerobo":
                    raw_angles_deg, _ = arm.get_raw_joint_angles()
                    if raw_angles_deg is None:
                        continue
                    for angle_deg in raw_angles_deg:
                        print(f"  - {angle_deg:.2f}")
                elif arm_type == "piper":
                    pos, rot = arm.get_arm_pose()
                    if pos:
                        for pos_i in pos:
                            print(f"  - {pos_i:.2f}")
                    if rot:
                        for rot_i in rot:
                            print(f"  - {rot_i:.2f}")
                else:
                    raise RuntimeError(f"Unknown arm_type {arm_type}")
            except Exception:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect_arm()


if __name__ == "__main__":
    main()
