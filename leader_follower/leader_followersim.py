"""
Leader-Follower Simulation Demo
物理主机械臂 (leader) 控制 MuJoCo 仿真机械臂 (follower)
读取 leader 的关节位置，转换为弧度后驱动仿真模型
"""

import sys
import os
import importlib.util

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lerobot", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import time
import json
import argparse
import yaml
import mujoco
import numpy as np

# mujoco-sim.py 文件名含连字符，用 importlib 加载
_sim_path = os.path.join(os.path.dirname(__file__), "..", "sim", "mujoco-sim.py")
_spec = importlib.util.spec_from_file_location("mujoco_sim", _sim_path)
_mujoco_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mujoco_sim)

SimMujocoModel = _mujoco_sim.SimMujocoModel
action2rad = _mujoco_sim.action2rad
_apply_pd_control = _mujoco_sim._apply_pd_control


def main():
    parser = argparse.ArgumentParser(description="Leader-Follower Simulation Demo")
    parser.add_argument(
        "--leader_calibration",
        type=str,
        default="calibration/koch_follower.json",
        help="Leader 机械臂校准文件路径",
    )
    parser.add_argument(
        "--sim_calibration",
        type=str,
        default="calibration/koch_follower.json",
        help="仿真 action->rad 转换使用的校准文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="urdf/meshes/mjmodel.xml",
        help="MuJoCo 模型 XML 文件路径",
    )
    parser.add_argument(
        "--use_degrees",
        action="store_true",
        default=False,
        help="Leader 机械臂是否使用 DEGREES 归一化模式",
    )
    args = parser.parse_args()

    # 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        print("Failed to load config.yaml")
        return

    leader_port = config.get("arm_port", None)
    if leader_port is None:
        raise ValueError("配置文件中没有设置主机械臂端口号 leader_port")

    # 初始化 leader 物理机械臂（只读，不需要 torque）
    leader_arm = DynamixelMotorsBus(
        port=leader_port,
        motors={
            "shoulder_pan": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
        },
    )

    leader_arm.connect()

    # 加载 leader 校准数据
    calibration_data = json.load(open(args.leader_calibration, "r"))
    for motor_name, calib in calibration_data.items():
        calibration_data[motor_name] = MotorCalibration(**calib)
    leader_arm.write_calibration(calibration_dict=calibration_data)

    print("Leader 机械臂已连接")

    # 初始化 MuJoCo 仿真模型
    sim = SimMujocoModel(args.model_path)
    sim.add_view("default", {"distance": 3.0, "lookat": [0.0, 0.0, 0.0], "elevation": -20.0, "azimuth": 135.0})
    sim.add_view("top", {"distance": 2.0, "lookat": [0.0, 0.0, 0.0], "elevation": -90.0, "azimuth": 0.0})
    sim.add_view("front", {"distance": 2.5, "lookat": [0.0, 0.0, 0.0], "elevation": 0.0, "azimuth": 0.0})
    sim.add_view("side", {"distance": 2.5, "lookat": [0.0, 0.0, 0.0], "elevation": 0.0, "azimuth": 90.0})
    sim.switch_view("default")

    print("MuJoCo 仿真已启动")
    print("\n=== Leader-Follower Simulation ===")
    print("移动物理 leader 机械臂，仿真模型会跟随移动")
    print("关闭仿真窗口退出\n")

    # 关节名称顺序（与 action2rad 一致）
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    frequency = 100  # Hz
    frame_count = 0

    try:
        while sim.viewer.is_running():
            try:
                # 读取 leader 各关节归一化位置
                leader_pos = {}
                for motor in leader_arm.motors.keys():
                    leader_pos[motor] = leader_arm.read("Present_Position", motor=motor)

                # 按关节顺序组装 action 列表
                action = [leader_pos[name] for name in joint_names]

                # 转换为弧度
                rad_angles = action2rad(action, calibration_file=args.sim_calibration, use_degrees=args.use_degrees)

                # 设置仿真关节角度
                sim.set_joint_angles(rad_angles)

                # 仿真步进
                sim.update_camera()
                _apply_pd_control(sim)
                mujoco.mj_step(sim.model, sim.data)
                sim.viewer.sync()

                frame_count += 1
                if frame_count % 200 == 0:
                    action_str = ", ".join(f"{v:.1f}" for v in action)
                    rad_str = ", ".join(f"{v:.3f}" for v in rad_angles)
                    print(f"[{frame_count}] action: [{action_str}]")
                    print(f"         rad: [{rad_str}]")

                time.sleep(1 / frequency)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n用户中断")

    # 清理
    leader_arm.disconnect()
    sim.close()
    print("已退出")


if __name__ == "__main__":
    main()
