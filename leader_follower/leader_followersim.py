"""
Leader-Follower Simulation Demo
物理主机械臂 (leader) 控制 MuJoCo 仿真机械臂 (follower)
读取 leader 的关节位置，转换为弧度后驱动仿真模型

键盘控制情况下会出现关节异常移动的现象（如仅控制夹住移动但关节还是移动了的情况）

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

REMOTE = True

IP = "192.168.2.12"
PORT = 3456
MODE = "pos" # pos 位置控制 or tor位姿控制

STEPS = 1
KEY_BOARD = False # 通过键盘控制


# mujoco-sim.py 文件名含连字符，用 importlib 加载
from sim.mujoco_sim import SimMujocoModel, action2rad, _apply_pd_control, KeyboardController

def step_towards(current, target, step_size = 10) -> dict:
    result = {}
    for key in target.keys():
        curr_value = current[key]
        target_value = target[key]
        if abs(target_value - curr_value) <= step_size:
            result[key] = target_value
        else:
            if target_value > curr_value:
                result[key] = curr_value + step_size
            else:
                result[key] = curr_value - step_size
    return result   

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

    
    # 如果键盘控制就使能
    if KEY_BOARD:
        kb = KeyboardController().start()
        print("\n=== 键盘控制 ===")
        print("上/下键: 切换关节  左/右键: 调整角度  空格: 重置  ESC: 退出\n")
        leader_arm.enable_torque()

    

    print("Leader 机械臂已连接")

    # 初始化 MuJoCo 仿真模型
    sim = SimMujocoModel(args.model_path)
    sim.add_view("default", {"distance": 3.0, "lookat": [0.0, 0.0, 0.0], "elevation": -20.0, "azimuth": 135.0})
    sim.add_view("top", {"distance": 2.0, "lookat": [0.0, 0.0, 0.0], "elevation": -90.0, "azimuth": 0.0})
    sim.add_view("front", {"distance": 2.5, "lookat": [0.0, 0.0, 0.0], "elevation": 0.0, "azimuth": 0.0})
    sim.add_view("side", {"distance": 2.5, "lookat": [0.0, 0.0, 0.0], "elevation": 0.0, "azimuth": 90.0})
    sim.switch_view("default")

    sim.add_obj()

    print("MuJoCo 仿真已启动")
    print("\n=== Leader-Follower Simulation ===")
    print("移动物理 leader 机械臂，仿真模型会跟随移动")
    print("关闭仿真窗口退出\n")

    # 关节名称顺序（与 action2rad 一致）
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    frequency = 100  # Hz
    frame_count = 0

    current_joint = 0
    prev_up = prev_down = prev_space = False

    try:
        while sim.viewer.is_running():
            try:
                # 读取 leader 各关节归一化位置
                leader_pos = {}
                for motor in leader_arm.motors.keys():
                    leader_pos[motor] = leader_arm.read("Present_Position", motor=motor)
                    # 读取的原始数据 - 归一化后的

                moving = False
                if KEY_BOARD:
                    up = kb.get('up')
                    down = kb.get('down')
                    space = kb.get(' ')

                    # 上升沿检测：切换关节
                    if up and not prev_up:
                        current_joint = (current_joint - 1 + 6) % 6
                        current_motor = joint_names[current_joint]
                        print(f"选中关节 {current_joint} ({current_motor}): {leader_pos[current_motor]:.2f}")
                    if down and not prev_down:
                        current_joint = (current_joint + 1) % 6
                        current_motor = joint_names[current_joint]
                        print(f"选中关节 {current_joint} ({current_motor}): {leader_pos[current_motor]:.2f}")

                    # 空格重置到初始位置
                    if space and not prev_space:
                        print("重置仿真位置")
                        mujoco.mj_resetData(sim.model, sim.data)
                        sim.target_joint_positions = sim.data.qpos[:sim.model.nu].copy()

                    # 每次对 action 中当前关节的值进行调整
                    current_motor = joint_names[current_joint]
                    if kb.get('left'):
                        leader_pos[current_motor] += STEPS
                        moving = True
                    if kb.get('right'):
                        leader_pos[current_motor] -= STEPS
                        moving = True

                    prev_up = up
                    prev_down = down
                    prev_space = space

                # 按关节顺序组装 action 列表
                action = [leader_pos[name] for name in joint_names]

                # 转换为弧度
                rad_angles = action2rad(action, calibration_file=args.sim_calibration, use_degrees=args.use_degrees)

                if KEY_BOARD and moving:
                    # 打印当前 action 和转换后的 rad
                    action_str = ", ".join(f"{v:.2f}" for v in action)
                    rad_str = ", ".join(f"{v:.3f}" for v in rad_angles)
                    print(f"action: [{action_str}]")
                    print(f"rad:    [{rad_str}]")
                    # 写入到电机中
                    for motor, pos in leader_pos.items():
                        leader_arm.write("Goal_Position", motor, pos)

                # 设置仿真关节角度
                sim.set_joint_angles(rad_angles)

                # 仿真步进
                sim.update_camera()
                _apply_pd_control(sim)
                mujoco.mj_step(sim.model, sim.data)
                sim.viewer.sync()
                if REMOTE:
                    # 不断发送数据，有程序来可以监听这里发送的数据并解析
                    import socket
                    # UDP 发送仿真关节角度
                    if not hasattr(main, "_sock"):
                        main._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        main._addr = (IP, PORT)
                    # 发送内容为 json 字符串
                    # 如果 是 位姿 控制模式 ， 计算出位姿数据并发送
                    
                    if MODE == "tor":
                        # 计算末端执行器位姿
                        poses = sim.get_current_poses()
                        pos, quat = poses['last_joint']['pos'], poses['last_joint']['quat']
                        gripper_angle = rad_angles[5]  # gripper 是第6个关节
                        # print(f"发送位姿数据: pos={pos}, quat={quat}, angle={gripper_angle}")
                        send_data = json.dumps({"position": pos, "orientation": quat, "gripper_angle": gripper_angle}).encode("utf-8")
                    else:
                        send_data = json.dumps({"rad_angles": rad_angles}).encode("utf-8")
                    main._sock.sendto(send_data, main._addr)

                frame_count += 1
                # if frame_count % 200 == 0:
                #     action_str = ", ".join(f"{v:.1f}" for v in action)
                #     rad_str = ", ".join(f"{v:.3f}" for v in rad_angles)
                #     print(f"[{frame_count}] action: [{action_str}]")
                #     print(f"         rad: [{rad_str}]")

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
