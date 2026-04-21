"""
Leader-Follower Simulation Demo
MuJoCo 仿真机械臂 (leader) 控制 物理主机械臂 (follower) 
读取 leader 的关节位置，转换为弧度后驱动 实际机器

该程序不完善，机械臂有较大的跳变现象，但比较符合之前说的问题
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
import cv2


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


def rad2action(rad_angles, calibration_file="calibration/koch_follower.json", use_degrees=False):
    """将弧度角度转换为归一化 action（与 action2rad 相反）

    转换流程：rad → prepo_direct → prepo → action
    """
    import math

    if not os.path.exists(calibration_file):
        calibration_file = os.path.join(os.path.dirname(__file__), '..', calibration_file)

    with open(calibration_file, 'r') as f:
        calibration = json.load(f)

    # 从 mujoco_sim.py 导入 MODEL_HOMING
    from sim.mujoco_sim import MODEL_HOMING

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Step 1: rad → prepo_direct (反向 prepo2rad_direct)
    prepo_direct = []
    for i in range(len(rad_angles)):
        rad = rad_angles[i]
        if i < len(rad_angles) - 1:
            # 关节: rad → degrees → [0, 4096]
            degrees = rad * (180.0 / math.pi)
            val = (degrees + 180.0) / 360.0 * 4096.0
        else:
            # 夹爪: rad → degrees → [2000, 2900]
            degrees = rad * (180.0 / math.pi)
            val = 2000 + (degrees - 180) / 80 * (2900 - 2000)
        prepo_direct.append(val)

    # Step 2: prepo_direct → prepo (反向 prepo2rad_homing)
    homing_offset = [calibration[joint]["homing_offset"] for joint in joint_names]
    my_prepose = prepo_direct
    actual = [my_prepose[i] - MODEL_HOMING[i] for i in range(len(prepo_direct))]
    prepo = [actual[i] + offset for i, offset in enumerate(homing_offset)]

    # Step 3: prepo → action (反向 action2prepo)
    action = []
    for i, joint_name in enumerate(joint_names):
        calib = calibration[joint_name]
        min_ = calib["range_min"]
        max_ = calib["range_max"]
        drive_mode = calib["drive_mode"]

        if joint_name == "gripper":
            # 反向: unnormalized → bounded_val → val → action
            unnormalized = prepo[i]
            bounded_val = (unnormalized - min_) / (max_ - min_) * 100
            val = bounded_val if not drive_mode else (100 - bounded_val)
            action.append(val)
        elif use_degrees:
            # DEGREES 模式
            mid = (min_ + max_) / 2
            max_res = 4095
            unnormalized = prepo[i]
            degrees = (unnormalized - mid) / max_res * 360
            action.append(degrees)
        else:
            # RANGE_M100_100 模式
            unnormalized = prepo[i]
            bounded_val = (unnormalized - min_) / (max_ - min_) * 200 - 100
            action.append(bounded_val)

    return action


def main():
    """主函数，创建仿真模型并运行"""
    parser = argparse.ArgumentParser(description="Sim-Follower Demo")
    parser.add_argument(
        "--follower_calibration",
        type=str,
        default="calibration/koch_follower.json",
        help="Follower 机械臂校准文件路径",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="urdf/meshes/mjmodel_opt.xml",
        help="MuJoCo 模型 XML 文件路径",
    )
    args = parser.parse_args()

    # 读取配置
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        print("Failed to load config.yaml")
        return

    follower_port = config.get("arm_port", None)
    if follower_port is None:
        raise ValueError("配置文件中没有设置机械臂端口号 arm_port")

    # 初始化 follower 物理机械臂
    follower_arm = DynamixelMotorsBus(
        port=follower_port,
        motors={
            "shoulder_pan": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
        },
    )

    follower_arm.connect()

    # 加载 follower 校准数据
    calibration_data = json.load(open(args.follower_calibration, "r"))
    for motor_name, calib in calibration_data.items():
        calibration_data[motor_name] = MotorCalibration(**calib)
    follower_arm.write_calibration(calibration_dict=calibration_data)

    # 使能扭矩
    follower_arm.enable_torque()
    print("Follower 机械臂已连接并使能")

    model_path = args.model_path
    sim = SimMujocoModel(model_path)

    # 注册视角
    sim.add_view('default', {'distance': 3.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -20.0, 'azimuth': 135.0})
    
    """
        distance: 0.3977693502162603
        lookat: [0.04549377 0.00060337 0.17765481]
        elevation: -89.0
        azimuth: 90.75984990619148
        trackbodyid: -1
        fixedcamid: -1
    """
    sim.add_view('top',     {'distance': 0.39, 'lookat': [0.04, 0.0, 0.17], 'elevation': -89.0, 'azimuth': 90.0})
    sim.switch_view('default')

    # 创建离屏渲染器和两个相机视角
    render_w, render_h = 640, 480
    renderer = mujoco.Renderer(sim.model, height=render_h, width=render_w)

    # Top 视角相机：俯视
    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_cam.distance = 0.39
    top_cam.lookat[:] = [0.04, 0.0, 0.17]
    top_cam.elevation = -89.0
    top_cam.azimuth = 90.0

    # Follow 视角相机：跟随夹爪（使用 XML 中定义的 gripper_cam）
    follow_cam = mujoco.MjvCamera()
    follow_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    follow_cam.fixedcamid = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_cam")

    kb = KeyboardController().start()
    print("\n=== 键盘控制 ===")
    print("上/下键: 切换关节  左/右键: 调整角度  空格: 重置  ESC: 退出\n")

    current_joint = 0
    prev_up = prev_down = prev_space = False

    # 运行仿真循环——通过 up/down 控制关节id，left/right 控制关节移动角度（弧度）
    while True:
        if not sim.viewer.is_running():
            break

        up    = kb.get('up')
        down  = kb.get('down')
        space = kb.get(' ')

        # 上升沿检测：切换关节
        if up and not prev_up:
            current_joint = (current_joint - 1) % max(sim.model.nu, 1)
            print(f"选中关节 {current_joint}: {sim.target_joint_positions[current_joint]:.3f} rad")
        if down and not prev_down:
            current_joint = (current_joint + 1) % max(sim.model.nu, 1)
            print(f"选中关节 {current_joint}: {sim.target_joint_positions[current_joint]:.3f} rad")

        # 持续控制选中关节角度
        moving = False
        if sim.model.nu > 0:
            if kb.get('left'):
                sim.target_joint_positions[current_joint] = np.clip(
                    sim.target_joint_positions[current_joint] + sim.position_speed * sim.dt,
                    -3.14159, 3.14159)
                moving = True
            if kb.get('right'):
                sim.target_joint_positions[current_joint] = np.clip(
                    sim.target_joint_positions[current_joint] - sim.position_speed * sim.dt,
                    -3.14159, 3.14159)
                moving = True

        if moving:
            poses = sim.get_current_poses()
            lj = poses['last_joint']
            gr = poses['gripper']
            print(f"[last_joint ] pos={np.array2string(lj['pos'], precision=4, suppress_small=True)}  quat={np.array2string(lj['quat'], precision=4, suppress_small=True)}")
            print(f"[gripper    ] pos={np.array2string(gr['pos'], precision=4, suppress_small=True)}  quat={np.array2string(gr['quat'], precision=4, suppress_small=True)}")

            # 将仿真关节角度转换为归一化 action 并发送给 follower 机械臂
            action = rad2action(sim.target_joint_positions[:6], calibration_file=args.follower_calibration)
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            for i, motor_name in enumerate(joint_names):
                follower_arm.write("Goal_Position", motor_name, action[i])

            action_str = ", ".join(f"{v:.2f}" for v in action)
            print(f"发送到 follower: [{action_str}]")
            
            
        # 空格重置
        if space and not prev_space:
            mujoco.mj_resetData(sim.model, sim.data)
            sim.target_joint_positions = sim.data.qpos[:sim.model.nu].copy()
            print("位置重置")
            cam = sim.viewer.cam
            print("type:", cam.type)           # 枚举：mujoco.mjtCamera.mjCAMERA_FREE/TRACKING/FIXED
            print("distance:", cam.distance)
            print("lookat:", cam.lookat)       # 长度 3 列表
            print("elevation:", cam.elevation)
            print("azimuth:", cam.azimuth)
            print("trackbodyid:", cam.trackbodyid)
            print("fixedcamid:", cam.fixedcamid)

        prev_up    = up
        prev_down  = down
        prev_space = space

        sim.update_camera()
        _apply_pd_control(sim)
        mujoco.mj_step(sim.model, sim.data)
        sim.viewer.sync()

        # 离屏渲染两个视角并用 OpenCV 显示
        renderer.update_scene(sim.data, top_cam)
        top_img = renderer.render()
        cv2.imshow("Top View", cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))

        renderer.update_scene(sim.data, follow_cam)
        follow_img = renderer.render()
        cv2.imshow("Follow View", cv2.cvtColor(follow_img, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

        time.sleep(sim.dt * 0.5)

    kb.stop()
    follower_arm.disconnect()
    renderer.close()
    cv2.destroyAllWindows()
    sim.close()
    print("已退出")


if __name__ == "__main__":
    main()
