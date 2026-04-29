import sys
import os
import platform

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lerobot", "src"))
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import tqdm
import time
import json
import argparse
import yaml
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from datetime import datetime

# Windows 上 select 不支持 stdin，使用 msvcrt
if platform.system() == 'Windows':
    import msvcrt
else:
    import select

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sim.mujoco_sim import SimMujocoModel, action2rad, _apply_pd_control


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
    

DISABLE_TIME = 300
TOP_CAM_IDX = 4
ARM_CAM_IDX = 2

def main():
    parser = argparse.ArgumentParser(description="Leader-Follower Control")
    parser.add_argument(
        "--calibration_left_path",
        type=str,
        default="calibration/koch_follower.json",
        help="Path to the leader arm calibration file",
    )
    parser.add_argument(
        "--follower_left_path",
        type=str,
        default="calibration/koch_follower.json",
        help="Path to the follower arm calibration file",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=True,
        help="Enable recording of trajectory and video streams",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./recordings",
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--enable_sim",
        action="store_true",
        default=True,
        help="Enable MuJoCo simulation",
    )
    parser.add_argument(
        "--enable_camera",
        action="store_true",
        default=True,
        help="Enable real camera recording",
    )
    args = parser.parse_args()

    # 从 config.yaml 读取端口信息
    config = None
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        print("Failed to load config.yaml")
        return
    leader_left_port = config.get("leader_port", None)
    if leader_left_port is None:
        raise ValueError("配置文件中没有设置主机械臂端口号 leader_port")
    follower_left_port = config.get("arm_port", None)
    if follower_left_port is None:
        raise ValueError("配置文件中没有设置从机械臂端口号 arm_port")

    leader_arm_path = args.calibration_left_path
    follower_arm_path = args.follower_left_path

    leader_arm_left = DynamixelMotorsBus(
        port=leader_left_port,
        motors={
            "shoulder_pan": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
        },
    )

    follower_arm_right = DynamixelMotorsBus(
        port=follower_left_port,
        motors={
            "shoulder_pan": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "xl430-w250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
        },
    )

    # 确保连接成功
    leader_arm_left.connect()
    follower_arm_right.connect()

    follower_arm_right.disable_torque()

    calibration_path_leader = leader_arm_path
    calibration_data_leader = json.load(open(calibration_path_leader, "r"))
    for motor_name, calib in calibration_data_leader.items():
        calibration_data_leader[motor_name] = MotorCalibration(**calib)
    leader_arm_left.write_calibration(calibration_dict=calibration_data_leader)

    calibration_path_follower = follower_arm_path
    calibration_data_follower = json.load(open(calibration_path_follower, "r"))
    for motor_name, calib in calibration_data_follower.items():
        calibration_data_follower[motor_name] = MotorCalibration(**calib)
    follower_arm_right.write_calibration(calibration_dict=calibration_data_follower)

    # 运行操作
    seconds = 3000
    frequency = 100

    follower_arm_right.enable_torque()
    enable_flag = True

    # 初始化录制相关变量
    trajectory_data = []
    video_writers = {}
    sim = None
    renderer = None

    # 创建输出目录
    if args.record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, timestamp)
        os.makedirs(output_path, exist_ok=True)
        print(f"录制数据将保存到: {output_path}")

    # 初始化 MuJoCo 仿真
    if args.enable_sim:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'meshes', 'mjmodel_opt.xml')
        sim = SimMujocoModel(model_path)
        sim.add_view('default', {'distance': 3.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -20.0, 'azimuth': 135.0})
        sim.switch_view('default')

        # 创建离屏渲染器
        render_w, render_h = 640, 480
        renderer = mujoco.Renderer(sim.model, height=render_h, width=render_w)

        # Top 视角相机：俯视
        top_cam = mujoco.MjvCamera()
        top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        top_cam.distance = 0.39
        top_cam.lookat[:] = [0.04, 0.0, 0.17]
        top_cam.elevation = -89.0
        top_cam.azimuth = 90.0

        # Follow 视角相机：跟随夹爪
        follow_cam = mujoco.MjvCamera()
        follow_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        follow_cam.fixedcamid = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_cam")

        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writers['sim_top'] = cv2.VideoWriter(
                os.path.join(output_path, 'sim_top.mp4'), fourcc, frequency, (render_w, render_h))
            video_writers['sim_follow'] = cv2.VideoWriter(
                os.path.join(output_path, 'sim_follow.mp4'), fourcc, frequency, (render_w, render_h))

    # 初始化真实相机
    camera_top = None
    camera_arm = None
    if args.enable_camera:
        """
            直接使用 opencv 的 VideoCapture 来读取相机数据，减少依赖和复杂度
            'index': 4, 'name': 'USB 视频设备'
            'index': 0, 'name': 'Orbbec Gemini 215 RGB Camera'
        """
        try:
            # 尝试打开顶部相机 (Orbbec Gemini 215 RGB Camera, index 0)
            camera_top = cv2.VideoCapture(TOP_CAM_IDX)
            if camera_top.isOpened():
                # 设置分辨率
                camera_top.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera_top.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("顶部相机初始化成功 (index 0)")

                if args.record:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writers['camera_top'] = cv2.VideoWriter(
                        os.path.join(output_path, 'camera_top.mp4'), fourcc, frequency, (640, 480))
            else:
                print("顶部相机初始化失败 (index 4)")
                camera_top = None
        except Exception as e:
            print(f"顶部相机初始化失败: {e}")
            camera_top = None

        try:
            # 尝试打开手臂相机 (USB 视频设备, index 4)
            camera_arm = cv2.VideoCapture(ARM_CAM_IDX)
            if camera_arm.isOpened():
                # 设置分辨率
                camera_arm.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera_arm.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("手臂相机初始化成功 (index 2)")

                if args.record:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writers['camera_arm'] = cv2.VideoWriter(
                        os.path.join(output_path, 'camera_arm.mp4'), fourcc, frequency, (640, 480))
            else:
                print("手臂相机初始化失败 (index 4)")
                camera_arm = None
        except Exception as e:
            print(f"手臂相机初始化失败: {e}")
            camera_arm = None


    print("开始运行 Leader-Follower 控制...")
    print("按 'q' + Enter 退出")

    start_time = time.time()
    

    for frame_idx in range(seconds * frequency):
        # 使用正确的键名 "left" 和 "right"
        try:
            leader_pos_left = {}
            follow_pos_right = {}
            for motor in leader_arm_left.motors.keys():
                leader_pos_left[motor] = leader_arm_left.read("Present_Position", motor=motor)

            if enable_flag:
                for motor in follower_arm_right.motors.keys():
                    follow_pos_right[motor] = follower_arm_right.read("Present_Position", motor=motor)
                target_pos = step_towards(follow_pos_right, leader_pos_left, step_size=10)
                # 将 leader 的位置发送到 follower
                for motor, leader_pos in target_pos.items():
                    follower_arm_right.write("Goal_Position", motor, leader_pos)

            if enable_flag and DISABLE_TIME > 0 and DISABLE_TIME < frame_idx :
                print("disable torque")
                follower_arm_right.disable_torque()
                enable_flag = False

            # 记录轨迹数据
            if args.record:
                current_time = time.time() - start_time
                # 将位置数据转换为列表格式
                action_list = [
                    leader_pos_left.get("shoulder_pan", 0),
                    leader_pos_left.get("shoulder_lift", 0),
                    leader_pos_left.get("elbow_flex", 0),
                    leader_pos_left.get("wrist_flex", 0),
                    leader_pos_left.get("wrist_roll", 0),
                    leader_pos_left.get("gripper", 0),
                ]
                trajectory_data.append({
                    'timestamp': current_time,
                    'action': action_list,
                    'leader_pos': leader_pos_left,
                    'follower_pos': follow_pos_right,
                })

            # 更新 MuJoCo 仿真
            if args.enable_sim and sim is not None:
                # 将 action 转换为弧度并更新仿真
                action_list = [
                    leader_pos_left.get("shoulder_pan", 0),
                    leader_pos_left.get("shoulder_lift", 0),
                    leader_pos_left.get("elbow_flex", 0),
                    leader_pos_left.get("wrist_flex", 0),
                    leader_pos_left.get("wrist_roll", 0),
                    leader_pos_left.get("gripper", 0),
                ]
                rad_angles = action2rad(action_list, calibration_file=args.calibration_left_path)
                sim.set_joint_angles(rad_angles)

                sim.update_camera()
                _apply_pd_control(sim)
                mujoco.mj_step(sim.model, sim.data)
                sim.viewer.sync()

                # 渲染并录制仿真视频
                if args.record and renderer is not None:
                    # Top 视角
                    renderer.update_scene(sim.data, top_cam)
                    top_img = renderer.render()
                    top_img_bgr = cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR)
                    video_writers['sim_top'].write(top_img_bgr)
                    cv2.imshow("Sim Top View", top_img_bgr)

                    # Follow 视角
                    renderer.update_scene(sim.data, follow_cam)
                    follow_img = renderer.render()
                    follow_img_bgr = cv2.cvtColor(follow_img, cv2.COLOR_RGB2BGR)
                    video_writers['sim_follow'].write(follow_img_bgr)
                    cv2.imshow("Sim Follow View", follow_img_bgr)

            # 采集并录制真实相机画面
            if args.enable_camera:
                # 顶部相机
                if camera_top is not None and camera_top.isOpened():
                    ret, frame_top = camera_top.read()
                    if ret and frame_top is not None:
                        cv2.imshow("Camera Top", frame_top)
                        if args.record:
                            video_writers['camera_top'].write(frame_top)

                # 手臂相机
                if camera_arm is not None and camera_arm.isOpened():
                    ret, frame_arm = camera_arm.read()
                    if ret and frame_arm is not None:
                        cv2.imshow("Camera Arm", frame_arm)
                        if args.record:
                            video_writers['camera_arm'].write(frame_arm)

            # 确保不会在过高频率下运行，可以加入一些延迟
            time.sleep(1 / frequency)

            # OpenCV 窗口事件处理
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                print("ESC pressed, exiting...")
                break

           
        except Exception as e:
            print(f"Error in control loop: {e}")
            pass

    # 保存录制数据
    if args.record:
        print("\n正在保存录制数据...")

        # 保存轨迹数据为 JSONL 格式
        trajectory_file = os.path.join(output_path, 'trajectory.jsonl')
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            for entry in trajectory_data:
                json.dump(entry, f)
                f.write('\n')
        print(f"轨迹数据已保存: {trajectory_file} ({len(trajectory_data)} 帧)")

        # 释放所有视频写入器
        for name, writer in video_writers.items():
            writer.release()
            print(f"视频已保存: {name}.mp4")

        print(f"所有数据已保存到: {output_path}")

    # 清理资源
    cv2.destroyAllWindows()

    # 释放真实相机
    if camera_top is not None:
        camera_top.release()
        print("顶部相机已关闭")

    if camera_arm is not None:
        camera_arm.release()
        print("手臂相机已关闭")

    if sim is not None:
        if renderer is not None:
            renderer.close()
        sim.close()
        print("仿真已关闭")

    follower_arm_right.disable_torque()
    follower_arm_right.disconnect()
    leader_arm_left.disable_torque()
    leader_arm_left.disconnect()
    print("机械臂已断开连接")


if __name__ == "__main__":
    
    main()
