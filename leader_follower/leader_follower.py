import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lerobot", "src"))
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import tqdm
import time
import json
import select
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Leader-Follower Control")
    parser.add_argument(
        "--calibration_left_path",
        type=str,
        default="calibration/koch_follower.json",
        help="Path to the follower left arm calibration file",
    )
    parser.add_argument(
        "--follower_left_path",
        type=str,
        default="calibration/koch_follower.json",
        help="Path to the follower left arm calibration file",
    )
    args = parser.parse_args()
    
    # 从 config.yaml 读取端口信息
    config = None
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        print("Failed to load config.yaml")
        return
    leader_left_port = config.get("leader_port", "COM1")
    follower_left_port = config.get("arm_port", "COM2")

    leader_arm_path = args.calibration_left_path
    follower_arm_path = args.follower_left_path
    

    norm_mode_body = MotorNormMode.DEGREES
    leader_arm_left = DynamixelMotorsBus(
        port=leader_left_port,
        motors={
            "shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
        },
    )

    follower_arm_left = DynamixelMotorsBus(
        port=follower_left_port,
        motors={
            "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
            "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
            "elbow_flex": Motor(3, "xl330-m288", norm_mode_body),
            "wrist_flex": Motor(4, "xl330-m288", norm_mode_body),
            "wrist_roll": Motor(5, "xl330-m288", norm_mode_body),
            "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
        },
    )

    # 确保连接成功
    leader_arm_left.connect()
    follower_arm_left.connect()

    follower_arm_left.disable_torque()

    calibration_path_leader = leader_arm_path
    calibration_data_leader = json.load(open(calibration_path_leader, "r"))
    for motor_name, calib in calibration_data_leader.items():
        calibration_data_leader[motor_name] = MotorCalibration(**calib)
    leader_arm_left.write_calibration(calibration_dict=calibration_data_leader)

    calibration_path_follower = follower_arm_path
    calibration_data_follower = json.load(open(calibration_path_follower, "r"))
    for motor_name, calib in calibration_data_follower.items():
        calibration_data_follower[motor_name] = MotorCalibration(**calib)
    follower_arm_left.write_calibration(calibration_dict=calibration_data_follower)

    # 运行操作
    seconds = 3000
    frequency = 100

    follower_arm_left.enable_torque()

    for _ in tqdm.tqdm(range(seconds * frequency)):
        # 使用正确的键名 "left" 和 "right"
        try:
            leader_pos_left = {}
            for motor in leader_arm_left.motors.keys():
                leader_pos_left[motor] = leader_arm_left.read("Present_Position", motor=motor)

            # print format .2f
            # print({k: f"{v:.2f}" for k, v in leader_pos_left.items()})
            # 将 leader 的位置发送到 follower
            for motor, leader_pos in leader_pos_left.items():
                follower_arm_left.write("Goal_Position", motor, leader_pos)

            # 确保不会在过高频率下运行，可以加入一些延迟
            time.sleep(1 / frequency)

            # 检测 控制台 是否输入 q 来退出
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line.strip() == 'q':
                    print("Exiting...")
                    break

        except Exception as e:
            pass

    follower_arm_left.disable_torque()
    follower_arm_left.disconnect()
    leader_arm_left.disable_torque()
    leader_arm_left.disconnect()


if __name__ == "__main__":
    
    main()
