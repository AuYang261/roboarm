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

CALIBRATION_FILE = "calibration/koch_follower.json"
# CALIBRATION_FILE = "calibration/koch_follower_2.json"
COM_PORT = "COM5"  # 替换为你的机械臂连接的串口号
norm_mode_body = MotorNormMode.DEGREES
# norm_mode_body = MotorNormMode.RANGE_M100_100


def main():
    parser = argparse.ArgumentParser(description="Leader-Follower Control")

    parser.add_argument(
        "--follower_left_path",
        type=str,
        default=CALIBRATION_FILE,
        help="Path to the follower left arm calibration file",
    )
    args = parser.parse_args()

    # 从 config.yaml 读取端口信息
    # config = None
    # with open("config.yaml", "r", encoding="utf-8") as f:
    #     config = yaml.safe_load(f)
    # if config is None:
    #     print("Failed to load config.yaml")
    #     return

    # follower_left_port = config.get("arm_port", None)
    follower_left_port = COM_PORT
    if follower_left_port is None:
        raise ValueError("配置文件中没有设置从机械臂端口号 arm_port")


    follower_arm_path = args.follower_left_path

    
    follower_arm_right = DynamixelMotorsBus(
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

    follower_arm_right.connect()

    follower_arm_right.disable_torque()


    calibration_path_follower = follower_arm_path
    calibration_data_follower = json.load(open(calibration_path_follower, "r"))
    for motor_name, calib in calibration_data_follower.items():
        calibration_data_follower[motor_name] = MotorCalibration(**calib)
    follower_arm_right.write_calibration(calibration_dict=calibration_data_follower)

    # 运行操作
    seconds = 3000
    frequency = 3

    # follower_arm_right.enable_torque()
    # 尝试添加摄像头

    # for _ in tqdm.tqdm(range(frequency)):
    for _ in range(int(frequency)):
        # 使用正确的键名 "left" 和 "right"
        try:

            follow_pos_right = {}
            follow_pos_right_normalize = {}
            print("\n")
            for motor in follower_arm_right.motors.keys():
                follow_pos_right[motor] = follower_arm_right.read("Present_Position", motor=motor, normalize=False)
                follow_pos_right_normalize[motor] = follower_arm_right.read("Present_Position", motor=motor, normalize=True)
                print(f"{motor}: {follow_pos_right[motor]}, normalized: {follow_pos_right_normalize[motor]}")
                # 写入文件
                # with open(f"logs/follow_pos_right_{motor}.txt", "a") as f:
                #     f.write(f"{follow_pos_right[motor]=}\n")
                #     f.write(f"{follow_pos_right_normalize[motor]=}\n")
                
            # 确保不会在过高频率下运行，可以加入一些延迟
            time.sleep(frequency)



        except Exception as e:
            pass

    follower_arm_right.disable_torque()
    follower_arm_right.disconnect()


if __name__ == "__main__":
    
    main()
