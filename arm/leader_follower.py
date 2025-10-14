import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import tqdm
import time
import json
import select

leader_left_port = "COM5"
follower_left_port = "COM6"


def main():

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

    calibration_path_leader = r"calibration/koch_leader_arm_1.json"
    calibration_data_leader = json.load(open(calibration_path_leader, "r"))
    for motor_name, calib in calibration_data_leader.items():
        calibration_data_leader[motor_name] = MotorCalibration(**calib)
    leader_arm_left.write_calibration(calibration_dict=calibration_data_leader)

    calibration_path_follower = r"calibration/koch_follower_arm_1.json"
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
    
    