import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode


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
    leader_arm_left.disable_torque()
    follower_arm_left.disable_torque()



if __name__ == "__main__":
    
    main()
    
    