# https://placo.readthedocs.io/en/latest/ - 官方文档
# https://github.com/Rhoban/placo-examples.git - 官方例程
# https://placo.readthedocs.io/en/latest/kinematics/getting_started.html - 运动学求解器
# lerobot/src/lerobot/model/kinematics.py - lerobot运动

import placo
import math
import sys
import random
import numpy as np
import kinpy as kp
import tqdm
import time
import json
import threading
import os
from placo_utils.tf import tf

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lerobot", "src")
)
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.model.kinematics import RobotKinematics


follower_left_port = "ttyACM0"
# leader = "ttyACM1"


def init_arm() -> DynamixelMotorsBus:
    norm_mode_body = MotorNormMode.DEGREES

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
    follower_arm_left.connect()

    follower_arm_left.disable_torque()

    calibration_path_follower = r"config/koch_follower_arm_1.json"
    calibration_data_follower = json.load(open(calibration_path_follower, "r"))
    for motor_name, calib in calibration_data_follower.items():
        calibration_data_follower[motor_name] = MotorCalibration(**calib)
    follower_arm_left.write_calibration(calibration_dict=calibration_data_follower)
    
    follower_arm_left.enable_torque()

    return follower_arm_left

def disable_arm(arm: DynamixelMotorsBus):
    try:
        arm.disable_torque()
        arm.disconnect()
    except Exception as e:
        print(f"Error occurred while disabling arm: {e}")

def get_cur_arm_angles(arm: DynamixelMotorsBus) -> list:
    pos = {}
    for motor in arm.motors.keys():
        pos[motor] = arm.read("Present_Position", motor=motor)
        
    # 获取 机械臂 位姿
    th_init = [
        math.radians(pos["shoulder_pan"]),
        math.radians(pos["shoulder_lift"]),
        math.radians(pos["elbow_flex"]),
        math.radians(pos["wrist_flex"]),
        math.radians(pos["wrist_roll"]),
        math.radians(pos["gripper"]),
    ]
    
    return th_init

def arm_move(arm: DynamixelMotorsBus, angles: list):

    # {'shoulder_pan': '4.97', 'shoulder_lift': '17.89', 'elbow_flex': '-39.43', 'wrist_flex': '-59.25', 'wrist_roll': '51.74', 'gripper': '3.93'}
    pos = {}
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for i, motor in enumerate(motor_names):
        pos[motor] = angles[i]
        
    # print({k: f"{v:.2f}" for k, v in pos.items()})
    
    try:
        for motor, leader_pos in pos.items():
            arm.write("Goal_Position", motor, leader_pos)
    except Exception as e:
        print(f"Error occurred while moving arm: {e}")  

def get_q_qd_qdd(robot, joint_names_list): # 官方-这个是关节配置吗
    
    # 获取关节位置
    positions = []
    for name in joint_names_list:
        positions.append(robot.get_joint(name))
    for i in range(len(positions)):
        print(f"joint {i} : {joint_names_list[i]} position = {positions[i]}")
    
    
    # 获取关节速度
    speeds = []
    for name in joint_names_list:
        speeds.append(robot.get_joint_velocity(name))
    for i in range(len(speeds)):
        print(f"joint {i} : {joint_names_list[i]} speed = {speeds[i]}")
        
    # 获取关节加速度
    accelerations = []
    for name in joint_names_list:
        accelerations.append(robot.get_joint_acceleration(name))
    for i in range(len(accelerations)):
        print(f"joint {i} : {joint_names_list[i]} acceleration = {accelerations[i]}")
    
    

def main():
    # 加载urdf模型
    robot = placo.RobotWrapper("urdf/low_cost_robot.urdf") # 默认开启碰撞检测
    
    # 创建运动学结算期
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)  # Fix the base
    
    joint_names_list = list(robot.joint_names())
    print("joint_names", joint_names_list)
    
    # 打印所有 frame 名称
    print("Frames in model:")
    for i, frame in enumerate(solver.robot.model.frames):
        print(i, frame.name)
    '''
    Frames in model:
    0 universe
    1 root_joint
    2 base_link
    3 joint1
    4 link1_1
    5 joint2
    6 link2_1
    7 joint3
    8 link3_1
    9 joint4
    10 link4_1
    11 joint5
    12 gripper_static_1
    13 joint_gripper
    14 gripper_moving_1
    '''


if __name__ == "__main__":
    main()
