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


sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lerobot", "src")
)
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.model.kinematics import RobotKinematics


follower_left_port = "COM6"

def rad(deg):
    return deg * math.pi / 180.0


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
        
    print({k: f"{v:.2f}" for k, v in pos.items()})
    
    try:
        for motor, leader_pos in pos.items():
            arm.write("Goal_Position", motor, leader_pos)
    except Exception as e:
        print(f"Error occurred while moving arm: {e}")  

# 旋转矩阵---->四元数
def matrix_to_quaternion(matrix):
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def init_kp(urdf_path: str) -> kp.chain.SerialChain:
    # 1. 读取 URDF
    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_xml = f.read()

    # 2. 构建链条
    end_link_name = "gripper_moving_1"
    serial_chain = kp.build_serial_chain_from_urdf(urdf_xml, end_link_name)
    
    # joint_names = serial_chain.get_joint_parameter_names()
    # print("\n=== SerialChain joint names ===")
    # print(joint_names)
    
    # print("type(serial_chain) =", type(serial_chain))

    return serial_chain

def forward_kinematics_demo(serial_chain : kp.chain.SerialChain, th_init):

    # 初始位置正向运动学
    init_tf = serial_chain.forward_kinematics(th_init, end_only=True)
    return init_tf

def inverse_kinematics_demo(serial_chain : kp.chain.SerialChain, th_cur, target_tf_pos, target_tf_rot=None) -> list:
    # 给定一个目标三维坐标，使用默认的旋转矩阵
    '''
    [
        [1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]
    ]
    '''
    if target_tf_rot is None:
        target_tf_rot = np.eye(3)

    target_tf = kp.Transform(pos=target_tf_pos, rot=matrix_to_quaternion(target_tf_rot))
    # 逆运动学解算
    ik_solution = serial_chain.inverse_kinematics(target_tf, th_cur)
    return ik_solution

#  逆运动学：从当前位姿 到 目标 xyz
def move_to_target(serial_chain : kp.chain.SerialChain, arm: DynamixelMotorsBus, xyz: list, steps=5):
    
    th_cur = get_cur_arm_angles(arm)
    
    for alpha in np.linspace(0, 1, steps):
        # 插值末端位姿 (只插值位置，旋转保持当前值简化处理)
        init_tf = serial_chain.forward_kinematics(th_cur, end_only=True)
        interp_pos = init_tf.pos * (1 - alpha) + np.array(xyz) * alpha
        interp_tf = kp.Transform(pos=interp_pos, rot=init_tf.rot)

        # 逆运动学解算
        ik_solution = serial_chain.inverse_kinematics(interp_tf, th_cur)

        # 正向验证
        check_tf = serial_chain.forward_kinematics(ik_solution, end_only=True)

        print(f"\nStep {alpha:.2f}:")
        print("  Joint angles (radians) =", ik_solution)
        print("  End-effector pos =", check_tf.pos)
        
        # 控制机械臂
        arm_move(arm, [math.degrees(angle) for angle in ik_solution])
        
        time.sleep(1)  # 等待机械臂运动完成


def main():

    # 1. 初始化 kinpy & 初始化机械臂
    try:
        follower_arm_left = init_arm()
        serial_chain = init_kp("./config/low_cost_robot.urdf")
    except Exception as e:
        print(f"Error occurred while initializing: {e}")  
        return
    
    # 2. 读取当前关节角度
    th_cur = get_cur_arm_angles(follower_arm_left)
    print("当前机械臂关节角度 (弧度):", th_cur)

    # 3. 初始位置正向运动学
    init_tf = serial_chain.forward_kinematics(th_cur, end_only=True)
    print("\n初始末端位姿:")
    print("pos =", init_tf.pos)
    print("rot =", init_tf.rot)
    
    # 4. 随机一个 pos 作为目标位置  在初始位置附近移动
    target_tf_pos = [random.uniform(-0.1, 0.1) for _ in range(3)]
    target_tf_pos += init_tf.pos  
    
    # 5. 逆运动学：从当前位姿 到 目标 xyz
    move_to_target(serial_chain, follower_arm_left, target_tf_pos, steps=5)
    
    # 6. 检测当前坐标
    th_cur = get_cur_arm_angles(follower_arm_left)
    print("当前机械臂关节角度 (弧度):", th_cur)
    init_tf = serial_chain.forward_kinematics(th_cur, end_only=True)
    print("当前末端位姿:")
    print("pos =", init_tf.pos)
    print("rot =", init_tf.rot)

if __name__ == "__main__":
    main()
