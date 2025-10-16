# Description: 机械臂控制封装
from collections.abc import Sequence
import sys
import os
import time

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lerobot/src/")
)
from lerobot.robots.koch_follower import config_koch_follower, koch_follower
from pathlib import Path
from typing import Union, List
import kinpy
import numpy as np


class Arm:

    def __init__(
        self,
        port,
        calibration_dir=os.path.join(os.path.dirname(__file__), "..", "calibration"),
        id="koch_follower",
        hand_eye_calibration_file=os.path.join(
            os.path.dirname(__file__), "hand-eye-data/2d_homography.npy"
        ),
        steps=5,
    ):
        """
        初始化机械臂
        port: 机械臂串口号
        calibration_dir: 标定文件夹路径，包含机械臂offset文件
        id: 机械臂型号，默认"koch_follower"
        hand_eye_calibration_file: 手眼标定文件路径，默认"hand-eye-data/2d_homography.npy"
        steps: 机械臂插值移动步数，步数越多越平滑但越慢
        """
        config = config_koch_follower.KochFollowerConfig(
            port=port,
            disable_torque_on_disconnect=True,
            use_degrees=True,
            id=id,
            calibration_dir=Path(calibration_dir).resolve(),
        )
        self.steps = steps
        # 这个offset是用来修正机械臂零位的，目前不知道为什么舵机全零位置不是机械臂的零位
        # 所以每次重新标定或在新机械臂上需要重新测量这个offset
        # 方法是标定完后，把机械臂放到零位位置(见docs/image1.png)，然后读取各关节角度，作为offset保存下来
        # 一行一个，单位度，夹爪角度不需要
        self.offset = [
            float(x.strip())
            for x in open(
                os.path.join(calibration_dir, "arm_offset.txt"), "r", encoding="utf-8"
            ).readlines()
            if x.strip() != ""
        ]
        if os.path.exists(hand_eye_calibration_file):
            self.hand_eye_calibration_matrix = np.load(hand_eye_calibration_file)
            with open(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "urdf",
                    "low_cost_robot.urdf",
                ),
                "r",
                encoding="utf-8",
            ) as f:
                urdf_content = f.read()
            self.chain = kinpy.build_serial_chain_from_urdf(
                urdf_content, "gripper_static_1"
            )

        if len(self.offset) != 5:
            raise ValueError("机械臂offset文件格式错误，应该有5个值")
        self.arm = koch_follower.KochFollower(config)
        self.arm.connect()

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_angle_deg: float | int | None = None,
    ):
        """
        设置机械臂角度和夹爪张开角度
        angles: 机械臂各关节角度列表，单位度，顺序从底座到末端执行器，None表示不改变当前角度
        gripper_angle: 夹爪张开角度，范围0-100，越大越开，单位度，None表示不改变当前角度
        """
        motor_names = list(self.arm.bus.motors.keys())
        action: dict[str, float] = {}
        if gripper_angle_deg is not None:
            action[motor_names[-1] + ".pos"] = np.clip(gripper_angle_deg, 0, 100)
        if angles_deg is not None:
            for motor_name, angle_deg in zip(motor_names[:-1], angles_deg, strict=True):
                if angle_deg is not None:
                    action[motor_name + ".pos"] = np.clip(angle_deg, -180, 180)
        if len(action) > 0:
            current_angles_deg, current_gripper_deg = self.get_read_arm_angles()
            if current_angles_deg is None or current_gripper_deg is None:
                self.arm.set_action(action)
                return
            current_angles_deg.append(current_gripper_deg)
            # 对角度插值
            for alpha in np.linspace(0, 1, self.steps):
                interp_action = {}
                for key, value in action.items():
                    motor_index = motor_names.index(key.removesuffix(".pos"))
                    current_angle = current_angles_deg[motor_index]
                    interp_angle = current_angle * (1 - alpha) + value * alpha
                    interp_action[key] = interp_angle + (
                        self.offset[motor_index]
                        if motor_index < len(self.offset)
                        else 0
                    )
                self.arm.send_action(interp_action)
                time.sleep(0.05)

    def get_read_arm_angles(
        self,
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        """
        获取机械臂各关节角度和夹爪状态，单位度
        """
        try:
            angles_deg = list(self.arm.get_observation().values())
        except Exception as e:
            return None, None
        return [
            angle - offset
            for angle, offset in zip(angles_deg[:-1], self.offset, strict=True)
        ], angles_deg[-1]

    def disconnect_arm(self):
        self.arm.disconnect()

    def enable_torque(self):
        """
        连接后默认是enabled力矩的
        """
        self.arm.bus.enable_torque()

    def disable_torque(self):
        self.arm.bus.disable_torque()

    def move_to_home(self, gripper_angle_deg: float | int | None = None):
        """
        机械臂回到初始位置
        """
        self.set_arm_angles([0, 0, 0, 0, 0], gripper_angle_deg=gripper_angle_deg)

    def move_to(
        self,
        pos: List[float],
        gripper_angle_deg: float | int | None = None,
        rot_rad: float | int | None = None,
        warning: bool = True,
    ):
        """
        机械臂移动到指定位置，单位米
        pos: [x, y, z]
        gripper_angle_deg: 夹爪张开角度，范围0-100，越大越开，单位度，None表示不改变当前角度
        rot_rad: 末端执行器绕z轴旋转角度，单位弧度，None表示不改变当前角度
        warning: 是否开启z轴位置警告，默认开启
        """
        if not hasattr(self, "chain"):
            raise ValueError("没有机械臂模型，无法使用位置控制")
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")
        if warning and (pos[2] < 0.05 or pos[2] > 0.1):
            print("Warning: z轴位置建议在0.07米以恰好在桌面上")
        # 控制夹爪角度的舵机逆时针为正，而欧拉角定义为绕z轴顺时针为正，所以这里取负号
        goal_tf = kinpy.Transform(
            pos=np.array(pos), rot=[0, 0, -rot_rad if rot_rad else 0]
        )
        angles_deg = np.rad2deg(self.chain.inverse_kinematics(goal_tf)).tolist()
        self.set_arm_angles(angles_deg, gripper_angle_deg=gripper_angle_deg)
        return self.get_read_arm_angles()

    def pixel2pos(self, u: float, v: float):
        """
        将图像坐标转换为机械臂坐标系位置，单位米
        u, v: 图像坐标，单位像素
        height: 目标物体高度，单位米，默认0.07米
        返回值: [x, y, z]
        """
        if not hasattr(self, "hand_eye_calibration_matrix"):
            raise ValueError("没有手眼标定数据，无法转换图像坐标")
        pixel_coords = np.array([[u], [v], [1]])
        world_coords = self.hand_eye_calibration_matrix @ pixel_coords
        world_coords /= world_coords[2]
        target_x, target_y = world_coords[0, 0], world_coords[1, 0]
        return [target_x, target_y]

    def catch(
        self,
        target_x: float,
        target_y: float,
        rad: float,
        place_pos: list[float | int] = [0.2, 0.2],
        height: float = 0.07,
        time_interval_s: float = 0.5,
        gripper_threshold_deg: float | int = 5,
    ):
        """
        机械臂抓取物体并放到指定位置
        target_x, target_y: 目标物体位置，单位米
        rad: 物体旋转角度，单位弧度
        place_pos: 放置位置，单位米，默认[0.2, 0.2]
        height: 目标物体高度，单位米，默认0.07米
        time_interval_s: 每个动作之间的时间间隔，单位秒，默认0.5秒
        gripper_threshold_deg: 夹爪闭合角度阈值，单位度，默认5度，小于该值认为夹取失败
        """

        # 移动机械臂到目标位置上方
        res = self.move_to(
            [target_x, target_y, height + 0.05],
            gripper_angle_deg=80,
            rot_rad=rad,
            warning=False,
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_angle_deg=80)
            return
        time.sleep(time_interval_s)

        # 下降到目标位置
        res = self.move_to(
            [target_x, target_y, height],
            gripper_angle_deg=80,
            rot_rad=rad,
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_angle_deg=80)
            return
        time.sleep(time_interval_s)

        # 夹紧物体
        self.set_arm_angles(gripper_angle_deg=0)
        time.sleep(time_interval_s)

        # 抬起物体
        res = self.move_to(
            [target_x, target_y, height + 0.05],
            gripper_angle_deg=0,
            rot_rad=rad,
            warning=False,
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_angle_deg=80)
            return
        time.sleep(time_interval_s)

        # 确认夹紧成功
        angles, gripper = self.get_read_arm_angles()
        if gripper is None or gripper < gripper_threshold_deg:
            print("夹取失败")
            self.move_to_home(gripper_angle_deg=80)
            return

        # 放到指定位置
        res = self.move_to(
            place_pos + [height + 0.1], gripper_angle_deg=0, warning=False
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_angle_deg=80)
            return
        time.sleep(time_interval_s)
        res = self.move_to(place_pos + [height], gripper_angle_deg=0)
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_angle_deg=80)
            return
        time.sleep(time_interval_s)
        self.set_arm_angles(gripper_angle_deg=80)
        time.sleep(time_interval_s)
        res = self.move_to(place_pos + [height + 0.1], warning=False)
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_angle_deg=80)
            return
        time.sleep(time_interval_s)
        self.move_to_home(gripper_angle_deg=80)
        time.sleep(time_interval_s)


if __name__ == "__main__":
    arm = Arm("COM3")
    # arm.disable_torque()
    # while True:
    #     print(arm.get_read_arm_angles())
    time.sleep(1)
    arm.move_to_home(gripper_angle_deg=80)
    time.sleep(1)
    angles, gripper = arm.get_read_arm_angles()
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    arm.set_arm_angles(None, gripper_angle_deg=0)
    time.sleep(1)
    angles, gripper = arm.get_read_arm_angles()
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    arm.move_to_home()
    time.sleep(1)
    arm.move_to([0.1, 0.1, 0.1])
    time.sleep(1)
    arm.move_to_home()
    time.sleep(1)
    arm.disconnect_arm()
