import os
import time
from collections.abc import Sequence
from math import inf
import cv2
import numpy as np
from config_getter import get_config_value


class Arm:
    # arm_type -> (module_path, class_name)
    _ARM_TYPES = {
        "piper": ("arm.piper_ctrl_by_sdk", "PiperBySDK"),
        "lerobo": ("arm.lerobo_arm_control", "LeroboArm"),
    }

    def __new__(cls, *args, **kwargs):
        if cls is Arm:
            import importlib

            arm_type = get_config_value("arm_type")
            entry = cls._ARM_TYPES.get(arm_type)
            if not entry:
                raise ValueError(
                    f"Unknown arm type in config: {arm_type}, "
                    f"available: {list(cls._ARM_TYPES)}"
                )
            module_path, class_name = entry
            module = importlib.import_module(module_path)
            target_class = getattr(module, class_name)
            return target_class(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        hand_eye_calibration_file=os.path.join(
            os.path.dirname(__file__), "hand-eye-data/2d_homography.npy"
        ),
    ):
        self.desktop_height = get_config_value("default_desktop_height")
        self.catch_raise_height = get_config_value("catch_raise_height", 0.1)
        self.place_raise_height = get_config_value("place_raise_height", 0.1)
        self.default_gripper_close_threshold = get_config_value(
            "default_gripper_close_threshold"
        )
        self.catch_time_interval_s = get_config_value("catch_time_interval_s")
        self.get_arm_angles_retry_times = get_config_value(
            "get_arm_angles_retry_times"
        )
        self.catch_times = 0
        self.uncatch_times = 0
        if os.path.exists(hand_eye_calibration_file):
            self.hand_eye_calibration_matrix = np.load(
                hand_eye_calibration_file)

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | None = None,
    ) -> bool:
        """
        设置机械臂角度和夹爪张开角度
        angles: 机械臂各关节角度列表，单位度，顺序从底座到末端执行器，None表示不改变当前角度
        gripper_angle: 夹爪张开角度，范围0-1，越大越开，None表示不改变当前角度
        """
        raise NotImplementedError(
            "set_arm_angles method must be implemented in subclass"
        )

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[list[float] | None, float | None]:
        raise NotImplementedError(
            "get_arm_angles method must be implemented in subclass"
        )

    def get_arm_pos(self) -> list[float] | None:
        raise NotImplementedError(
            "get_arm_pos method must be implemented in subclass"
        )

    def move_to_home(self, gripper_open_0to1: float | int | None = None):
        raise NotImplementedError(
            "move_to_home method must be implemented in subclass"
        )

    def move_to(
        self,
        pos: list[float],
        gripper_open_0to1: float | int | None = None,
        rot_rad: float | int | None = None,
    ):
        """
        机械臂移动到指定位置，单位米
        pos: [x, y, z]
        gripper_open_0to1: 夹爪张开角度，范围0-1，越大越开，None表示不改变当前角度
        rot_rad: 末端执行器绕z轴旋转角度，单位弧度，None表示不改变当前角度
        """
        raise NotImplementedError(
            "move_to method must be implemented in subclass")

    def disconnect_arm(self):
        raise NotImplementedError(
            "disconnect_arm method must be implemented in subclass"
        )

    def enable_torque(self):
        raise NotImplementedError(
            "enable_torque method must be implemented in subclass"
        )

    def disable_torque(self):
        raise NotImplementedError(
            "disable_torque method must be implemented in subclass"
        )

    def catch(
        self,
        target_x: float,
        target_y: float,
        rad: float,
        height: float = inf,
    ):
        self.catch_times += 1
        if height == inf:
            height = self.desktop_height

        res = self.move_to(
            [target_x, target_y, height + self.catch_raise_height],
            gripper_open_0to1=1,
            rot_rad=rad,
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)

        res = self.move_to(
            [target_x, target_y, height],
            gripper_open_0to1=1,
            rot_rad=rad,
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)

        self.set_arm_angles(gripper_open_0to1=0)
        time.sleep(self.catch_time_interval_s)

        res = self.move_to(
            [target_x, target_y, height + self.catch_raise_height],
            gripper_open_0to1=0,
            rot_rad=rad,
        )
        if res is None:
            print("移动到目标位置失败，取消抓取")
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)

        _, gripper = self.get_arm_angles()
        if gripper is None or gripper < self.default_gripper_close_threshold:
            print("夹取失败")
            self.uncatch_times += 1
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)
        return True

    def place(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        rad: float = 0,
    ):
        res = self.move_to(
            [target_x, target_y, target_z + self.place_raise_height],
            gripper_open_0to1=0,
            rot_rad=rad,
        )
        if res is None:
            print("移动到目标位置失败，取消放置")
            self.move_to_home(gripper_open_0to1=1)
            return False
        time.sleep(self.catch_time_interval_s)

        self.set_arm_angles(gripper_open_0to1=1)
        time.sleep(self.catch_time_interval_s)
        return True

    def catch_and_place(
        self,
        target_x: float,
        target_y: float,
        catch_rotate_rad: float,
        place_pos: list[float | int] = [0.2, 0.0],
        height: float = inf,
        place_rotate_rad: float = 0,
    ):
        if len(place_pos) == 2:
            place_x, place_y = place_pos
            place_z = self.desktop_height
        elif len(place_pos) == 3:
            place_x, place_y, place_z = place_pos
        else:
            print("放置位置格式错误，应该是[x, y]或[x, y, z]")
            return False
        if not self.catch(target_x, target_y, catch_rotate_rad, height=height):
            self.move_to_home(gripper_open_0to1=1)
            return False
        self.move_to_home()
        if not self.place(place_x, place_y, place_z, place_rotate_rad):
            self.move_to_home(gripper_open_0to1=1)
            return False
        self.move_to_home(gripper_open_0to1=1)
        return True

    def pixel2pos(self, u: float, v: float) -> tuple[float, float]:
        if not hasattr(self, "hand_eye_calibration_matrix"):
            raise ValueError("没有手眼标定数据，无法转换图像坐标")
        pixel_coords = np.array([[u], [v], [1]])
        world_coords = self.hand_eye_calibration_matrix @ pixel_coords
        world_coords /= world_coords[2]
        target_x, target_y = world_coords[0, 0], world_coords[1, 0]
        return target_x, target_y

    @staticmethod
    def gripper_angle_by_longer(
        u: float, v: float, w: float, h: float, angle_deg: float
    ) -> float:
        box_points = cv2.boxPoints(((u, v), (w, h), angle_deg))
        if np.linalg.norm(box_points[0] - box_points[1]) > np.linalg.norm(
            box_points[1] - box_points[2]
        ):
            box_points = (
                [box_points[0], box_points[1]]
                if box_points[0][0] < box_points[1][0]
                else [box_points[1], box_points[0]]
            )
        else:
            box_points = (
                [box_points[1], box_points[2]]
                if box_points[1][0] < box_points[2][0]
                else [box_points[2], box_points[1]]
            )
        gripper_angle_rad = np.pi / 2 + np.arctan2(
            box_points[1][1] - box_points[0][1],
            box_points[1][0] - box_points[0][0],
        )
        if gripper_angle_rad > np.pi / 2:
            gripper_angle_rad -= np.pi
        return gripper_angle_rad
