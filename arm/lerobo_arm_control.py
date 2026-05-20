# Description: 机械臂控制封装
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "lerobot/src/")
)


import time
from arm.arm_base import Arm, StepCallback
from sim.sim_client import SimArmClient
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import numpy as np
import kinpy
from typing import Union, List
from pathlib import Path
from utils.config_getter import get_config_value
from collections.abc import Sequence
from lerobot.robots.koch_follower import config_koch_follower, koch_follower


class LeroboArm(Arm):
    """LeRobot koch_follower 机械臂控制实现。"""

    MAX_GRIPPER_ANGLE_DEG = 100

    def __init__(
        self,
        calibration_dir=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "calibration"
        ),
        robot_id="koch_follower",
        hand_eye_calibration_file=os.path.join(
            os.path.dirname(__file__), "hand-eye-data/2d_homography.npy"
        ),
    ):
        """初始化 LeRobot 机械臂控制器。

        Args:
            calibration_dir: 标定文件目录，包含机械臂 offset 文件。
            robot_id: 机械臂在 LeRobot 中的设备标识。
            hand_eye_calibration_file: 手眼标定矩阵文件路径。
            steps: 插值步数，越大越平滑，但耗时越长。
        """
        super().__init__(hand_eye_calibration_file=hand_eye_calibration_file)
        self.arm_backend = get_config_value(
            "arm_backend", "real", raise_if_missing=False
        )
        self.steps = 20
        self.offset = (
            get_config_value("arm_offset") if self.arm_backend != "sim" else [0] * 5
        )

        if len(self.offset) != 5:
            raise ValueError(
                "配置文件中没有正确设置机械臂offset arm_offset, 应该是5个关节的角度列表"
                "运行arm/calibrate_offset.py以获取arm_offset"
            )
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "urdf",
                "lerobo",
                "low_cost_robot.urdf",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            urdf_content = f.read()
        self.chain = kinpy.build_serial_chain_from_urdf(
            urdf_content, "gripper_static_1"
        )

        if self.arm_backend == "sim":
            self.sim_arm_client = SimArmClient(
                host=get_config_value("arm_sim_host"),
                port=get_config_value("arm_sim_port"),
            )
            while True:
                try:
                    self.sim_arm_client.ping()
                    break
                except ConnectionError as e:
                    print(e, "retry in 1 s")
                    time.sleep(1)
            print("成功连接仿真服务")
        else:
            port = get_config_value("arm_port")
            self.arm = koch_follower.KochFollower(
                config_koch_follower.KochFollowerConfig(
                    port=port,
                    disable_torque_on_disconnect=True,
                    use_degrees=True,
                    id=robot_id,
                    calibration_dir=Path(calibration_dir).resolve(),
                )
            )
            try:
                self.arm.connect()
            except ConnectionError as e:
                raise ConnectionError(
                    f"机械臂连接失败: {e}\n请检查端口号{port}是否正确"
                    + (
                        "，以及是否有777权限(sudo chmod 777 {port})"
                        if os.name != "nt"
                        else ""
                    )
                ) from e

    def _get_joint_names(self) -> list[str]:
        if self.arm_backend == "sim":
            return self.sim_arm_client.get_joint_names()
        return list(self.arm.bus.motors.keys())

    def get_raw_joint_angles(
        self, retry_times=None
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        try:
            if self.arm_backend == "sim":
                return self.sim_arm_client.get_raw_joint_angles()
            angles_deg = list(self.arm.get_observation().values())
            return angles_deg[:-1], np.clip(
                angles_deg[-1] / self.MAX_GRIPPER_ANGLE_DEG, 0, 1
            )
        except Exception:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_raw_joint_angles(retry_times - 1)
            return None, None

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        joint_names = self._get_joint_names()
        target_joint_angles = None if angles_deg is None else list(angles_deg)
        target_gripper = gripper_open_0to1
        if target_gripper is not None and not 0 <= target_gripper <= 1:
            raise ValueError("gripper_open_0to1 must in [0, 1]")

        if target_joint_angles is None and target_gripper is None:
            return True

        current_angles_deg, current_gripper_0to1 = self.get_arm_angles()
        if current_angles_deg is None or current_gripper_0to1 is None:
            return False

        desired_joint_angles = list(current_angles_deg)
        if target_joint_angles is not None:
            for index, angle_deg in enumerate(target_joint_angles):
                if angle_deg is not None:
                    desired_joint_angles[index] = float(np.clip(angle_deg, -180, 180))

        desired_gripper = (
            current_gripper_0to1 if target_gripper is None else float(target_gripper)
        )

        current_joint_angles = list(current_angles_deg)
        current_gripper_angle_deg = current_gripper_0to1 * self.MAX_GRIPPER_ANGLE_DEG
        desired_gripper_angle_deg = desired_gripper * self.MAX_GRIPPER_ANGLE_DEG

        for step_index, alpha in enumerate(np.linspace(0, 1, self.steps + 1)[1:]):
            interp_joint_angles = []
            for current_angle, desired_angle in zip(
                current_joint_angles, desired_joint_angles, strict=True
            ):
                interp_joint_angles.append(
                    current_angle * (1 - alpha) + desired_angle * alpha
                )
            interp_gripper_angle_deg = (
                current_gripper_angle_deg * (1 - alpha)
                + desired_gripper_angle_deg * alpha
            )
            try:
                if self.arm_backend == "sim":
                    self.sim_arm_client.send_joint_targets(
                        joint_names,
                        interp_joint_angles,
                        interp_gripper_angle_deg / self.MAX_GRIPPER_ANGLE_DEG,
                    )
                else:
                    action = {
                        motor_name + ".pos": interp_angle + (self.offset[index])
                        for index, (motor_name, interp_angle) in enumerate(
                            zip(joint_names[:-1], interp_joint_angles, strict=True)
                        )
                    }
                    action["gripper.pos"] = interp_gripper_angle_deg
                    self.arm.send_action(action)
                if step_callback is not None:
                    step_callback(
                        {
                            "joint_names": list(joint_names),
                            "target_joint_angles_deg": list(interp_joint_angles),
                            "target_gripper_open_0to1": float(
                                interp_gripper_angle_deg
                                / self.MAX_GRIPPER_ANGLE_DEG
                            ),
                            "final_target_joint_angles_deg": list(
                                desired_joint_angles
                            ),
                            "final_target_gripper_open_0to1": float(
                                desired_gripper
                            ),
                            "step_index": int(step_index),
                            "steps": int(self.steps),
                            "alpha": float(alpha),
                        }
                    )
            except Exception as e:
                print(f"设置机械臂角度失败: {e}")
                return False
            time.sleep(0.5 / self.steps)

        if self.arm_backend == "sim" and angles_deg is not None:
            try:
                self.wait_until_reached(desired_joint_angles)
            except TimeoutError as e:
                print(f"仿真机械臂未在超时内到达目标位姿: {e}")
                return False
        return True

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        """获取机械臂各关节角度和夹爪开合程度。

        Args:
            retry_times: 读取失败后的重试次数；`None` 表示使用默认配置。

        Returns:
            一个二元组 `(angles_deg, gripper_open_0to1)`：
            - `angles_deg` 为关节角度列表，单位为度；失败时为 `None`
            - `gripper_open_0to1` 为夹爪开合程度，范围为 `[0, 1]`；失败时为 `None`
        """
        raw_angles_deg, gripper_open_0to1 = self.get_raw_joint_angles(retry_times)
        if raw_angles_deg is None or gripper_open_0to1 is None:
            return None, None
        return [
            angle - offset
            for angle, offset in zip(raw_angles_deg, self.offset, strict=True)
        ], gripper_open_0to1

    def get_arm_pose(self) -> tuple[list[float] | None, list[float] | None]:
        angles_deg, _ = self.get_arm_angles()
        if angles_deg is None:
            return None, None
        fk: kinpy.Transform = self.chain.forward_kinematics(
            np.deg2rad(angles_deg).tolist()
        )  # type: ignore
        return (
            fk.pos.tolist(),
            R.from_euler("xyz", fk.rot_euler, degrees=True)
            .as_euler("zyx", degrees=True)
            .tolist(),
        )

    def disconnect_arm(self):
        if self.arm_backend == "sim":
            self.sim_arm_client.disconnect()
            return
        self.arm.disconnect()

    def enable_torque(self):
        if self.arm_backend == "sim":
            self.sim_arm_client.set_torque_enabled(True)
            return
        self.arm.bus.enable_torque()

    def disable_torque(self):
        if self.arm_backend == "sim":
            self.sim_arm_client.set_torque_enabled(False)
            return
        self.arm.bus.disable_torque()

    def move_to_home(
        self,
        gripper_open_0to1: float | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        return self.set_arm_angles(
            [0, 0, 0, 0, 0],
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )

    def move_to(
        self,
        pos: List[float],
        gripper_open_0to1: float | None = None,
        rot_rad: float | int | None = None,
        euler_angles_deg_zyx: list[float] | None = None,
        step_callback: StepCallback | None = None,
    ) -> bool:
        if not hasattr(self, "chain"):
            raise ValueError("没有机械臂模型，无法使用位置控制")
        if euler_angles_deg_zyx:
            print("Warning: not support euler_angles_deg_zyx currently in lerobo")
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")
        goal_tf = kinpy.Transform(
            pos=np.array(pos), rot=[0, 0, -rot_rad if rot_rad else 0]
        )
        angles_deg = minimize(
            self._ik_cost_function,
            x0=np.zeros(len(self.chain.get_joint_parameter_names())),
            args=(goal_tf.matrix(), self.chain),
            method="SLSQP",
        )
        if not angles_deg.success:
            print("逆运动学不收敛，无法到达指定位置")
            return False
        angles_deg = np.rad2deg(angles_deg.x).tolist()
        if not self.set_arm_angles(
            angles_deg,
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        ):
            return False
        return True

    def set_gripper(
        self,
        gripper_open_0to1: float,
        step_callback: StepCallback | None = None,
    ):
        self.set_arm_angles(
            gripper_open_0to1=gripper_open_0to1,
            step_callback=step_callback,
        )


if __name__ == "__main__":
    arm: LeroboArm = Arm()  # type: ignore
    time.sleep(1)
    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    angles, gripper = arm.get_arm_angles()
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    arm.set_arm_angles(None, gripper_open_0to1=0)
    time.sleep(1)
    angles, gripper = arm.get_arm_angles()
    print("机械臂角度:", angles)
    print("夹爪状态:", gripper)
    arm.move_to_home()
    time.sleep(1)
    arm.move_to([0.1, 0.1, 0.1])
    time.sleep(1)
    arm.move_to_home()
    time.sleep(1)
    arm.move_to([0.2, 0.2, 0.17])
    time.sleep(1)
    arm.move_to_home()
    time.sleep(1)
    arm.disconnect_arm()
