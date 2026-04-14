import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, List
from collections.abc import Sequence
from arm.arm_base import Arm
from config_getter import get_config_value
import time
import numpy as np
from piper_sdk import C_PiperInterface_V2

from scipy.spatial.transform import Rotation as R


class PiperBySDK(Arm):
    FACTOR = 1000.0
    joint_num = 6
    # 默认末端朝下的欧拉角 [RX, RY, RZ]（度）
    DEFAULT_EULER_DEG = [0.0, 170.0, 0.0]

    def __init__(self, move_mode_end_pose: bool = False, debug_mode=True):
        super().__init__()
        self.debug_mode = debug_mode
        self.move_mode_end_pose = move_mode_end_pose

        self.piper = C_PiperInterface_V2(get_config_value("arm_port"))
        self.piper.ConnectPort()
        self.piper.JointConfig(clear_err=0xAE)
        # self.piper.JointConfig(set_zero=0xAE, clear_err=0xAE)
        if not self._enable_fun():
            print(self.piper.GetArmStatus())
            raise RuntimeError("Failed to enable Piper arm.")
        self.reset(self.move_mode_end_pose)

    def __del__(self):
        print("Resetting Piper arm to initial state.")
        try:
            self.reset(move_mode_end_pose=False)
        except TimeoutError as e:
            print(f"Error during reset: {e}")
        self.disable()

    # ========== 高层接口实现 ==========

    def set_arm_angles(
        self,
        angles_deg: Sequence[float | int] | None = None,
        gripper_open_0to1: float | int | None = None,
    ) -> bool:
        if gripper_open_0to1 is not None:
            self.set_gripper(gripper_open_0to1=gripper_open_0to1)

        if angles_deg is not None:
            if len(angles_deg) != self.joint_num:
                print(
                    f"关节角度数量错误，期望{self.joint_num}个，实际{len(angles_deg)}个"
                )
                return False
            was_end_pose = self.move_mode_end_pose
            if was_end_pose:
                self.set_move_mode(move_mode_end_pose=False)
                self.move_mode_end_pose = False

            ctrl = np.array(angles_deg) * self.FACTOR
            self.piper.JointCtrl(
                joint_1=int(ctrl[0]),
                joint_2=int(ctrl[1]),
                joint_3=int(ctrl[2]),
                joint_4=int(ctrl[3]),
                joint_5=int(ctrl[4]),
                joint_6=int(ctrl[5]),
            )
            timeout = 10 if self.debug_mode else 5
            start = time.time()
            while True:
                status = self.piper.GetArmStatus()
                if status.arm_status.motion_status == 0x00:
                    break
                if time.time() - start > timeout:
                    print("set_arm_angles 超时")
                    break

            if was_end_pose:
                self.set_move_mode(move_mode_end_pose=True)
                self.move_mode_end_pose = True

        return True

    def get_arm_angles(
        self, retry_times=None
    ) -> tuple[Union[List[float], None], Union[float, None]]:
        try:
            joints = self.piper.GetArmJointMsgs()
            angles_deg = [
                joints.joint_state.joint_1 / self.FACTOR,
                joints.joint_state.joint_2 / self.FACTOR,
                joints.joint_state.joint_3 / self.FACTOR,
                joints.joint_state.joint_4 / self.FACTOR,
                joints.joint_state.joint_5 / self.FACTOR,
                joints.joint_state.joint_6 / self.FACTOR,
            ]
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_deg = gripper_msgs.gripper_state.grippers_angle / self.FACTOR
            return angles_deg, gripper_deg
        except Exception:
            if retry_times is None:
                retry_times = self.get_arm_angles_retry_times
            if retry_times > 0:
                time.sleep(self.catch_time_interval_s)
                return self.get_arm_angles(retry_times - 1)
            return None, None

    def get_arm_pos(self) -> list[float] | None:
        try:
            return self.get_ee_pos().tolist()
        except Exception:
            return None

    def move_to_home(self, gripper_open_0to1: float | None = None):
        was_end_pose = self.move_mode_end_pose
        if was_end_pose:
            self.set_move_mode(move_mode_end_pose=False)
            self.move_mode_end_pose = False

        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        timeout = 10 if self.debug_mode else 5
        start = time.time()
        while True:
            status = self.piper.GetArmStatus()
            if status.arm_status.motion_status == 0x00:
                break
            if time.time() - start > timeout:
                print("move_to_home 超时")
                break

        if gripper_open_0to1 is not None:
            self.set_gripper(gripper_open_0to1=gripper_open_0to1)

        if was_end_pose:
            self.set_move_mode(move_mode_end_pose=True)
            self.move_mode_end_pose = True

    def move_to(
        self,
        pos: list[float],
        gripper_open_0to1: float | None = None,
        rot_rad: float | int | None = None,
    ):
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")

        if not self.move_mode_end_pose:
            self.set_move_mode(move_mode_end_pose=True)
            self.move_mode_end_pose = True

        euler = list(self.DEFAULT_EULER_DEG)
        if rot_rad is not None:
            euler[2] = np.degrees(rot_rad)

        self.set_ee_pose(position=pos, euler_angles=euler)

        if gripper_open_0to1 is not None:
            self.set_gripper(gripper_open_0to1=gripper_open_0to1)

        return self.get_arm_angles()

    def disconnect_arm(self):
        self.disable()

    def enable_torque(self):
        self.piper.EnableArm(7)

    def disable_torque(self):
        self.piper.DisableArm()

    # ========== 内部方法 ==========

    def reset(self, move_mode_end_pose: bool | None = None, timeout: int = 5):
        if self.debug_mode:
            timeout *= 2
        self.piper.JointConfig(clear_err=0xAE)
        self.piper.CrashProtectionConfig(0, 0, 0, 0, 0, 0)
        self.set_move_mode(move_mode_end_pose=False, timeout=timeout)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        start = time.time()
        while True:
            status = self.piper.GetArmStatus()
            if status.arm_status.motion_status == 0x00:
                break
            if time.time() - start > timeout:
                raise TimeoutError("Failed to reset within the specified timeout.")
        self.set_gripper(gripper_open_0to1=1)

        if move_mode_end_pose is None:
            move_mode_end_pose = self.move_mode_end_pose
        else:
            self.move_mode_end_pose = move_mode_end_pose
        if move_mode_end_pose:
            self.set_move_mode(move_mode_end_pose=True, timeout=timeout)

    def get_ee_pos(self) -> np.ndarray:
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return (
            np.array(
                [end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis],
                dtype=np.float32,
            )
            / self.FACTOR
            / 1000.0
        )

    def get_ee_euler(self) -> np.ndarray:
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return (
            np.array(
                [end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis],
                dtype=np.float32,
            )
            / self.FACTOR
        )

    def get_ee_quat(self) -> np.ndarray:
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "zyx",
            np.array([end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis])
            / self.FACTOR,
            degrees=True,
        ).as_quat()

    def set_ee_pose(
        self, position: list[float], euler_angles: list[float], timeout: int = 5
    ):
        if self.move_mode_end_pose is False:
            raise RuntimeError("Cannot set end-effector pose in joint control mode.")
        position_scaled = np.array(position) * self.FACTOR * 1000.0
        euler_scaled = np.array(euler_angles) * self.FACTOR
        self.piper.EndPoseCtrl(
            X=int(position_scaled[0]),
            Y=int(position_scaled[1]),
            Z=int(position_scaled[2]),
            RX=int(euler_scaled[0]),
            RY=int(euler_scaled[1]),
            RZ=int(euler_scaled[2]),
        )

        start = time.time()
        while True:
            status = self.piper.GetArmStatus().arm_status
            if status.arm_status != 0x0:
                print(self.arm_status2str(status.arm_status))
                break
            if status.motion_status == 0x00:
                break
            if time.time() - start > timeout:
                print("set_ee_pose 超时")
                break

    def set_gripper(self, gripper_open_0to1: float):
        """
        夹爪开闭程度，越大越开，0~1
        """
        if not 0 <= gripper_open_0to1 <= 1:
            raise ValueError("gripper_open_0to1 must in [0, 1]")
        self.piper.GripperCtrl(
            int(gripper_open_0to1 * 100 * self.FACTOR),
            gripper_effort=1000,
            gripper_code=0x01,
            set_zero=0,
        )

    def _enable_fun(self) -> bool:
        timeout = 5
        start_time = time.time()
        while True:
            print("--------------------")
            self.piper.EnableArm(7)
            msgs = self.piper.GetArmLowSpdInfoMsgs()
            enable_flag = (
                msgs.motor_1.foc_status.driver_enable_status
                and msgs.motor_2.foc_status.driver_enable_status
                and msgs.motor_3.foc_status.driver_enable_status
                and msgs.motor_4.foc_status.driver_enable_status
                and msgs.motor_5.foc_status.driver_enable_status
                and msgs.motor_6.foc_status.driver_enable_status
            )
            print("使能状态:", enable_flag)
            print("--------------------")
            if enable_flag:
                return True
            if time.time() - start_time > timeout:
                print("程序自动使能超时")
                return False
            time.sleep(1)

    def disable(self):
        self.piper.DisableArm()
        self.piper.GripperCtrl(0, 1000, 0x02, 0)
        self.piper.DisconnectPort()
        print("Piper arm disabled.")

    def set_move_mode(self, move_mode_end_pose: bool, timeout: int = 5):
        start = time.time()
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x0 if move_mode_end_pose else 0x01,
            move_spd_rate_ctrl=10 if self.debug_mode else 100,
            is_mit_mode=0x00,
        )
        while True:
            status = self.piper.GetArmStatus()
            if status.arm_status.mode_feed == (0x00 if move_mode_end_pose else 0x01):
                break
            if time.time() - start > timeout:
                raise TimeoutError(
                    "Failed to switch move mode within the specified timeout."
                )

    @staticmethod
    def arm_status2str(status):
        status_dict = {
            0x00: "正常",
            0x01: "急停",
            0x02: "无解",
            0x03: "奇异点",
            0x04: "目标角度超过限",
            0x05: "关节通信异常",
            0x06: "关节抱闸未打开",
            0x07: "机械臂发生碰撞",
            0x08: "拖动示教时超速",
            0x09: "关节状态异常",
            0x0A: "其它异常",
            0x0B: "示教记录",
            0x0C: "示教执行",
            0x0D: "示教暂停",
            0x0E: "主控NTC过温",
            0x0F: "释放电阻NTC过温",
        }
        return status_dict.get(status, "未知状态: " + hex(status))


if __name__ == "__main__":
    arm: PiperBySDK = Arm(debug_mode=False)
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", arm.get_arm_pos())

    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", arm.get_arm_pos())

    arm.move_to([0.3, 0.2, 0.2], gripper_open_0to1=1, rot_rad=0)
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", arm.get_arm_pos())

    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", arm.get_arm_pos())

    arm.disconnect_arm()
