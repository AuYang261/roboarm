import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, List
from collections.abc import Sequence
from arm.arm_base import Arm
from config_getter import get_config_value
import time
import numpy as np
import piper_sdk
from piper_sdk import C_PiperInterface_V2
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import subprocess
import re


class PiperBySDK(Arm):
    FACTOR = 1000.0
    joint_num = 6
    # 默认末端朝前的欧拉角 [RZ, RY, RX]（度）
    DEFAULT_EULER_DEG = [0.0, 60.0, 0.0]
    # 默认末端朝下的欧拉角 [RZ, RY, RX]（度）
    DEFAULT_DOWN_EULER_DEG = [0.0, 150.0, 0.0]

    def __init__(self, move_mode_end_pose: bool = True, debug_mode=True):
        super().__init__()
        self.debug_mode = debug_mode
        self.move_mode_end_pose = move_mode_end_pose
        self.timeout = 10 if debug_mode else 5

        config_can_port = get_config_value("arm_port")
        try:
            self.piper = C_PiperInterface_V2(config_can_port)
            self.piper.ConnectPort()
        except:
            can_ports = self.activate_can()
            if config_can_port not in can_ports:
                raise RuntimeError(
                    f"arm port {config_can_port} in config not in scanned port({can_ports})"
                )
            self.piper = C_PiperInterface_V2(config_can_port)
            self.piper.ConnectPort()
        self.piper.JointConfig(clear_err=0xAE)
        if not self._enable_fun():
            print(self.piper.GetArmStatus())
            raise RuntimeError("Failed to enable Piper arm.")
        self.reset(self.move_mode_end_pose)

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
            start = time.time()
            while True:
                status = self.piper.GetArmStatus()
                if status.arm_status.motion_status == 0x00:
                    break
                if time.time() - start > self.timeout:
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
        if self.move_mode_end_pose:
            self.set_move_mode(move_mode_end_pose=False)

        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        start = time.time()
        while True:
            status = self.piper.GetArmStatus()
            if status.arm_status.motion_status == 0x00:
                break
            if time.time() - start > self.timeout:
                print("move_to_home 超时")
                break

        if gripper_open_0to1 is not None:
            self.set_gripper(gripper_open_0to1=gripper_open_0to1)

        if self.move_mode_end_pose:
            self.set_move_mode(move_mode_end_pose=True)

    def move_to(
        self,
        pos: list[float],
        gripper_open_0to1: float | None = None,
        rot_rad: float | int | None = None,
        euler_angles_deg_zyx: list[float] | None = None,
    ):
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")

        if not self.move_mode_end_pose:
            self.set_move_mode(move_mode_end_pose=True)
            self.move_mode_end_pose = True

        if not euler_angles_deg_zyx is None:
            self.set_ee_pose(position=pos, euler_angles_deg_zyx=euler_angles_deg_zyx)
        else:
            euler = list(self.DEFAULT_DOWN_EULER_DEG)
            if rot_rad is not None:
                euler[0] = np.degrees(rot_rad)

            self.set_ee_pose(position=pos, euler_angles_deg_zyx=euler)

        if gripper_open_0to1 is not None:
            self.set_gripper(gripper_open_0to1=gripper_open_0to1)

        return self.get_arm_angles()

    def disconnect_arm(self):
        print("Resetting piper arm to initial state.")
        try:
            self.reset()
        except TimeoutError as e:
            print(f"Error during reset: {e}")
        self.disable_torque()
        self.piper.DisconnectPort()
        print("Arm disconnected")

    def enable_torque(self):
        self.piper.EnableArm(7)
        self.piper.GripperCtrl(
            gripper_angle=0,
            gripper_effort=2000 if self.debug_mode else 5000,
            gripper_code=0x03,
        )
        print("Piper arm enabled.")

    def disable_torque(self):
        self.piper.DisableArm()
        self.piper.GripperCtrl(gripper_code=0x02)
        print("Piper arm disabled.")

    # ========== 内部方法 ==========

    def reset(self, move_mode_end_pose: bool | None = None):
        self.piper.JointConfig(clear_err=0xAE)
        self.piper.CrashProtectionConfig(0, 0, 0, 0, 0, 0)
        self.move_to_home(gripper_open_0to1=1)

        if move_mode_end_pose is not None:
            self.move_mode_end_pose = move_mode_end_pose
        if move_mode_end_pose:
            self.set_move_mode(move_mode_end_pose=True)

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

    def get_ee_euler_zyx(self) -> np.ndarray:
        # GetArmEndPoseMsgs获取到的欧拉角顺序是xyz
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "xyz",
            np.array(
                [end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis],
                dtype=np.float32,
            )
            / self.FACTOR,
            degrees=True,
        ).as_euler("zyx", degrees=True)

    def get_ee_quat(self) -> np.ndarray:
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        return R.from_euler(
            "xyz",
            np.array([end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis])
            / self.FACTOR,
            degrees=True,
        ).as_quat()

    def set_ee_pose(self, position: list[float], euler_angles_deg_zyx: list[float]):
        if self.move_mode_end_pose is False:
            raise RuntimeError("Cannot set end-effector pose in joint control mode.")
        position_scaled = np.array(position) * self.FACTOR * 1000.0
        euler_scaled = (
            R.from_euler("zyx", np.array(euler_angles_deg_zyx), degrees=True).as_euler(
                "xyz", degrees=True
            )
            * self.FACTOR
        )
        pass
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
            if time.time() - start > self.timeout:
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
            gripper_effort=2000 if self.debug_mode else 5000,
            gripper_code=0x03,
            set_zero=0,
        )

    def _enable_fun(self) -> bool:
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
            if time.time() - start_time > self.timeout:
                print("程序自动使能超时")
                return False
            time.sleep(1)

    def activate_can(self):
        """
        自动查找并激活机械臂对应的 CAN 端口。
        """
        base_dir = Path(piper_sdk.__file__).parent
        find_script = base_dir / "find_all_can_port.sh"
        activate_script = base_dir / "can_activate.sh"
        baudrate = "1000000"

        # 1. 检查脚本文件是否存在
        if not find_script.exists() or not activate_script.exists():
            raise FileNotFoundError(f"找不到 Shell 脚本，请检查路径: {base_dir}")

        try:
            # 2. 赋予脚本可执行权限 (chmod +x)
            subprocess.run(["chmod", "+x", str(find_script)], check=True)
            subprocess.run(["chmod", "+x", str(activate_script)], check=True)
            print("✅ 脚本执行权限已配置。")

            # 3. 运行第一个脚本获取端口列表
            print("🔍 正在扫描 CAN 端口...")
            result = subprocess.run(
                ["sudo", str(find_script)], capture_output=True, text=True, check=True
            )

            # 4. 解析输出结果
            # 正则表达式匹配类似 "can4" 和 "1-4.2:1.0" (含有横杠和冒号的 USB 拓扑路径)
            # 忽略 ".mttcan" 结尾的端口
            pattern = re.compile(
                r"Interface (can\d+) is connected to USB port (\d+-[\d\.]+:\d+\.\d+)"
            )
            matches = pattern.findall(result.stdout)

            ret = []
            if not matches:
                print("⚠️ 未找到匹配的机械臂 CAN 端口 (未匹配到类似 1-4.2:1.0 的拓扑)。")
                print("原始输出如下:\n", result.stdout)
                return ret

            # 5. 遍历匹配结果，运行激活脚本
            for can_iface, usb_port in matches:
                print(f"🎯 发现机械臂端口: {can_iface} -> {usb_port}，准备激活...")

                # 拼接并执行命令: ./can_activate.sh can4 1000000 1-4.2:1.0
                activate_cmd = [
                    "sudo",
                    str(activate_script),
                    can_iface,
                    baudrate,
                    usb_port,
                ]

                subprocess.run(activate_cmd, check=True)
                print(f"🚀 成功激活 {can_iface} ({usb_port})！\n")
                ret.append(can_iface)

        except subprocess.CalledProcessError as e:
            print(f"❌ 运行 Shell 脚本时出错!")
            print(f"命令: {e.cmd}")
            print(f"错误输出: {e.stderr if e.stderr else '无详细错误信息'}")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")
        return ret

    def set_move_mode(self, move_mode_end_pose: bool):
        start = time.time()
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x0 if move_mode_end_pose else 0x01,
            move_spd_rate_ctrl=50 if self.debug_mode else 100,
            is_mit_mode=0x00,
        )
        while True:
            status = self.piper.GetArmStatus()
            if status.arm_status.mode_feed == (0x00 if move_mode_end_pose else 0x01):
                break
            if time.time() - start > self.timeout:
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
    print("末端位置:", np.array(arm.get_arm_pos()).round(2).tolist())
    print("末端zyx:", arm.get_ee_euler_zyx().round(2).tolist())

    arm.move_to([0.2, 0.0, 0.2], gripper_open_0to1=0, rot_rad=np.pi / 6)
    time.sleep(2)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", np.array(arm.get_arm_pos()).round(2).tolist())
    print("末端zyx:", arm.get_ee_euler_zyx().round(2).tolist())

    arm.move_to_home(gripper_open_0to1=1)
    time.sleep(2)
    print("关节角度:", arm.get_arm_angles())
    print("末端位置:", np.array(arm.get_arm_pos()).round(2).tolist())
    print("末端zyx:", arm.get_ee_euler_zyx().round(2).tolist())

    arm.disconnect_arm()
