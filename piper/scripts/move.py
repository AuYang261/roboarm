#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
from pathlib import Path
import time
import piper_sdk
from piper_sdk import *
import sys
import subprocess
import re


def activate_can():
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
            activate_cmd = ["sudo", str(activate_script), can_iface, baudrate, usb_port]

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


def enable_fun(piper:C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)

if __name__ == "__main__":
    can = activate_can()
    if len(can) == 0:
        raise RuntimeError("No can port")
    piper = C_PiperInterface(can[0])
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    # piper.DisableArm(7)
    piper.GripperCtrl(0,1000,0x01, 0)
    factor = 57324.840764 #1000*180/3.14
    position = [0,0,0,0,0,0,0]
    count = 0
    while True:
        print(piper.GetArmStatus())
        import time
        count  = count + 1
        # print(count)
        if(count == 0):
            print("1-----------")
            position = [0,0,0,0,0,0,0]
        elif(count == 500):
            print("2-----------")
            position = [0.2,0.2,-0.2,0.3,-0.2,0.5,0.08]
        elif(count == 1000):
            print("1-----------")
            position = [0,0,0,0,0,0,0]
            count = 0

        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        joint_6 = round(position[6]*1000*1000)
        # piper.MotionCtrl_1()
        piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        time.sleep(0.005)
