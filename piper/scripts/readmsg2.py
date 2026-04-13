# can_piper_left
import time
# 导入piper_sdk模块
from piper_sdk import *
 
from typing import (
    Optional,
)


if __name__ == "__main__":
    # 实例化interface，形参的默认参数如下
    # can_name(str -> default 'can0'): can port name
    # judge_flag(bool -> default True): 创建该实例时是否开启判断can模块，若使用的是非官方模块，请将其设置False
    # can_auto_init(bool): 创建该实例时是否自动进行初始化来打开can bus，如果设置为False，请在ConnectPort参数中将can_init形参设置为True
    # dh_is_offset([0,1] -> default 0x01): 使用的dh参数是新版dh还是旧版dh，S-V1.6-3以前的为旧版，S-V1.6-3固件及以后的为新版，对应fk的计算
    #             0 -> old
    #             1 -> new
    # start_sdk_joint_limit(bool -> default False):是否开启SDK的关节角度限位，会对反馈消息和控制消息都做限制
    # start_sdk_gripper_limit(bool -> default False):是否开启SDK的夹爪位置限位，会对反馈消息和控制消息都做限制
    # logger_level(LogLevel -> default LogLevel.WARNING):设定log的日志等级
    #         参数有如下可选: 
    #               LogLevel.DEBUG
    #               LogLevel.INFO
    #               LogLevel.WARNING
    #               LogLevel.ERROR
    #               LogLevel.CRITICAL
    #               LogLevel.SILENT
    # log_to_file(bool -> default False):是否打开log写入文件功能，True则打开，默认关闭
    # log_file_path(str -> default False):设定log写入文件的路径，默认在sdk路径下的log文件夹
    # piper = C_PiperInterface(can_name="can_piper_left",
    #                         judge_flag=False,
    #                         can_auto_init=True,
    #                         dh_is_offset=1,
    #                         start_sdk_joint_limit=False,
    #                         start_sdk_gripper_limit=False,
    #                         logger_level=LogLevel.WARNING,
    #                         log_to_file=False,
    #                         log_file_path=None)
    piper = C_PiperInterface("can4")
 
    # print(piper.)
    # 开启can收发线程
    piper.ConnectPort()
    
    time.sleep(0.1) # 需要时间去读取固件反馈帧，否则会反馈-0x4AF = -1199
    print(piper.GetPiperFirmwareVersion())
    time.sleep(0.1)
    print(piper.GetPiperFirmwareVersion())
    time.sleep(0.1)
 
    # 循环打印消息，注意所有的消息第一帧都是默认数值，比如关节角消息第一帧的消息内容默认为0
    i = 10
    while i > 0:
        # a.SearchMotorMaxAngleSpdAccLimit(1, 1)
        # a.ArmParamEnquiryAndConfig(1,0,2,0,3)
        # a.GripperCtrl(50000,1500,0x01)
        print(piper.GetArmJointMsgs())
        print(piper.GetArmGripperMsgs())
        time.sleep(0.005)# 200hz
        i = i - 1
 