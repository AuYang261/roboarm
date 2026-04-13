import numpy as np
from config_getter import get_config_value


class Arm:
    _registry = {}

    def __new__(cls, *args, **kwargs):
        arm_type = get_config_value("arm_type")
        target_class = cls._registry.get(arm_type)
        if not target_class:
            raise ValueError(f"Unknown arm type in config: {arm_type}")

        return target_class(*args, **kwargs)


class ArmBase(Arm):

    def __init__(self):
        pass

    # 当有子类继承 ArmBase 时，这个方法会被自动触发
    def __init_subclass__(cls, arm_type, **kwargs):
        super().__init_subclass__(**kwargs)
        if arm_type is not None:
            # 自动将子类加入注册表
            cls._registry[arm_type] = cls

    def reset(self):
        """
        重置控制器状态，子类需要实现具体的重置逻辑
        """
        raise NotImplementedError(
            "reset method must be implemented in subclass")

    def send_a_step(self):
        """
        发送一步控制指令，子类需要实现具体的发送逻辑
        """
        raise NotImplementedError(
            "send_a_step method must be implemented in subclass")

    def set_joint(self, joint_id2positions: dict[str, float]) -> dict[str, float]:
        """
        设置关节位置，单位: 弧度
        返回超出关节限制的量
        """
        raise NotImplementedError(
            "set_joint method must be implemented in subclass")

    def add_joint(self, joint_id2action: dict[str, float]) -> dict[str, float]:
        """
        增量模式设置关节位置，单位: 弧度
        返回超出关节限制的量
        """
        raise NotImplementedError(
            "add_joint method must be implemented in subclass")

    def get_joint(self) -> np.ndarray:
        """
        不包括夹爪关节，单位: 弧度
        """
        raise NotImplementedError(
            "get_joint method must be implemented in subclass")

    def get_gripper(self) -> float:
        """
        获取夹爪位置
        单位: m
        """
        raise NotImplementedError(
            "get_gripper method must be implemented in subclass")

    def get_ee_pos(self) -> np.ndarray:
        """
        单位: m
        """
        raise NotImplementedError(
            "get_ee_pos method must be implemented in subclass")

    def get_ee_quat(self) -> np.ndarray:
        """
        获取末端执行器的姿态四元数
        """
        raise NotImplementedError(
            "get_ee_quat method must be implemented in subclass")

    def set_gripper(self, close: bool = True):
        """
        设置夹爪状态
        close: 是否闭合夹爪
        """
        raise NotImplementedError(
            "set_gripper method must be implemented in subclass")

    def render(self):
        """
        渲染控制器状态
        """
        raise NotImplementedError(
            "render method must be implemented in subclass")
