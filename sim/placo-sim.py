import numpy as np
import kinpy as kp
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz
from placo_utils.tf import tf

"""
6axis robot following an infinity sign (∞) trajectory
"""

# 加载 URDF，为了可视化继续使用 placo 的 RobotWrapper（仅用于显示）
try:
    import placo
    robot = placo.RobotWrapper("urdf/low_cost_robot.urdf", placo.Flags.ignore_collisions)
except Exception:
    robot = None

# 使用 kinpy 构建串联链（以末端执行器链接为目标）
with open("urdf/low_cost_robot.urdf", "r", encoding="utf-8") as f:
    urdf_txt = f.read()

# 末端执行器链接名称（来自 URDF）
EE_LINK = "gripper_moving_1"
chain = kp.build_serial_chain_from_urdf(urdf_txt, EE_LINK)

viz = robot_viz(robot) if robot is not None else None
t = 0
dt = 0.01
last_targets = []
last_target_t = 0

# 初始关节角（若可视化存在，从 placo 读取，否则用零）
if robot is not None:
    q = np.array(robot.state.q, dtype=float)
else:
    q = np.zeros(len(chain.joint_names), dtype=float)

@schedule(interval=dt)
def loop():
    global t, last_targets, last_target_t
    t += dt

    # 更新末端目标（位置）
    target = np.array([np.cos(t) * 0.05, 0.1, 0.1 + np.sin(2 * t) * 0.03])

    # 当前姿态（使用 kinpy 前向运动学）
    q_dict = {name: float(val) for name, val in zip(chain.joint_names, q)}
    T_ee = chain.forward_kinematics(q_dict)[EE_LINK]
    p = T_ee[:3, 3]

    # 位置误差与雅可比（仅用平移部分进行速度 IK）
    e = target - p
    J = chain.jacobian(q_dict)  # 6xN
    J_pos = J[:3, :]
    # 伪逆求关节速度，加入简单增益与限幅
    lam = 1e-4  # 阻尼项，提升数值稳定性
    JT = J_pos.T
    JJ = J_pos @ JT
    dq = JT @ np.linalg.solve(JJ + lam * np.eye(3), e * 10.0)  # 增益 10.0
    dq = np.clip(dq, -0.5, 0.5)  # 限制单步速度幅度
    q = q + dq * dt

    # 可视化更新（若可用）
    if robot is not None and viz is not None:
        robot.state.q = list(q)
        robot.update_kinematics()
        viz.display(robot.state.q)
        robot_frame_viz(robot, EE_LINK)
        frame_viz("target", tf.translation_matrix(target.tolist()))

    # Drawing the last 50 targets (adding one point every 100ms)
    if t - last_target_t > 0.1:
        last_target_t = t
        last_targets.append(target)
        last_targets = last_targets[-50:]
        points_viz("targets", last_targets, color=0xaaff00)


run_loop()
