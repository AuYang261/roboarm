"""
MuJoCo simulation for robotic arm with URDF.
The arm draws a figure-8 trajectory.
"""

import sys
import os
import mujoco
import mujoco.viewer
import time
import numpy as np
from threading import Lock
import json
import math
import time
import cv2

MODEL_HOMING = [27, 1094, 126, 994, 62, 79] # 对应 xml 模型 urdf\meshes\mjmodel.xml
    
try:
    from pynput import keyboard
except ImportError:
    import types
    keyboard = types.ModuleType('keyboard')
    class Key:
        up = 'up'; down = 'down'; left = 'left'; right = 'right'
        esc = 'esc'; space = 'space'
    keyboard.Key = Key
    class _DummyListener:
        def __init__(self, **_): pass
        def start(self): pass
        def stop(self): pass
    keyboard.Listener = _DummyListener
    sys.modules['pynput.keyboard'] = keyboard

class KeyboardController:
    """键盘控制器，使用 pynput 监听按键状态"""

    def __init__(self):
        self.keys_pressed = {
            'up': False, 'down': False, 'left': False, 'right': False, ' ': False
        }
        self.lock = Lock()
        self.listener = None

    def on_press(self, key):
        try:
            k = key.char.lower()
            if k in self.keys_pressed:
                with self.lock:
                    self.keys_pressed[k] = True
        except AttributeError:
            mapping = {
                keyboard.Key.up: 'up', keyboard.Key.down: 'down',
                keyboard.Key.left: 'left', keyboard.Key.right: 'right',
                keyboard.Key.space: ' '
            }
            k = mapping.get(key)
            if k:
                with self.lock:
                    self.keys_pressed[k] = True

    def on_release(self, key):
        if key == keyboard.Key.esc:
            return False
        try:
            k = key.char.lower()
            if k in self.keys_pressed:
                with self.lock:
                    self.keys_pressed[k] = False
        except AttributeError:
            mapping = {
                keyboard.Key.up: 'up', keyboard.Key.down: 'down',
                keyboard.Key.left: 'left', keyboard.Key.right: 'right',
                keyboard.Key.space: ' '
            }
            k = mapping.get(key)
            if k:
                with self.lock:
                    self.keys_pressed[k] = False

    def get(self, key):
        with self.lock:
            return self.keys_pressed.get(key, False)

    def start(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        return self

    def stop(self):
        if self.listener:
            self.listener.stop()

class SimMujocoModel:
    """MuJoCo机械臂仿真器，封装了仿真循环、控制、视角和稳定模式"""
    # 初始化
    def __init__(self, urdf_path):
        self.urdf_path = os.path.abspath(urdf_path)
        self._added_objs = []
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)
        # 加载 "home" keyframe，将所有关节和控制器初始化为 0
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        # print(f"Model loaded: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.lookat = [0.0, 0.0, 0.0]
        self.viewer.cam.elevation = -20.0
        self.viewer.cam.azimuth = 135.0

        self.dt = self.model.opt.timestep
        self.position_speed = 100  # 弧度/秒

        # 目标关节角度（弧度）
        self.target_joint_positions = (
            self.data.qpos[:self.model.nu].copy() if self.model.nu > 0 else np.array([])
        )
        for i in range(len(self.target_joint_positions)):
            self.target_joint_positions[i] = np.clip(self.target_joint_positions[i], -3.14159, 3.14159)

        # PD 控制参数
        self.kp = 15.0
        self.kd = 1.0
        self.stability_mode = False

        # 缓存末端 body ID
        self._last_joint_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_static_1")
        self._gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_moving_1")

        # 视角字典 {name: {distance, lookat, elevation, azimuth, follow_body?}}
        self.views = {}
        self.current_view = None  # 当前激活的视角名称

    # 设置移动速度
    def set_speed(self, speed):
        self.position_speed = speed

    # 输入关节角度列表，单位为弧度
    def set_joint_angles(self, joint_angles):
        n = min(len(joint_angles), self.model.nu)
        for i in range(n):
            self.target_joint_positions[i] = np.clip(joint_angles[i], -3.14159, 3.14159)

    # 添加视角
    def add_view(self, view_name, view_params):
        """
        添加固定视角或跟随视角。

        view_params: dict
          固定视角: {distance, lookat, elevation, azimuth}
          跟随视角（TRACKING）: {distance, elevation, azimuth, follow_body}
            - follow_body: str 或 int，跟随 body 位置但不跟随旋转
          固定相机视角（FIXED）: {fixed_camera: str}
            - fixed_camera: XML 中定义的 <camera name="..."/>，跟随 body 位置和旋转
        """
        if 'follow_body' in view_params:
            fb = view_params['follow_body']
            if isinstance(fb, str):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, fb)
                if body_id < 0:
                    print(f"警告: 未找到 body '{fb}'")
                view_params = dict(view_params, _follow_body_id=body_id)
            else:
                view_params = dict(view_params, _follow_body_id=int(fb))
        if 'fixed_camera' in view_params:
            cam_name = view_params['fixed_camera']
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                print(f"警告: 未找到 camera '{cam_name}'")
            view_params = dict(view_params, _fixed_camera_id=cam_id)
        self.views[view_name] = view_params

    # 切换视角
    def switch_view(self, view_name):
        if view_name not in self.views:
            print(f"未找到视角: {view_name}")
            return
        self.current_view = view_name
        p = self.views[view_name]
        if 'fixed_camera' in p:
            # 固定相机视角：使用 XML 中定义的相机，跟随 body 位置和旋转
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = p.get('_fixed_camera_id', -1)
        elif 'follow_body' in p:
            # 跟随视角：TRACKING 模式，跟随位置但不跟随旋转
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = p.get('_follow_body_id', -1)
            self.viewer.cam.distance  = p.get('distance',  0.3)
            self.viewer.cam.elevation = p.get('elevation', -20.0)
            self.viewer.cam.azimuth   = p.get('azimuth',   135.0)
        else:
            # 固定视角：FREE 模式
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.distance  = p.get('distance',  3.0)
            self.viewer.cam.elevation = p.get('elevation', -20.0)
            self.viewer.cam.azimuth   = p.get('azimuth',   135.0)
            self.viewer.cam.lookat = list(p.get('lookat', [0.0, 0.0, 0.0]))

    # 更新跟随视角（每帧调用）
    def update_camera(self):
        """TRACKING / FIXED 模式由 MuJoCo 自动处理"""
        pass

    # 切换稳定模式（增加阻尼，减少晃动）
    def toggle_stable_mode(self, enable):
        self.stability_mode = enable
        self.kd = 2.0 if enable else 1.0
        print(f"稳定模式: {'启用' if enable else '关闭'}")

    def joint_angles_to_poses(self, joint_angles):
        """
        正运动学：根据关节角度计算末端位姿。

        Parameters
        ----------
        joint_angles : list[float]
            各关节角度（弧度），长度应与 model.nq 一致。

        Returns
        -------
        dict
            {
                'last_joint': {'pos': np.ndarray(3), 'quat': np.ndarray(4)},
                'gripper':    {'pos': np.ndarray(3), 'quat': np.ndarray(4)},
            }
            last_joint — gripper_static_1（最后一个机械臂关节 joint5 所在连杆）
            gripper    — gripper_moving_1（实际工作点）
            quat 为 MuJoCo 格式 [w, x, y, z]
        """
        tmp_data = mujoco.MjData(self.model)
        n = min(len(joint_angles), self.model.nq)
        for i in range(n):
            tmp_data.qpos[i] = joint_angles[i]
        mujoco.mj_forward(self.model, tmp_data)

        return {
            'last_joint': {
                'pos':  tmp_data.xpos[self._last_joint_body_id].copy(),
                'quat': tmp_data.xquat[self._last_joint_body_id].copy(),
            },
            'gripper': {
                'pos':  tmp_data.xpos[self._gripper_body_id].copy(),
                'quat': tmp_data.xquat[self._gripper_body_id].copy(),
            },
        }

    def get_current_poses(self):
        """
        从当前仿真状态直接读取末端位姿（无需额外计算）。

        Returns
        -------
        dict  （格式同 joint_angles_to_poses）
        """
        return {
            'last_joint': {
                'pos':  self.data.xpos[self._last_joint_body_id].copy(),
                'quat': self.data.xquat[self._last_joint_body_id].copy(),
            },
            'gripper': {
                'pos':  self.data.xpos[self._gripper_body_id].copy(),
                'quat': self.data.xquat[self._gripper_body_id].copy(),
            },
        }

    def close(self):
        if hasattr(self, 'viewer') and self.viewer.is_running():
            self.viewer.close()

    def add_obj(self, obj_xml_path=None, pos=None):
        """在当前场景中动态添加一个物体，默认加载 urdf/meshes/cube.xml"""
        if obj_xml_path is None:
            obj_xml_path = os.path.join(os.path.dirname(self.urdf_path), "cube.xml")
        if pos is None:
            #TODO 从 xml 中读
            pos = [-0.2, 0.0, 0.05]

        import xml.etree.ElementTree as ET

        main_tree = ET.parse(self.urdf_path)
        main_root = main_tree.getroot()
        main_wb = main_root.find("worldbody")

        obj_tree = ET.parse(obj_xml_path)
        obj_wb = obj_tree.getroot().find("worldbody")

        obj_count = len(self._added_objs)
        for body in list(obj_wb):
            orig_name = body.get("name", "obj")
            suffix = f"_{obj_count}"
            body.set("name", orig_name + suffix)
            body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
            for child in body.iter():
                if child.get("name"):
                    child.set("name", child.get("name") + suffix)
            main_wb.append(body)

        tmp_path = self.urdf_path + ".tmp.xml"
        main_tree.write(tmp_path, xml_declaration=False)

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()
        self.model = mujoco.MjModel.from_xml_path(tmp_path)
        self.data = mujoco.MjData(self.model)
        os.remove(tmp_path)

        n = min(len(saved_qpos), self.model.nq)
        self.data.qpos[:n] = saved_qpos[:n]
        self.data.qvel[:min(len(saved_qvel), self.model.nv)] = saved_qvel[:min(len(saved_qvel), self.model.nv)]
        self.data.ctrl[:min(len(saved_ctrl), self.model.nu)] = saved_ctrl[:min(len(saved_ctrl), self.model.nu)]
        mujoco.mj_forward(self.model, self.data)

        self._last_joint_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_static_1")
        self._gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_moving_1")

        self.viewer.close()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.current_view:
            self.switch_view(self.current_view)

        self._added_objs.append({"xml": obj_xml_path, "pos": pos})
        print(f"已添加物体: {os.path.basename(obj_xml_path)} 位置={pos}")
    
    def reset(self):
        """重置仿真状态，移除所有动态添加的物体，恢复初始模型"""
        if self._added_objs:
            self.model = mujoco.MjModel.from_xml_path(self.urdf_path)
            self.data = mujoco.MjData(self.model)
            self._last_joint_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_static_1")
            self._gripper_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_moving_1")
            self._added_objs.clear()
            self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self.target_joint_positions = (
            self.data.qpos[:self.model.nu].copy() if self.model.nu > 0 else np.array([])
        )
        mujoco.mj_forward(self.model, self.data)

        if self.current_view:
            self.switch_view(self.current_view)
        print("仿真已重置")

def _apply_pd_control(sim):
    """PD 控制：将 target_joint_positions 写入 data.ctrl"""
    for i in range(sim.model.nu):
        adjusted = sim.target_joint_positions[i] + (sim.kd / sim.kp) * (-sim.data.qvel[i])
        sim.data.ctrl[i] = np.clip(adjusted, -3.14159, 3.14159)

def main():
    """主函数，创建仿真模型并运行"""
    model_path = 'urdf/meshes/mjmodel_opt.xml'
    sim = SimMujocoModel(model_path)

    # 注册视角
    sim.add_view('default', {'distance': 3.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -20.0, 'azimuth': 135.0})
    
    """
        distance: 0.3977693502162603
        lookat: [0.04549377 0.00060337 0.17765481]
        elevation: -89.0
        azimuth: 90.75984990619148
        trackbodyid: -1
        fixedcamid: -1
    """
    sim.add_view('top',     {'distance': 0.39, 'lookat': [0.04, 0.0, 0.17], 'elevation': -89.0, 'azimuth': 90.0})
    sim.switch_view('default')

    # 创建离屏渲染器和两个相机视角
    render_w, render_h = 640, 480
    renderer = mujoco.Renderer(sim.model, height=render_h, width=render_w)

    # Top 视角相机：俯视
    top_cam = mujoco.MjvCamera()
    top_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_cam.distance = 0.39
    top_cam.lookat[:] = [0.04, 0.0, 0.17]
    top_cam.elevation = -89.0
    top_cam.azimuth = 90.0

    # Follow 视角相机：跟随夹爪（使用 XML 中定义的 gripper_cam）
    follow_cam = mujoco.MjvCamera()
    follow_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    follow_cam.fixedcamid = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_cam")

    kb = KeyboardController().start()
    print("\n=== 键盘控制 ===")
    print("上/下键: 切换关节  左/右键: 调整角度  空格: 重置  ESC: 退出\n")

    current_joint = 0
    prev_up = prev_down = prev_space = False

    # 运行仿真循环——通过 up/down 控制关节id，left/right 控制关节移动角度（弧度）
    while True:
        if not sim.viewer.is_running():
            break

        up    = kb.get('up')
        down  = kb.get('down')
        space = kb.get(' ')

        # 上升沿检测：切换关节
        if up and not prev_up:
            current_joint = (current_joint - 1) % max(sim.model.nu, 1)
            print(f"选中关节 {current_joint}: {sim.target_joint_positions[current_joint]:.3f} rad")
        if down and not prev_down:
            current_joint = (current_joint + 1) % max(sim.model.nu, 1)
            print(f"选中关节 {current_joint}: {sim.target_joint_positions[current_joint]:.3f} rad")

        # 持续控制选中关节角度
        moving = False
        if sim.model.nu > 0:
            if kb.get('left'):
                sim.target_joint_positions[current_joint] = np.clip(
                    sim.target_joint_positions[current_joint] + sim.position_speed * sim.dt,
                    -3.14159, 3.14159)
                moving = True
            if kb.get('right'):
                sim.target_joint_positions[current_joint] = np.clip(
                    sim.target_joint_positions[current_joint] - sim.position_speed * sim.dt,
                    -3.14159, 3.14159)
                moving = True

        if moving:
            poses = sim.get_current_poses()
            lj = poses['last_joint']
            gr = poses['gripper']
            print(f"[last_joint ] pos={np.array2string(lj['pos'], precision=4, suppress_small=True)}  quat={np.array2string(lj['quat'], precision=4, suppress_small=True)}")
            print(f"[gripper    ] pos={np.array2string(gr['pos'], precision=4, suppress_small=True)}  quat={np.array2string(gr['quat'], precision=4, suppress_small=True)}")

        # 空格重置
        if space and not prev_space:
            mujoco.mj_resetData(sim.model, sim.data)
            sim.target_joint_positions = sim.data.qpos[:sim.model.nu].copy()
            print("位置重置")
            cam = sim.viewer.cam
            print("type:", cam.type)           # 枚举：mujoco.mjtCamera.mjCAMERA_FREE/TRACKING/FIXED
            print("distance:", cam.distance)
            print("lookat:", cam.lookat)       # 长度 3 列表
            print("elevation:", cam.elevation)
            print("azimuth:", cam.azimuth)
            print("trackbodyid:", cam.trackbodyid)
            print("fixedcamid:", cam.fixedcamid)

        prev_up    = up
        prev_down  = down
        prev_space = space

        sim.update_camera()
        _apply_pd_control(sim)
        mujoco.mj_step(sim.model, sim.data)
        sim.viewer.sync()

        # 离屏渲染两个视角并用 OpenCV 显示
        renderer.update_scene(sim.data, top_cam)
        top_img = renderer.render()
        cv2.imshow("Top View", cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))

        renderer.update_scene(sim.data, follow_cam)
        follow_img = renderer.render()
        cv2.imshow("Follow View", cv2.cvtColor(follow_img, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

        time.sleep(sim.dt * 0.5)

    kb.stop()
    renderer.close()
    cv2.destroyAllWindows()
    sim.close()

"""
real: present postion 
关节范围[0, 4096] 对应 [0, 360] 度
夹爪范围[2000, 2900] 对应 [180, 260] 度 2000是闭合状态，2900是完全张开状态

数据流: record中action字段
    action = self.bus.sync_read("Present_Position")
        - Present_Position 同 real : present postion 
        - sync_read
            - normalize: bool = True
            - _normalize
                - RANGE_M100_100(body 关节默认)-将校准范围 [range_min, range_max] 线性映射到 [-100, +100]
                    - min_ = self.calibration[motor].range_min
                    - norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                    - normalized_values[id_] = -norm if drive_mode else norm
                - RANGE_0_100(夹爪默认)-将校准范围 [range_min, range_max] 线性映射到 [0, 100]
                    - norm = ((bounded_val - min_) / (max_ - min_)) * 100
                    normalized_values[id_] = 100 - norm if drive_mode else norm
                - DEGREES(当设置为Degree=True时)
                    根据 MODEL_RESOLUTION 确定范围（目前电机仅有4096-1这个值）
                    mid = (min_ + max_) / 2
                    max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                    normalized_values[id_] = (val - mid) * 360 / max_res
        motors初始化在 koch_leader.py L49-L56
            - 夹爪为RANGE_0_100
            - 其他关节为RANGE_M100_100
        标定文件: calibration\koch_follower.json
            - drive_mode: bool
            - range_min: int
            - range_max: int
关节范围[-100, 100]
夹爪范围[0, 100] 100 是开，0是闭合

sim: 测试-跟real移动范围一致
关节范围[-3.14159, 3.14159] rad 对应 [-180, 180] 度 移动范围360度
夹爪范围[0, -1.3962634] rad 对应 [0, -80] 度 移动范围60度 sim中 0 是闭合状态
"""

def prepo2rad_homing(prepo, calibration_file="calibration/koch_follower.json"):
    # 减去 homing offset 后再转换为弧度
    # actual = prepose - homing offset
    # my_prepose = actual + MY_HOMING
    # my_homing -> rad
    if not os.path.exists(calibration_file):
        calibration_file = os.path.join(os.path.dirname(__file__), '..', calibration_file)
    with open(calibration_file, 'r') as f:
        calibration = json.load(f)
    # 读取 homing offset
    homing_offset = [calibration[joint]["homing_offset"] for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
    # print(f"func homing_offset: {homing_offset}")
    # 减去 homing offset
    actual = [prepo - offset for prepo, offset in zip(prepo, homing_offset)]
    # print(f"func actual: {actual}")
    my_prepose = [actual[i] + MODEL_HOMING[i] for i in range(len(prepo))]
    # print(f"func my_prepose: {my_prepose}")
    return prepo2rad_direct(my_prepose)

def prepo2rad_direct(action):
    """
    直接将原始位置信号（present position）转换为弧度，假设范围已经是 [0, 4096] 对应 [-3.14159, 3.14159] 度

    Parameters:
    -----------
    action: list of floats
        原始位置信号，对应各个关节和夹爪的原始值

    Returns:
    --------
    list of floats
        弧度值，关节: [-π, π]，夹爪: [-1.3962634, 0]
    """
    import numpy as np
    import math

    action = np.array(action, dtype=np.float64)
    rad = np.zeros_like(action)

    # 前5个是关节，最后一个是夹爪
    for i in range(len(action)):
        val = action[i]
        if i < len(action) - 1:
            # 关节: [0, 4096] -> [-180, 180] -> [-3.14159, 3.14159] rad
            degrees = (val / 4096.0) * 360.0 - 180.0
            rad[i] = degrees * (math.pi / 180.0)
        else:
            # 夹爪: [2000, 2900] ->[180, 260]° -> [0, -80]° -> [0, -1.3962634] rad 
            degrees = 180 + (val - 2000) / (2900 - 2000) * 80
            # 180->0; 260->-80 # 当 val = 2000 degrees = 0, 当 val = 2900 degrees = -80
            degrees = -degrees + 180
            rad[i] = degrees * (math.pi / 180.0)
    return rad.tolist()

def prepo2action(ids_values, calibration_file="calibration/koch_follower.json", apply_drive_mode=True, use_degrees=True):
    """
    将原始位置值（ids_values）转换为归一化值（normalized_values）

    Parameters:
    -----------
    ids_values: list of floats
        原始位置信号，按关节顺序排列
    calibration_file: str
        校准文件路径
    apply_drive_mode: bool
        是否应用 drive_mode 反转

    Returns:
    --------
    list of floats
        归一化值，关节: [-100, 100]，夹爪: [0, 100]
    """
    if not os.path.exists(calibration_file):
        calibration_file = os.path.join(os.path.dirname(__file__), '..', calibration_file)

    with open(calibration_file, 'r') as f:
        calibration = json.load(f)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    normalized = []

    for i, joint_name in enumerate(joint_names):
        if i >= len(ids_values):
            break
        calib = calibration[joint_name]
        min_ = calib["range_min"]
        max_ = calib["range_max"]
        
        drive_mode = apply_drive_mode and calib["drive_mode"]

        if max_ == min_:
            raise ValueError(f"Invalid calibration for motor '{joint_name}': min and max are equal.")

        bounded_val = min(max_, max(min_, ids_values[i]))

        if joint_name == "gripper":
            # RANGE_0_100: [range_min, range_max] -> [0, 100]
            norm = ((bounded_val - min_) / (max_ - min_)) * 100
            """
            bounded_val = (norm / 100) * (max_ - min_) + min_
            """
            normalized.append(100 - norm if drive_mode else norm)
        elif use_degrees:
            # Degree
            mid = (min_ + max_) / 2
            max_res = 4095 # 目前都是4095，后续如果有不同的电机需要调整
            norm = (ids_values[i] - mid) * 360 / max_res
            """
            ids_values[i] = norm * max_res / 360 + mid
            """
            normalized.append(norm)
        else: 
            # RANGE_M100_100: [range_min, range_max] -> [-100, 100]
            norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
            """
            bounded_val = (norm + 100)/200 * (max_ - min_) + min_
            """
            normalized.append(-norm if drive_mode else norm)
            
        # print(f"{joint_name=}, {min_=}, {max_=}, {ids_values[i]=}, norm={norm}, drive_mode={drive_mode}, bounded_val={bounded_val}, normalized={normalized[-1]}")

    return normalized

def action2prepo(action, calibration_file="calibration/koch_follower.json", use_degrees=True):
    """
    将归一化值（normalized_values）反转为原始位置值（ids_values）

    Parameters:
    -----------
    action: list of floats
        归一化值，关节: [-100, 100]，夹爪: [0, 100]
    calibration_file: str
        校准文件路径

    Returns:
    --------
    list of floats
        原始位置信号，按关节顺序排列
    """
    if not os.path.exists(calibration_file):
        calibration_file = os.path.join(os.path.dirname(__file__), '..', calibration_file)

    with open(calibration_file, 'r') as f:
        calibration = json.load(f)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    prepo = []

    for i, joint_name in enumerate(joint_names):
        if i >= len(action):
            break
        calib = calibration[joint_name]
        min_ = calib["range_min"]
        max_ = calib["range_max"]
        drive_mode = calib["drive_mode"]

        if joint_name == "gripper": # RANGE_0_100
            val = (100 - action[i]) if drive_mode else action[i]
            bounded_val = min(100.0, max(0.0, val))
            unnormalized_values = int((bounded_val / 100) * (max_ - min_) + min_)
        elif use_degrees: # DEGREES
            val = action[i]
            mid = (min_ + max_) / 2
            max_res = 4095
            unnormalized_values = int((val * max_res / 360) + mid)
        else: # RANGE_M100_100
            val = -action[i] if drive_mode else action[i]
            bounded_val = min(100.0, max(-100.0, val))
            unnormalized_values = int((bounded_val + 100) / 200 * (max_ - min_) + min_)

        # print(f"{joint_name=}, {min_=}, {max_=}, {action[i]=}, {unnormalized_values=}, {drive_mode=}")

        prepo.append(unnormalized_values)

    return prepo

def action2rad(action, calibration_file="calibration/koch_follower.json", use_degrees=False):
    prepo = action2prepo(action, calibration_file, use_degrees)
    rad = prepo2rad_homing(prepo, calibration_file=calibration_file)
    return rad

def replay():
    """回放函数，加载之前保存的仿真数据并回放"""
    # jsonl_file = 'sim/VLA_action.jsonl'
    calibration_file = "calibration\\czn_calibration\\follower_arm_1.json"
    jsonl_file = 'sim/VLA_part.jsonl'  # use_degrees=True 版本
    jsonl_angle_file = 'sim/VLA_part_angle.jsonl'  # 已经转换为角度的版本，直接回放角度数据
    # 加载数据 sim/VLA_action.jsonl
    trajectory = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                action = entry.get('action', [])
                rad = action2rad(action, calibration_file=calibration_file)
                
                # 保存 angle 数据 到 VLA_part_angle.json 中
                # if jsonl_angle_file is not None:
                #     with open(jsonl_angle_file, 'a') as f:
                #         json.dump({'full_action_radians': rad, 'timestamp': entry.get('timestamp', 0.0)}, f)
                #         f.write('\n')
                    
                trajectory.append({
                    'full_action_radians': rad,
                    'timestamp': entry.get('timestamp', 0.0)
                })
        print(f"成功加载 {len(trajectory)} 帧轨迹数据")
    except FileNotFoundError:
        print(f"文件未找到: {jsonl_file}")
        return

    if not trajectory:
        print("轨迹为空，退出")
        return

    # 创建仿真模型
    model_path = 'urdf/meshes/mjmodel.xml'
    sim = SimMujocoModel(model_path)
    sim.add_view('default', {'distance': 3.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -20.0, 'azimuth': 135.0})
    sim.add_view('top',     {'distance': 2.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -90.0, 'azimuth': 0.0})
    sim.add_view('follow',  {'fixed_camera': 'gripper_cam'})

    # 选择视角
    print("\n=== 选择回放视角 ===")
    print("1. default (默认透视)")
    print("2. top     (顶部俯视)")
    print("3. follow  (跟随夹爪)")
    choice = input("请输入编号 (默认1): ").strip()
    view_map = {'1': 'default', '2': 'top', '3': 'follow'}
    sim.switch_view(view_map.get(choice, 'default'))

    # 等待仿真稳定
    for _ in range(50):
        if not sim.viewer.is_running():
            sim.close()
            return
        mujoco.mj_step(sim.model, sim.data)
        sim.viewer.sync()

    # 回放数据——根据 sim/VLA_action.jsonl 中的关节角度列表设置仿真模型的关节角度
    prev_ts = trajectory[0]['timestamp']
    for i, frame in enumerate(trajectory):
        if not sim.viewer.is_running():
            break

        sim.set_joint_angles(frame['full_action_radians'])

        dt_frame = frame['timestamp'] - prev_ts
        prev_ts = frame['timestamp']
        steps = max(1, int(round(dt_frame / sim.dt)))

        for _ in range(steps):
            if not sim.viewer.is_running():
                break
            sim.update_camera()
            _apply_pd_control(sim)
            mujoco.mj_step(sim.model, sim.data)
            sim.viewer.sync()
            time.sleep(sim.dt * 0.05)

        if i % 30 == 0:
            print(f"帧 {i}/{len(trajectory)} ts={frame['timestamp']:.3f}s")

    print("轨迹回放完成，进入键盘控制模式")

    # 回放结束后进入键盘控制
    kb = KeyboardController().start()
    current_joint = 0
    prev_up = prev_down = prev_space = False

    while True:
        if not sim.viewer.is_running():
            break

        up    = kb.get('up')
        down  = kb.get('down')
        space = kb.get(' ')

        if up and not prev_up:
            current_joint = (current_joint - 1) % max(sim.model.nu, 1)
        if down and not prev_down:
            current_joint = (current_joint + 1) % max(sim.model.nu, 1)

        if sim.model.nu > 0:
            if kb.get('left'):
                sim.target_joint_positions[current_joint] = np.clip(
                    sim.target_joint_positions[current_joint] + sim.position_speed * sim.dt,
                    -3.14159, 3.14159)
            if kb.get('right'):
                sim.target_joint_positions[current_joint] = np.clip(
                    sim.target_joint_positions[current_joint] - sim.position_speed * sim.dt,
                    -3.14159, 3.14159)

        if space and not prev_space:
            mujoco.mj_resetData(sim.model, sim.data)
            sim.target_joint_positions = sim.data.qpos[:sim.model.nu].copy()

        prev_up    = up
        prev_down  = down
        prev_space = space

        sim.update_camera()
        _apply_pd_control(sim)
        mujoco.mj_step(sim.model, sim.data)
        sim.viewer.sync()
        time.sleep(sim.dt * 0.5)

    kb.stop()
    sim.close()

def show(pose_type, pose, use_degrees=False, calibration_file="calibration\\koch_follower.json", model_path='urdf/meshes/mjmodel.xml', duration=60):
    """
    显示机械臂姿态

    Parameters:
    -----------
    pose: list of floats
        姿态数据，根据pose_type不同可以是弧度、action或present position
    pose_type: str
        数据类型: 'rad'（弧度）、'action'（归一化值或角度值）、'present'（原始位置信号）
    model_path: str
        MuJoCo模型文件路径
    duration: float
        显示持续时间（秒）
    """
    # model_path='urdf/meshes/mjmodel.xml'
    # duration=60
    # action 值是走 校准后得到的，所以直接使用会有问题
    # pose_type='action'
    # use_degrees=False
    # calibration_file="calibration\\czn_calibration\\follower_arm_1.json"
    # pose = [7.985347747802734,-74.59283447265625,-76.93339538574219,-36.49921417236328,7.252747058868408,50]
    # calibration_file="calibration\\koch_follower_arm_1_fix.json"
    # pose = [3.2478632478632647, 3.2377428307122926, -70.02688172043011, -94.39393939393939, 0.26862026862026767, 6.7669172932330826]
    # pose_type='present'
    # pose = [2114, 2426, 2085, 1975, 2053, 2110]
    
    # 根据输入类型转换为弧度
    if pose_type == 'rad':
        rad_angles = pose
    elif pose_type == 'action':
        # action可以是归一化值或角度值
        rad_angles = action2rad(pose, calibration_file, use_degrees)
    elif pose_type == 'present':
        # present position原始信号
        rad_angles = prepo2rad_direct(pose)
    else:
        raise ValueError(f"不支持的pose_type: {pose_type}，可选: 'rad', 'action', 'present'")

    # print(f"input type: {pose_type}")
    # print(f"org input: {pose}")
    # print(f"rad: {rad_angles}")

    # 创建仿真器
    sim = SimMujocoModel(model_path)

    # 添加视角
    sim.add_view('default', {'distance': 3.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -20.0, 'azimuth': 135.0})
    sim.add_view('top', {'distance': 2.0, 'lookat': [0.0, 0.0, 0.0], 'elevation': -90.0, 'azimuth': 0.0})
    sim.add_view('front', {'distance': 2.5, 'lookat': [0.0, 0.0, 0.0], 'elevation': 0.0, 'azimuth': 0.0})
    sim.add_view('side', {'distance': 2.5, 'lookat': [0.0, 0.0, 0.0], 'elevation': 0.0, 'azimuth': 90.0})
    sim.switch_view('default')

    # 设置关节角度
    sim.set_joint_angles(rad_angles)

    # 应用PD控制并更新仿真
    start_time = time.time()
    # print(f"\n显示姿态中，按ESC退出...")
    print("切换视角: 1=default, 2=top, 3=front, 4=side, s=稳定模式")

    # 简单键盘控制（非阻塞式）
    try:
        from pynput import keyboard

        current_view = 'default'
        views_cycle = ['default', 'top', 'front', 'side']
        view_index = 0

        def on_press(key):
            nonlocal current_view, view_index
            try:
                if key.char == '1':
                    sim.switch_view('default')
                    current_view = 'default'
                    view_index = 0
                    print("切换到默认视角")
                elif key.char == '2':
                    sim.switch_view('top')
                    current_view = 'top'
                    view_index = 1
                    print("切换到顶部视角")
                elif key.char == '3':
                    sim.switch_view('front')
                    current_view = 'front'
                    view_index = 2
                    print("切换到正面视角")
                elif key.char == '4':
                    sim.switch_view('side')
                    current_view = 'side'
                    view_index = 3
                    print("切换到侧面视角")
                elif key.char == 's':
                    sim.toggle_stable_mode(not sim.stability_mode)
                elif key.char == 'v':
                    # 循环切换视角
                    view_index = (view_index + 1) % len(views_cycle)
                    sim.switch_view(views_cycle[view_index])
                    current_view = views_cycle[view_index]
                    print(f"切换到{current_view}视角")
                elif key == keyboard.Key.esc:
                    print("退出显示")
                    print("显示结束")
                    sim.close()
                    exit(0)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        while time.time() - start_time < duration:
            if not sim.viewer.is_running():
                break

            sim.update_camera()
            _apply_pd_control(sim)
            mujoco.mj_step(sim.model, sim.data)
            sim.viewer.sync()
            time.sleep(sim.dt)

        listener.stop()

    except ImportError:
        # 如果没有pynput，只显示固定时间
        while time.time() - start_time < duration:
            if not sim.viewer.is_running():
                break

            sim.update_camera()
            _apply_pd_control(sim)
            mujoco.mj_step(sim.model, sim.data)
            sim.viewer.sync()
            time.sleep(sim.dt)
    # while(1):
    #     time.sleep(1)
    # 按 esc 退出
    # keyboard.wait('esc')
    
    # print("显示结束")
    sim.close()

def test():
    """
        shoulder_pan: 2233, normalized: 9.059829059829056, degree: 16.307692307692307
        shoulder_lift: 2344, normalized: -11.933395004625353, degree: -5.670329670329671
        elbow_flex: 2112, normalized: -66.3978494623656, degree: -43.42857142857143
        wrist_flex: 2255, normalized: -51.96969696969697, degree: -30.153846153846153
        wrist_roll: 2045, normalized: -0.12210012210012167, degree: -0.21978021978021978
        gripper: 2144, normalized: 10.41890440386681, degree: 10.311493018259936
    """
    # print("--------------------------------")
    # calibration_file="calibration\\koch_follower.json"
    # pose = [2233, 2344, 2112, 2255, 2045, 2144]
    # pose = [2116, 2194, 1845, 2507, 1943, 2591] # czn 机械臂数据 映射到当前model上 对的
    # print("prepose:", pose)
    # print("pre2action:", prepo2action(pose, calibration_file=calibration_file, use_degrees=False))
    # print("pre2action degree:", prepo2action(pose, calibration_file=calibration_file, use_degrees=True))
    # rad = prepo2rad_direct(pose)
    # print("pre2rad:",rad)
    # show('rad', rad)
    # print("--------------------------------")
    # pose = [9.059829059829056, -11.933395004625353, -66.3978494623656, -51.96969696969697, -0.12210012210012167, 10.41890440386681]
    # print("action:", pose)
    # print("action2prepo:", action2prepo(pose, calibration_file=calibration_file, use_degrees=False))
    # rad = action2rad(pose, calibration_file=calibration_file, use_degrees=False)
    # print("action2rad:", rad)
    # show('rad', rad)
    # print("--------------------------------")
    # pose = [16.307692307692307, -5.670329670329671, -43.42857142857143, -30.153846153846153, -0.21978021978021978, 10.311493018259936]
    # print("action:", pose)
    # print("action2prepo:", action2prepo(pose, calibration_file=calibration_file, use_degrees=True))
    # rad = action2rad(pose, calibration_file=calibration_file, use_degrees=True)
    # print("action2rad:", rad)
    # show('rad', rad)
    # print("===============================")
    
    # actual + homing offset = prepose
    pose = [2034, 2114, 1696, 1550, 1864, 2384]
    homing_offset = [-55,1014,-23,37,-17,-128]
    actual = [pose[i] - homing_offset[i] for i in range(len(pose))] # actual = [2089, 1100, 1719, 1513, 1881, 2512]
    print(f"actual : {actual}")
    my_homing_offset = [27, 1094, 126, 994, 62, 79]
    my_pose = [actual[i] + my_homing_offset[i] for i in range(len(pose))] # my_pose = [2116, 2194, 1845, 2507, 1943, 2591] # 对的
    print(f"my_pose : {my_pose}")
    
    # print("===============================")
    
    # calibration_file="calibration\\czn_calibration\\follower_arm_1.json"
    # pose = [2034, 2114, 1696, 1550, 1864, 2384]
    # print("czn prepose:", pose)
    # print("czn pre2action:", prepo2action(pose, calibration_file=calibration_file, use_degrees=False))
    # print("pre2action degree:", prepo2action(pose, calibration_file=calibration_file, use_degrees=True))
    # rad = prepo2rad_homing(pose, calibration_file=calibration_file)
    # print("pre2rad:",rad)
    # show('rad', rad)
    # print("--------------------------------")
    # calibration_file="calibration\\czn_calibration\\follower_arm_1.json"
    # model_path = 'urdf\\meshes\\mjmodel.xml'
    # pose = [7.985347747802734,-74.59283447265625,-76.93339538574219,-36.49921417236328,7.252747058868408,50]
    # print("action:", pose)
    # print("action2prepo:", action2prepo(pose, calibration_file=calibration_file, use_degrees=False))
    # rad = action2rad(pose, calibration_file=calibration_file, use_degrees=False)
    # print("action2rad:", rad)
    # show('rad', rad, model_path=model_path)
    # print("--------------------------------")

if __name__ == "__main__":
    main()
    # replay()
    # test()
