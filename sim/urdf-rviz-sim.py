# urdf-viz urdf/arm.urdf -p 7777
# https://github.com/openrr/urdf-viz/releases
# "urdf/low_cost_robot.urdf" is required to run this example.


import kinpy as kp
import numpy as np
import time
import json
import urllib.request


class KinpyURDFVizController:
    def __init__(self, urdf_path, http_base="http://127.0.0.1:7777"):
        self.urdf_path = urdf_path
        self.http_base = http_base
        self.chain = None
        
    def load_robot(self):
        """加载URDF并构建运动学链"""
        with open(self.urdf_path, 'r') as f:
            urdf_content = f.read()
        self.chain = kp.build_chain_from_urdf(urdf_content)
        
        # 获取关节名称（兼容不同版本的kinpy）
        print("self.chain: ")
        print(self.chain)

        if hasattr(self.chain, "joint_names"):
            # 新版本直接暴露joint_names属性
            self.joint_names = list(self.chain.joint_names)
        elif hasattr(self.chain, "get_joint_parameter_names"):
            # 旧版本使用方法获取
            self.joint_names = list(self.chain.get_joint_parameter_names())
        else:
            self.joint_names = []
            print("警告: 无法获取关节名称，后续功能可能受限")

        print(f"可用关节: {self.joint_names}")
        
    # ---------------- HTTP 接口 -----------------
    def http_get(self, path):
        url = self.http_base + path
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                return json.loads(data)
            return data.decode("utf-8", errors="ignore")

    def http_post_json(self, path, payload):
        url = self.http_base + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                return json.loads(data)
            return data.decode("utf-8", errors="ignore")

    def get_joint_positions_http(self):
        return self.http_get("/get_joint_positions")

    def set_joint_positions_http(self, angles_dict):
        # 将 dict 转为 README 所需的 names/positions 数组
        names = list(angles_dict.keys())
        positions = [angles_dict[n] for n in names]
        return self.http_post_json("/set_joint_positions", {"names": names, "positions": positions})

    def get_robot_origin_http(self):
        return self.http_get("/get_robot_origin")

    def set_robot_origin_http(self, position, quaternion):
        return self.http_post_json("/set_robot_origin", {"position": position, "quaternion": quaternion})

    def get_urdf_text_http(self):
        return self.http_get("/get_urdf_text")

    def ensure_joint_names_from_http(self):
        try:
            data = self.get_joint_positions_http()
            if isinstance(data, dict) and "names" in data:
                self.joint_names = list(data["names"])
                print(f"HTTP端获取的关节: {self.joint_names}")
        except Exception as e:
            print(f"通过HTTP获取关节名称失败: {e}")
        
    def move_to_joint_angles(self, angles_dict):
        """通过 HTTP 更新关节角度"""
        try:
            self.set_joint_positions_http(angles_dict)
        except Exception as e:
            print(f"HTTP 设置关节失败: {e}")
    
    def calculate_forward_kinematics(self, angles_dict):
        """计算正向运动学"""
        if self.chain:
            # 计算末端执行器位姿
            transforms = self.chain.forward_kinematics(angles_dict)
            end_effector = transforms[self.chain.links[-1].name]
            return end_effector
        return None
    
    def calculate_inverse_kinematics(self, target_position, initial_angles=None):
        """计算逆向运动学（如果kinpy支持）"""
        if self.chain and hasattr(self.chain, 'inverse_kinematics'):
            if initial_angles is None:
                initial_angles = {name: 0.0 for name in self.joint_names}
            
            result = self.chain.inverse_kinematics(
                target_position, 
                initial_angles
            )
            return result
        else:
            print("此版本的kinpy可能不支持逆运动学")
            return None
    
    def create_circular_trajectory(self, center, radius, steps=50):
        """生成针对前3个关节的小幅轨迹（示例）"""
        names = (self.joint_names or [])[:3]
        trajectory = []
        for i in range(steps):
            t = 2 * np.pi * i / steps
            angles = {}
            if len(names) >= 1:
                angles[names[0]] = np.sin(t) * 0.5
            if len(names) >= 2:
                angles[names[1]] = np.cos(t) * 0.3
            if len(names) >= 3:
                angles[names[2]] = np.sin(t * 0.5) * 0.2
            trajectory.append(angles)
        return trajectory

def main():
    # 1. 创建控制器（HTTP 端口与 urdf-viz --web-server-port 保持一致）
    controller = KinpyURDFVizController(
        "./urdf/arm.urdf",
        http_base="http://127.0.0.1:7777",
    )

    # 2. 加载机器人模型
    controller.load_robot()

    # 3. 通过 HTTP 同步 urdf-viz 实际关节名
    controller.ensure_joint_names_from_http()

    # 4. 获取并打印当前关节角度（HTTP）
    try:
        jp = controller.get_joint_positions_http()
        print("当前关节角度:")
        if isinstance(jp, dict):
            names = jp.get("names", [])
            positions = jp.get("positions", [])
            print(list(zip(names, positions)))
    except Exception as e:
        print(f"获取关节角度失败: {e}")

    # 5. 构造一个针对前3个关节的小测试角度并移动
    test_angles = {}
    for idx, name in enumerate((controller.joint_names or [])[:3]):
        test_angles[name] = [0.5, -0.2, 0.3][idx]
    print(f"移动到测试角度: {test_angles}")
    controller.move_to_joint_angles(test_angles)
    time.sleep(1)

    # 6. 生成并播放轨迹（HTTP）
    print("播放轨迹...")
    trajectory = controller.create_circular_trajectory([0, 0, 0.5], 0.2, 30)
    for angles in trajectory:
        controller.move_to_joint_angles(angles)
        time.sleep(0.1)

    # 7. 结束

if __name__ == "__main__":
    main()
    
    