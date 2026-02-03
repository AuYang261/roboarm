#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标对比工具
功能：对比正运动学计算的机械臂位姿坐标和通过手眼标定转换的位姿坐标

使用步骤：
1. 运行手眼标定获取单应性矩阵（如果还没有）
2. 运行此脚本进行坐标对比
3. 分析误差统计和可视化结果
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from arm.arm_control import Arm
from camera.camera_api import Camera
import kinpy


class CoordinateComparator:
    """坐标对比类"""

    def __init__(self,
                 hand_eye_matrix_path: str = None,
                 urdf_path: str = None):
        """
        初始化坐标对比器

        Args:
            hand_eye_matrix_path: 手眼标定矩阵文件路径
            urdf_path: URDF文件路径
        """
        # 初始化机械臂
        self.arm = Arm()
        self.arm.disable_torque()

        # 加载手眼标定矩阵
        if hand_eye_matrix_path is None:
            hand_eye_matrix_path = os.path.join(
                os.path.dirname(__file__),
                "hand-eye-data",
                "2d_homography.npy"
            )

        if os.path.exists(hand_eye_matrix_path):
            self.hand_eye_matrix = np.load(hand_eye_matrix_path)
            print(f"手眼标定矩阵已加载，形状: {self.hand_eye_matrix.shape}")
        else:
            self.hand_eye_matrix = None
            print("警告：未找到手眼标定矩阵文件")

        # 加载URDF模型
        if urdf_path is None:
            urdf_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "urdf",
                "low_cost_robot.urdf"
            )

        with open(urdf_path, 'r', encoding='utf-8') as f:
            urdf_content = f.read()

        self.chain = kinpy.build_serial_chain_from_urdf(
            urdf_content, "gripper_static_1"
        )

        # 初始化相机
        self.camera = Camera(color=True, depth=False)

        # 存储对比结果
        self.comparison_results = []

    def forward_kinematics_from_image_point(self, pixel_point: Tuple[float, float]) -> np.ndarray:
        """
        通过手眼标定将像素坐标转换为机械臂位姿坐标

        Args:
            pixel_point: 图像像素坐标 (x, y)

        Returns:
            np.ndarray: 机械臂末端在基座坐标系中的2D位置 [x, y]
        """
        if self.hand_eye_matrix is None:
            raise ValueError("手眼标定矩阵未加载")

        # 将像素坐标转换为齐次坐标 [x, y, 1]
        pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1.0])

        # 应用单应性矩阵变换：机器人坐标 = H * 像素坐标
        robot_homogeneous = self.hand_eye_matrix @ pixel_homogeneous

        # 转换为笛卡尔坐标：除以齐次坐标的第三分量
        robot_point = robot_homogeneous[:2] / robot_homogeneous[2]

        return robot_point

    def forward_kinematics_from_joint_angles(self, joint_angles_deg: List[float]) -> Dict:
        """
        通过正运动学从关节角度计算机械臂末端位姿

        Args:
            joint_angles_deg: 关节角度（度）

        Returns:
            Dict: 包含位置和旋转的字典
        """
        # 角度转换为弧度（kinpy使用弧度）
        joint_angles_rad = np.deg2rad(joint_angles_deg).tolist()

        # 计算正运动学：关节角度 -> 末端位姿
        fk_result = self.chain.forward_kinematics(joint_angles_rad, end_only=True)

        return {
            'position': np.array(fk_result.pos),  # 末端位置 [x, y, z] (米)
            'rotation_euler': np.array(fk_result.rot_euler),  # 末端欧拉角 [rx, ry, rz] (弧度)
            'rotation_matrix': np.array(fk_result.rot_mat)  # 末端旋转矩阵 3x3
        }

    def collect_comparison_data(self, num_points: int = 10) -> List[Dict]:
        """
        采集对比数据

        Args:
            num_points: 需要采集的点数

        Returns:
            List[Dict]: 对比数据列表
        """
        print(f"开始采集 {num_points} 个对比点...")
        print("操作说明：")
        print("1. 在图像窗口中点击选择点")
        print("2. 按空格键记录当前机械臂位姿")
        print("3. 按ESC键结束采集")

        points_collected = 0
        comparison_data = []

        # 创建图像窗口
        window_name = "Coordinate Comparison - Click Points"
        cv2.namedWindow(window_name)

        # 鼠标回调函数
        clicked_point = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_point
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_point = (x, y)
                print(f"点击位置: ({x}, {y})")
                joint_angles, gripper_state = self.arm.get_arm_angles()
                # 通过正运动学计算位姿
                fk_result = self.forward_kinematics_from_joint_angles(joint_angles)
                print(f"  正运动学位姿: {fk_result['position']}")
                print(f"  夹爪角度: {gripper_state}")
            

        cv2.setMouseCallback(window_name, mouse_callback)

        while points_collected < num_points:
            try:
                # 获取当前图像
                frames = self.camera.get_frames()
                color_image = frames.get("color")
                if color_image is None:
                    print("获取图像失败")
                    continue

                # 显示图像
                display_image = color_image.copy()

                # 如果有点击，显示点
                if clicked_point is not None:
                    cv2.circle(display_image, clicked_point, 5, (0, 255, 0), -1)
                    cv2.putText(display_image, f"Point {points_collected + 1}",
                               (clicked_point[0] + 10, clicked_point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow(window_name, display_image)

                key = cv2.waitKey(1)

                if key == 27:  # ESC键
                    print("采集结束")
                    break
                elif key == ord(' ') and clicked_point is not None:  # 空格键记录
                    # 获取当前机械臂关节角度
                    joint_angles, gripper_state = self.arm.get_arm_angles()
                    if joint_angles is None:
                        print("获取机械臂关节角度失败")
                        continue

                    # 通过手眼标定计算位姿
                    hand_eye_position = self.forward_kinematics_from_image_point(clicked_point)

                    # 通过正运动学计算位姿
                    fk_result = self.forward_kinematics_from_joint_angles(joint_angles)

                    # 存储对比数据
                    comparison_point = {
                        'point_id': points_collected + 1,
                        'pixel_coordinates': clicked_point,
                        'joint_angles_deg': joint_angles,
                        'hand_eye_position_2d': hand_eye_position.tolist(),  # [x, y]
                        'forward_kinematics_position': fk_result['position'].tolist(),  # [x, y, z]
                        'forward_kinematics_rotation': fk_result['rotation_euler'].tolist(),  # [rx, ry, rz]
                        'timestamp': datetime.now().isoformat()
                    }

                    comparison_data.append(comparison_point)
                    points_collected += 1

                    print(f"已采集点 {points_collected}/{num_points}")
                    print(f"  像素坐标: {clicked_point}")
                    print(f"  手眼标定位姿: {hand_eye_position}")
                    print(f"  正运动学位姿: {fk_result['position']}")
                    print(f"  夹爪角度: {gripper_state}")

                    # 重置点击点
                    clicked_point = None

            except KeyboardInterrupt:
                break

        cv2.destroyAllWindows()
        return comparison_data

    def calculate_errors(self, comparison_data: List[Dict]) -> Dict:
        """
        计算坐标对比误差

        Args:
            comparison_data: 对比数据列表

        Returns:
            Dict: 误差统计
        """
        if not comparison_data:
            return {}

        errors_2d = []  # XY平面误差（欧氏距离）
        errors_z = []   # Z方向误差（与默认桌面高度比较）
        positions_hand_eye = []  # 手眼标定位置
        positions_fk = []        # 正运动学位置

        for data in comparison_data:
            # 手眼标定的2D位置（从像素坐标转换而来）
            hand_eye_pos = np.array(data['hand_eye_position_2d'])

            # 正运动学的3D位置（取XY平面进行比较）
            fk_pos = np.array(data['forward_kinematics_position'])
            fk_pos_2d = fk_pos[:2]  # 只取XY坐标

            # 计算2D误差：手眼标定位置与正运动学位置的欧氏距离
            error_2d = np.linalg.norm(hand_eye_pos - fk_pos_2d)
            errors_2d.append(error_2d)

            # Z方向误差：手眼标定假设Z固定（桌面高度），与正运动学计算的Z值比较
            default_z = 0.075  # 默认桌面高度（米），可从配置文件读取
            error_z = abs(fk_pos[2] - default_z)
            errors_z.append(error_z)

            positions_hand_eye.append(hand_eye_pos)
            positions_fk.append(fk_pos_2d)

        # 转换为numpy数组
        errors_2d = np.array(errors_2d)
        errors_z = np.array(errors_z)
        positions_hand_eye = np.array(positions_hand_eye)
        positions_fk = np.array(positions_fk)

        # 计算统计信息
        error_stats = {
            'num_points': len(comparison_data),
            'errors_2d': {
                'mean': float(np.mean(errors_2d)),
                'std': float(np.std(errors_2d)),
                'max': float(np.max(errors_2d)),
                'min': float(np.min(errors_2d)),
                'median': float(np.median(errors_2d)),
                'all_values': errors_2d.tolist()
            },
            'errors_z': {
                'mean': float(np.mean(errors_z)),
                'std': float(np.std(errors_z)),
                'max': float(np.max(errors_z)),
                'min': float(np.min(errors_z)),
                'median': float(np.median(errors_z)),
                'all_values': errors_z.tolist()
            },
            'positions': {
                'hand_eye': positions_hand_eye.tolist(),
                'forward_kinematics': positions_fk.tolist()
            }
        }

        return error_stats

    def visualize_comparison(self, comparison_data: List[Dict], error_stats: Dict):
        """
        可视化对比结果

        Args:
            comparison_data: 对比数据列表
            error_stats: 误差统计
        """
        if not comparison_data:
            print("没有数据可可视化")
            return

        # 从误差统计中提取数据
        positions_hand_eye = np.array(error_stats['positions']['hand_eye'])  # 手眼标定位置
        positions_fk = np.array(error_stats['positions']['forward_kinematics'])  # 正运动学位置
        errors_2d = np.array(error_stats['errors_2d']['all_values'])  # 2D误差值

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. XY平面位置对比散点图
        ax1 = axes[0, 0]
        ax1.scatter(positions_hand_eye[:, 0], positions_hand_eye[:, 1],
                   c='blue', label='Hand-Eye Calibration', alpha=0.6, s=50)
        ax1.scatter(positions_fk[:, 0], positions_fk[:, 1],
                   c='red', label='Forward Kinematics', alpha=0.6, s=50, marker='x')

        # 连接对应的点
        for i in range(len(positions_hand_eye)):
            ax1.plot([positions_hand_eye[i, 0], positions_fk[i, 0]],
                    [positions_hand_eye[i, 1], positions_fk[i, 1]],
                    'gray', alpha=0.3, linewidth=1)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('XY Plane Position Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 2. 2D误差直方图
        ax2 = axes[0, 1]
        ax2.hist(errors_2d * 1000, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(errors_2d) * 1000, color='red', linestyle='--',
                   label=f'Mean: {np.mean(errors_2d)*1000:.2f} mm')
        ax2.set_xlabel('2D Error (mm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('2D Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 误差与位置关系图
        ax3 = axes[1, 0]
        scatter = ax3.scatter(positions_hand_eye[:, 0], positions_hand_eye[:, 1],
                            c=errors_2d * 1000, cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Error Distribution in XY Plane')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Error (mm)')

        # 4. 误差统计文本
        ax4 = axes[1, 1]
        ax4.axis('off')

        stats_text = f"""Error Statistics (Total {len(comparison_data)} points):

2D Plane Error:
  Mean: {error_stats['errors_2d']['mean']*1000:.2f} mm
  Std Dev: {error_stats['errors_2d']['std']*1000:.2f} mm
  Max: {error_stats['errors_2d']['max']*1000:.2f} mm
  Min: {error_stats['errors_2d']['min']*1000:.2f} mm
  Median: {error_stats['errors_2d']['median']*1000:.2f} mm

Z Direction Error:
  Mean: {error_stats['errors_z']['mean']*1000:.2f} mm
  Std Dev: {error_stats['errors_z']['std']*1000:.2f} mm
  Max: {error_stats['errors_z']['max']*1000:.2f} mm
  Min: {error_stats['errors_z']['min']*1000:.2f} mm

Calibration Accuracy:
  {'Excellent' if error_stats['errors_2d']['mean']*1000 < 5 else 'Good' if error_stats['errors_2d']['mean']*1000 < 10 else 'Fair' if error_stats['errors_2d']['mean']*1000 < 20 else 'Needs Improvement'}
"""

        ax4.text(0.1, 0.95, stats_text, fontsize=10,
                verticalalignment='top', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("comparison_results")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"对比结果已保存至: {output_path}")

    def save_results(self, comparison_data: List[Dict], error_stats: Dict):
        """
        保存对比结果到文件

        Args:
            comparison_data: 对比数据列表
            error_stats: 误差统计
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("comparison_results")
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始数据
        data_file = os.path.join(output_dir, f"comparison_data_{timestamp}.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'comparison_data': comparison_data,
                'error_stats': error_stats
            }, f, indent=2, ensure_ascii=False)

        # 保存统计摘要
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Coordinate Comparison Results Summary - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total points: {len(comparison_data)}\n\n")

            f.write("2D Plane Error Statistics:\n")
            f.write(f"  Mean: {error_stats['errors_2d']['mean']*1000:.2f} mm\n")
            f.write(f"  Std Dev: {error_stats['errors_2d']['std']*1000:.2f} mm\n")
            f.write(f"  Max: {error_stats['errors_2d']['max']*1000:.2f} mm\n")
            f.write(f"  Min: {error_stats['errors_2d']['min']*1000:.2f} mm\n")
            f.write(f"  Median: {error_stats['errors_2d']['median']*1000:.2f} mm\n\n")

            f.write("Detailed Point Data:\n")
            for i, data in enumerate(comparison_data):
                f.write(f"\nPoint {i+1}:\n")
                f.write(f"  Pixel coordinates: {data['pixel_coordinates']}\n")
                f.write(f"  Joint angles: {data['joint_angles_deg']}\n")
                f.write(f"  Hand-eye position: {data['hand_eye_position_2d']}\n")
                f.write(f"  Forward kinematics position: {data['forward_kinematics_position']}\n")
                f.write(f"  2D error: {error_stats['errors_2d']['all_values'][i]*1000:.2f} mm\n")

        print(f"结果已保存至目录: {output_dir}")
        print(f"  原始数据: {os.path.basename(data_file)}")
        print(f"  统计摘要: {os.path.basename(summary_file)}")

    def run_comparison(self, num_points: int = 10, visualize: bool = True):
        """
        运行完整的坐标对比流程

        Args:
            num_points: 采集点数
            visualize: 是否可视化结果
        """
        print("=" * 60)
        print("坐标对比工具 - 开始运行")
        print("=" * 60)

        # 1. 采集数据
        print("\n1. 数据采集阶段")
        comparison_data = self.collect_comparison_data(num_points)

        if not comparison_data:
            print("未采集到数据，程序退出")
            return

        # 2. 计算误差
        print("\n2. 误差计算阶段")
        error_stats = self.calculate_errors(comparison_data)

        # 3. 显示结果
        print("\n3. 结果显示阶段")
        print(f"\n采集到 {len(comparison_data)} 个点的数据")
        print(f"2D平面误差均值: {error_stats['errors_2d']['mean']*1000:.2f} mm")
        print(f"2D平面误差标准差: {error_stats['errors_2d']['std']*1000:.2f} mm")

        # 4. 可视化
        if visualize:
            print("\n4. 可视化阶段")
            self.visualize_comparison(comparison_data, error_stats)

        # 5. 保存结果
        print("\n5. 结果保存阶段")
        self.save_results(comparison_data, error_stats)

        print("\n" + "=" * 60)
        print("坐标对比完成")
        print("=" * 60)

        # 清理资源
        self.camera.close()
        self.arm.disconnect_arm()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Coordinate Comparison Tool')
    parser.add_argument('--num-points', type=int, default=10,
                       help='Number of points to collect (default: 10)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Do not show visualization charts')
    parser.add_argument('--hand-eye-matrix', type=str,
                       help='Path to hand-eye calibration matrix file')
    parser.add_argument('--urdf-path', type=str,
                       help='Path to URDF file')

    args = parser.parse_args()

    try:
        # 创建坐标对比器
        comparator = CoordinateComparator(
            hand_eye_matrix_path=args.hand_eye_matrix,
            urdf_path=args.urdf_path
        )

        # 运行对比
        comparator.run_comparison(
            num_points=args.num_points,
            visualize=not args.no_visualize
        )

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()