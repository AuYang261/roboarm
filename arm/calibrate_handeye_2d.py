# Description: 机械臂手眼标定2D版
# 使用方法：准备一个显眼的点（如一个小球），用鼠标点击图片上该点的位置
# 再将机械臂末端移动到该位置（要求夹爪static连杆垂直于桌面，即保证夹爪根部和末端xy坐标相同），按空格键记录机械臂末端位姿
# 改变点的位置，重复4次以上，越多误差越小，按ESC键退出计算标定结果
# 标定完成后会得到一个矩阵，表示相机坐标系（二维）到机械臂基座坐标系（z轴为桌面不变，故也是二维）的变换
from collections.abc import Sequence
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.orb_camera import open_camera, get_frames, close_camera
from arm.arm_control import Arm


import argparse
import cv2
import numpy as np
import kinpy
import time
import threading

def read_urdf(urdf_content: str) -> kinpy.chain.SerialChain:
    chain = kinpy.build_serial_chain_from_urdf(urdf_content, "gripper_static_1")
    return chain


def forward_kinematics(
    chain: kinpy.chain.SerialChain, joint_angles_rad: Sequence[float | int]
) -> kinpy.Transform:
    # 使用kinpy计算正运动学，欧拉角单位为弧度
    return chain.forward_kinematics(joint_angles_rad, end_only=True)  # type: ignore


POINTS = []


# 收集图片和机械臂末端坐标数据
def collect_image_pose(image_points_path, angles_deg_list_path):
    # 定义鼠标事件回调函数
    def mouse_callback(event, x, y, flags, param):
        global POINTS
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            print(f"Left button clicked at ({x}, {y})")
            POINTS.append((x, y))

    # 创建窗口并绑定鼠标回调函数
    window_name = "Camera"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    angles_deg_list = []
    arm = Arm()
    arm.disable_torque()
    cam = open_camera(color=True, depth=False)
    while True:
        try:
            angles_deg, gripper = arm.get_read_arm_angles()
            if angles_deg is None:
                print("获取机械臂角度失败")
                continue

            frames = get_frames(cam)
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                continue
            cv2.imshow(window_name, color_image)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord(" "):
                angles_deg_list.append(angles_deg)
                print("机械臂角度:", angles_deg, "夹爪状态:", gripper)
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()
    close_camera(cam)
    arm.disconnect_arm()

    np.save(image_points_path, np.array(POINTS))
    np.save(angles_deg_list_path, np.array(angles_deg_list))

    return image_points_path, angles_deg_list_path


def calibrate_2d(
    chain: kinpy.chain.SerialChain, image_points_path, angles_deg_list_path
):
    image_points = np.load(image_points_path).astype(np.float32)  # Nx2
    angles_deg_list = np.load(angles_deg_list_path).astype(np.float32)  # Nx6

    assert (
        image_points.shape[0] == angles_deg_list.shape[0]
    ), "图片点和机械臂位姿数量不匹配"

    assert image_points.shape[0] >= 4, "标定点数量不足，至少需要4个点"

    poses_list: list[kinpy.Transform] = []
    for angles_deg in angles_deg_list:
        end_pose = forward_kinematics(
            chain=chain, joint_angles_rad=np.deg2rad(angles_deg).tolist()
        )
        poses_list.append(end_pose)
    poses = np.array(
        [
            [
                p.pos[0],
                p.pos[1],
                p.pos[2],
                p.rot_euler[0],
                p.rot_euler[1],
                p.rot_euler[2],
            ]
            for p in poses_list
        ],
        dtype=np.float32,
    )  # Nx6
    print("机械臂末端位姿:")
    print(poses)
    # 计算单应性矩阵 M
    # M 就是你的 "像素->机器人" 转换器
    M, mask = cv2.findHomography(image_points, poses[:, :2], cv2.RANSAC, 5.0)
    return M


def test_homography(chain: kinpy.chain.SerialChain, M, image_point, z=0.07):

    arm = Arm()
    arm.set_arm_angles([0, 0, 0, 0, 0], gripper_angle_deg=None)
    time.sleep(2)

    """测试单应性矩阵"""
    z = 0.1
    image_point_homogeneous = np.array([image_point[0], image_point[1], 1.0])
    robot_point_homogeneous = M @ image_point_homogeneous
    robot_point = robot_point_homogeneous[:2] / robot_point_homogeneous[2]
    angles_deg = np.rad2deg(
        chain.inverse_kinematics(
            kinpy.Transform(
                pos=np.array([robot_point[0], robot_point[1], z]), rot=[0, 0, 0]
            )
        )
    )
    print(
        f"测试点 {image_point} 对应机械臂末端位置 {robot_point}, 逆解关节角度 {angles_deg}"
    )
    arm.set_arm_angles(angles_deg.tolist(), gripper_angle_deg=80)
    time.sleep(2)

    # 下降
    z = 0.05
    angles_deg = np.rad2deg(
        chain.inverse_kinematics(
            kinpy.Transform(
                pos=np.array([robot_point[0], robot_point[1], z]), rot=[0, 0, 0]
            )
        )
    )
    print(
        f"测试点 {image_point} 对应机械臂末端位置 {robot_point}, 逆解关节角度 {angles_deg}"
    )
    arm.set_arm_angles(angles_deg.tolist(), gripper_angle_deg=80)
    time.sleep(2)

    # 归0
    arm.set_arm_angles([0, 0, 0, 0, 0], gripper_angle_deg=None)
    time.sleep(2)
    arm.disconnect_arm()


def main():
    argparser = argparse.ArgumentParser(description="机械臂手眼标定2D版")
    argparser.add_argument("--mode", type=str, default="calibrate", help="模式")
    args = argparser.parse_args()
    
    
    image_points_path = os.path.join(
        os.path.dirname(__file__), "hand-eye-data", "2d_image_points.npy"
    )
    angles_deg_list_path = os.path.join(
        os.path.dirname(__file__), "hand-eye-data", "2d_angles_deg_list.npy"
    )
    if not os.path.exists(os.path.dirname(image_points_path)):
        os.makedirs(os.path.dirname(image_points_path))
    if not os.path.exists(os.path.dirname(angles_deg_list_path)):
        os.makedirs(os.path.dirname(angles_deg_list_path))
    homography_matrix_path = os.path.join(
        os.path.dirname(__file__), "hand-eye-data", "2d_homography.npy"
    )

    chain = read_urdf(
        open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "urdf",
                "low_cost_robot.urdf",
            )
        ).read()
    )

    if args.mode == "calibrate":
        # 采集数据
        collect_image_pose(image_points_path, angles_deg_list_path)
        # 计算单应性矩阵
        homography_matrix = calibrate_2d(chain, image_points_path, angles_deg_list_path)
        np.save(
            homography_matrix_path,
            homography_matrix,
        )

        angles_deg_list = np.load(angles_deg_list_path).astype(np.float32)  # Nx6
        print("机械臂角度:")
        print(angles_deg_list)
        points = np.load(image_points_path).astype(np.float32)  # Nx2
        print("图片上点的像素坐标:")
        print(points)
        homography_matrix = np.load(homography_matrix_path)
        print("计算得到的单应性矩阵:")
        print(homography_matrix)
        for point in points:
            test_homography(chain, homography_matrix, point)
    elif args.mode == "test":
        homography_matrix = np.load(homography_matrix_path)
        test_handeye_2d(chain, homography_matrix)


def test_handeye_2d(chain: kinpy.chain.SerialChain, homography_matrix):
    # 回调函数：获取point并移动
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            print(f"Left button clicked at ({x}, {y})")
            # 创建一个线程去执行移动函数
            threading.Thread(target=test_homography, args=(chain, homography_matrix, (x, y))).start()

    # 获取2d坐标 创建窗口并绑定鼠标回调函数
    window_name = "Camera"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 开启相机
    cam = open_camera(color=True, depth=False)
    while True:
        try:
            frames = get_frames(cam)
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                time.sleep(0.5)
                continue
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1)
            # esc退出
            if key == 27:
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    close_camera(cam)
    arm = Arm()
    arm.disable_torque()
    arm.disconnect_arm()


if __name__ == "__main__":
    main()
