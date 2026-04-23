# Description: 机械臂手眼标定2D版
# 使用方法：准备一个显眼的点（如一个小球），用鼠标点击图片上该点的位置
# 再将机械臂末端移动到该位置（要求夹爪static连杆垂直于桌面，即保证夹爪根部和末端xy坐标相同），按空格键记录图片点与末端位姿
# 改变点的位置，重复4次以上，越多误差越小，按ESC键退出计算标定结果
# 标定完成后会得到一个矩阵，表示相机坐标系（二维）到机械臂基座坐标系（z轴为桌面不变，故也是二维）的变换
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from config_getter import get_config_value
from arm.arm_base import Arm
import argparse
import cv2
import numpy as np
import time
import threading


def pack_end_pose(
    end_pos: list[float] | None, end_rot_deg_zyx: list[float] | None
) -> np.ndarray | None:
    if end_pos is None or end_rot_deg_zyx is None:
        return None
    return np.array([*end_pos, *end_rot_deg_zyx], dtype=np.float32)


def image_to_robot_xy(
    homography_matrix: np.ndarray, image_point: tuple[float, float] | np.ndarray
) -> np.ndarray:
    image_point_homogeneous = np.array(
        [image_point[0], image_point[1], 1.0], dtype=np.float64
    )
    robot_point_homogeneous = homography_matrix @ image_point_homogeneous
    return robot_point_homogeneous[:2] / robot_point_homogeneous[2]


# 收集图片和机械臂末端坐标数据
def collect_image_pose(image_points_path, end_poses_path):
    image_points: list[tuple[int, int]] = []
    end_poses: list[np.ndarray] = []
    selected_point: tuple[int, int] | None = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            print(f"选择图片点: ({x}, {y})，将机械臂移动到该点后按空格记录。")

    window_name = "Camera"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    arm = Arm()
    arm.disable_torque()
    cam = Camera(color=True, depth=False)
    while True:
        try:
            frames = cam.get_frames()
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                continue

            image_to_show = color_image.copy()
            if selected_point is not None:
                cv2.circle(image_to_show, selected_point, 5, (0, 0, 255), -1)
            cv2.putText(
                image_to_show,
                f"pairs: {len(image_points)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, image_to_show)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == ord(" "):
                if selected_point is None:
                    print("请先点击图片点，再按空格记录。")
                    continue

                end_pos, end_rot_deg_zyx = arm.get_arm_pose()
                end_pose = pack_end_pose(end_pos, end_rot_deg_zyx)
                if end_pose is None:
                    print("获取机械臂末端位姿失败")
                    continue

                image_points.append(selected_point)
                end_poses.append(end_pose)
                print("记录图片点:", selected_point)
                print(
                    "记录末端位姿 [x, y, z, RZ, RY, RX] (欧拉角顺序为ZYX):",
                    end_pose.tolist(),
                )
                selected_point = None
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()
    cam.close()
    arm.disconnect_arm()

    np.save(image_points_path, np.array(image_points, dtype=np.float32))
    np.save(end_poses_path, np.array(end_poses, dtype=np.float32))

    return image_points_path, end_poses_path


def calibrate_2d(image_points_path, end_poses_path):
    image_points = np.load(image_points_path).astype(np.float32)
    end_poses = np.load(end_poses_path).astype(np.float32)

    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("图片点数据格式错误，应该是Nx2")
    if end_poses.ndim != 2 or end_poses.shape[1] < 2:
        raise ValueError("机械臂末端位姿数据格式错误，应该至少包含x和y坐标")

    assert image_points.shape[0] == end_poses.shape[0], "图片点和机械臂位姿数量不匹配"
    assert image_points.shape[0] >= 4, "标定点数量不足，至少需要4个点"

    print("机械臂末端位姿 [x, y, z, RZ, RY, RX] (欧拉角顺序为ZYX):")
    print(end_poses)

    robot_points = end_poses[:, :2]
    homography_matrix, _ = cv2.findHomography(
        image_points, robot_points, cv2.RANSAC, 5.0
    )
    if homography_matrix is None:
        raise RuntimeError("计算单应性矩阵失败")
    return homography_matrix


def test_homography(homography_matrix, image_point):
    arm = Arm()
    arm.move_to_home()
    time.sleep(1)

    robot_point = image_to_robot_xy(homography_matrix, image_point)
    z = get_config_value("default_desktop_height", 0.075)
    print(f"测试点 {image_point} 对应机械臂末端位置 {robot_point.tolist()}")
    arm.move_to(
        [float(robot_point[0]), float(robot_point[1]), z],
        gripper_open_0to1=1,
        rot_rad=0,
    )
    time.sleep(1)

    arm.move_to_home()
    time.sleep(1)
    arm.disconnect_arm()


def test_moveto(arm: Arm, homography_matrix, image_point, move_lock: threading.Lock):
    with move_lock:
        arm.move_to_home()
        time.sleep(1)

        target_x, target_y = image_to_robot_xy(homography_matrix, image_point)
        target_z = get_config_value("default_desktop_height", 0.075)
        print(
            f"Clicked image point: ({image_point[0]}, {image_point[1]}), "
            f"Mapped arm position: ({target_x}, {target_y})"
        )
        arm.move_to(
            [float(target_x), float(target_y), target_z],
            gripper_open_0to1=1,
            rot_rad=0,
        )


def test_moveto_double_arm(homography_matrix, image_point):
    pass


def test_handeye_2d(homography_matrix):
    arm = Arm()
    move_lock = threading.Lock()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left button clicked at ({x}, {y})")
            threading.Thread(
                target=test_moveto,
                args=(arm, homography_matrix, (x, y), move_lock),
                daemon=True,
            ).start()

    window_name = "Camera"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    cam = Camera(color=True, depth=False)
    while True:
        try:
            frames = cam.get_frames()
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                time.sleep(0.5)
                continue
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1)
            if key == 27:
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    cam.close()

    arm.disconnect_arm()


def main():
    argparser = argparse.ArgumentParser(description="机械臂手眼标定2D版")
    argparser.add_argument("--mode", type=str, default="test", help="模式")
    args = argparser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "hand-eye-data")
    image_points_path = os.path.join(data_dir, "2d_image_points.npy")
    end_poses_path = os.path.join(data_dir, "2d_end_poses.npy")
    homography_matrix_path = os.path.join(data_dir, "2d_homography.npy")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.mode == "calibrate":
        collect_image_pose(image_points_path, end_poses_path)
        homography_matrix = calibrate_2d(image_points_path, end_poses_path)
        np.save(homography_matrix_path, homography_matrix)

        end_poses = np.load(end_poses_path).astype(np.float32)
        print("机械臂末端位姿 [x, y, z, RZ, RY, RX] (欧拉角顺序为ZYX):")
        print(end_poses)
        points = np.load(image_points_path).astype(np.float32)
        print("图片上点的像素坐标:")
        print(points)
        homography_matrix = np.load(homography_matrix_path)
        print("计算得到的单应性矩阵:")
        print(homography_matrix)
        for point in points:
            test_homography(homography_matrix, point)
    elif args.mode == "test":
        homography_matrix = np.load(homography_matrix_path)
        test_handeye_2d(homography_matrix)


if __name__ == "__main__":
    main()
