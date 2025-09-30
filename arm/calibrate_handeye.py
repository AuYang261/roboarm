from collections.abc import Sequence
import sys
import os

from matplotlib.transforms import Transform

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.orb_camera import open_camera, get_frames, close_camera
from arm.arm_control import Arm
import cv2
import xlrd2, xlwt
from math import cos, sin, pi
import numpy as np
import kinpy
from pynput import keyboard
import time
import threading


# 这个类从https://blog.csdn.net/yangbisheng1121/article/details/128643477复制过来的
class Calibration:
    def __init__(self):
        # 相机内参
        self.K = np.array(
            [
                [2.54565632e03, 0.00000000e00, 9.68119560e02],
                [0.00000000e00, 2.54565632e03, 5.31897821e02],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            dtype=np.float64,
        )
        # 相机畸变参数
        self.distortion = np.array([[-0.2557898, 0.81056366, 0.0, 0.0, -8.39153683]])
        # 标定板格数
        self.target_x_number = 8
        self.target_y_number = 8
        # 标定板格子大小，单位mm
        self.target_cell_size = 40

    # 欧拉角转旋转矩阵
    @staticmethod
    def angle2rotation(x, y, z):
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    # 末端位姿的旋转和平移矩阵，角度单位为度，位移单位为mm
    @staticmethod
    def gripper2base(x, y, z, tx, ty, tz):
        thetaX = x / 180 * pi
        thetaY = y / 180 * pi
        thetaZ = z / 180 * pi
        R_gripper2base = Calibration.angle2rotation(thetaX, thetaY, thetaZ)
        T_gripper2base = np.array([[tx], [ty], [tz]])
        Matrix_gripper2base = np.column_stack([R_gripper2base, T_gripper2base])
        Matrix_gripper2base = np.row_stack(
            (Matrix_gripper2base, np.array([0, 0, 0, 1]))
        )
        R_gripper2base = Matrix_gripper2base[:3, :3]
        T_gripper2base = Matrix_gripper2base[:3, 3].reshape((3, 1))
        return R_gripper2base, T_gripper2base

    def target2camera(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (self.target_x_number, self.target_y_number), None
        )
        corner_points = np.zeros((2, corners.shape[0]), dtype=np.float64)
        for i in range(corners.shape[0]):
            corner_points[:, i] = corners[i, 0, :]
        object_points = np.zeros(
            (3, self.target_x_number * self.target_y_number), dtype=np.float64
        )
        count = 0
        for i in range(self.target_y_number):
            for j in range(self.target_x_number):
                object_points[:2, count] = np.array(
                    [
                        (self.target_x_number - j - 1) * self.target_cell_size,
                        (self.target_y_number - i - 1) * self.target_cell_size,
                    ]
                )
                count += 1
        retval, rvec, tvec = cv2.solvePnP(
            object_points.T, corner_points.T, self.K, distCoeffs=self.distortion
        )
        Matrix_target2camera = np.column_stack(((cv2.Rodrigues(rvec))[0], tvec))
        Matrix_target2camera = np.row_stack(
            (Matrix_target2camera, np.array([0, 0, 0, 1]))
        )
        R_target2camera = Matrix_target2camera[:3, :3]
        T_target2camera = Matrix_target2camera[:3, 3].reshape((3, 1))
        return R_target2camera, T_target2camera

    def process(self, img_path, pose_path):
        image_list = []
        for root, dirs, files in os.walk(img_path):
            if files:
                for file in files:
                    image_name = os.path.join(root, file)
                    image_list.append(image_name)
        R_target2camera_list = []
        T_target2camera_list = []
        for img_path in image_list:
            img = cv2.imread(img_path)
            R_target2camera, T_target2camera = self.target2camera(img)
            R_target2camera_list.append(R_target2camera)
            T_target2camera_list.append(T_target2camera)
        R_gripper2base_list = []
        T_gripper2base_list = []
        data = xlrd2.open_workbook(pose_path)
        table = data.sheets()[0]
        for row in range(table.nrows):
            x = table.cell_value(row, 0)
            y = table.cell_value(row, 1)
            z = table.cell_value(row, 2)
            tx = table.cell_value(row, 3)
            ty = table.cell_value(row, 4)
            tz = table.cell_value(row, 5)
            R_gripper2base, T_gripper2base = self.gripper2base(x, y, z, tx, ty, tz)
            R_gripper2base_list.append(R_gripper2base)
            T_gripper2base_list.append(T_gripper2base)
        R_camera2base, T_camera2base = cv2.calibrateHandEye(
            R_gripper2base_list,
            T_gripper2base_list,
            R_target2camera_list,
            T_target2camera_list,
        )
        return (
            R_camera2base,
            T_camera2base,
            R_gripper2base_list,
            T_gripper2base_list,
            R_target2camera_list,
            T_target2camera_list,
        )

    def check_result(self, R_cb, T_cb, R_gb, T_gb, R_tc, T_tc):
        for i in range(len(R_gb)):
            RT_gripper2base = np.column_stack((R_gb[i], T_gb[i]))
            RT_gripper2base = np.row_stack((RT_gripper2base, np.array([0, 0, 0, 1])))
            RT_base2gripper = np.linalg.inv(RT_gripper2base)
            print(RT_base2gripper)

            RT_camera_to_base = np.column_stack((R_cb, T_cb))
            RT_camera_to_base = np.row_stack(
                (RT_camera_to_base, np.array([0, 0, 0, 1]))
            )
            print(RT_camera_to_base)

            RT_target_to_camera = np.column_stack((R_tc[i], T_tc[i]))
            RT_target_to_camera = np.row_stack(
                (RT_target_to_camera, np.array([0, 0, 0, 1]))
            )
            RT_camera2target = np.linalg.inv(RT_target_to_camera)
            print(RT_camera2target)

            RT_target_to_gripper = (
                RT_base2gripper @ RT_camera_to_base @ RT_camera2target
            )
            print("第{}次验证结果为:".format(i))
            print(RT_target_to_gripper)
            print("")


def read_urdf(urdf_content: str) -> kinpy.chain.SerialChain:
    chain = kinpy.build_serial_chain_from_urdf(urdf_content, "gripper_static_1")
    return chain


def forward_kinematics(
    chain: kinpy.chain.SerialChain, joint_angles: Sequence[float | int]
) -> kinpy.Transform:
    # 使用kinpy计算正运动学
    return chain.forward_kinematics(joint_angles, end_only=True)  # type: ignore


goal_angles = [0, 0, 0, 0, 0]
goal_gripper_angle = 80


def on_press(key):
    global goal_angles
    try:
        key = key.char
    except AttributeError:
        return
    if key == "1":
        goal_angles[0] += 10
    elif key == "3":
        goal_angles[0] -= 10
    elif key == "q":
        goal_angles[1] += 10
    elif key == "e":
        goal_angles[1] -= 10
    elif key == "a":
        goal_angles[2] += 10
    elif key == "d":
        goal_angles[2] -= 10
    elif key == "z":
        goal_angles[3] += 10
    elif key == "c":
        goal_angles[3] -= 10
    elif key == "x":
        goal_angles[4] += 10
    elif key == "v":
        goal_angles[4] -= 10


def on_release(key):
    pass


def get_input():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# 收集图片和位姿数据，生成图片和excel文件
def collect_image_pose():
    global goal_angles, goal_gripper_angle
    image_path = os.path.join(os.path.dirname(__file__), "hand-eye-data", "images/")
    pose_path = os.path.join(os.path.dirname(__file__), "hand-eye-data", "pose.xlsx")
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path))
    if not os.path.exists(os.path.dirname(pose_path)):
        os.makedirs(os.path.dirname(pose_path))

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
    arm = Arm("COM3")
    arm.disable_torque()
    time.sleep(1)
    data = xlwt.Workbook()
    table = data.add_sheet("pose")
    rows_num = 0
    cam = open_camera(color=True, depth=False)
    while True:
        try:
            # arm.set_arm_angles(
            #     [goal_angles[i] for i in range(len(goal_angles))],
            #     gripper_angle=goal_gripper_angle,
            # )
            angles, gripper = arm.get_read_arm_angles()
            if angles is None:
                print("获取机械臂角度失败")
                continue
            print("机械臂角度:", angles, "夹爪状态:", gripper)
            end_pose = forward_kinematics(chain=chain, joint_angles=angles)
            print("夹爪位置:", end_pose.pos)
            print("夹爪旋转矩阵:\n", end_pose.rot_mat)

            frames = get_frames(cam)
            color_image = frames.get("color")
            if color_image is None:
                print("failed to get color image")
                continue
            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord(" "):
                cv2.imwrite(f"{image_path}/{time.time()}.png", color_image)
                new_row = [
                    end_pose.pos[0],
                    end_pose.pos[1],
                    end_pose.pos[2],
                    end_pose.rot_euler[0],
                    end_pose.rot_euler[1],
                    end_pose.rot_euler[2],
                ]
                for i in range(len(new_row)):
                    table.write(rows_num, i, new_row[i])
                rows_num += 1
            # elif key == ord("g"):
            #     goal_gripper_angle = 0 if goal_gripper_angle == 80 else 80
            #     arm.set_arm_angles(goal_angles, goal_gripper_angle)
            #     print("夹爪状态:", goal_gripper_angle)
        except KeyboardInterrupt:
            break
    data.save(pose_path)
    cv2.destroyAllWindows()
    close_camera(cam)
    arm.disconnect_arm()

    return image_path, pose_path


def main():
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    image_path, pose_path = collect_image_pose()
    calibrator = Calibration()
    R_cb, T_cb, R_gb, T_gb, R_tc, T_tc = calibrator.process(image_path, pose_path)
    calibrator.check_result(R_cb, T_cb, R_gb, T_gb, R_tc, T_tc)


if __name__ == "__main__":
    main()
