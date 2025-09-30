# display.launch.py: 用于在RViz中显示机器人模型
# 运行ros2 launch display.launch.py
# 进入rviz2后，左侧Fixed Frame选择base_link
# 如果没有显示模型，点击左侧Add按钮，选择RobotModel添加
# 再选择RobotModel，在右侧选择Description Topic为/robot_description
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    # !! 修改这里 !! -> 指向你的URDF文件路径
    urdf_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "low_cost_robot.urdf",
    )

    # 确保URDF文件存在
    if not os.path.exists(urdf_file_path):
        raise IOError(f"URDF file not found at {urdf_file_path}")

    with open(urdf_file_path, "r") as infp:
        robot_desc = infp.read()

    # replace $(find low_cost_robot_description) with the actual path
    robot_desc = robot_desc.replace(
        "$(find low_cost_robot_description)",
        os.path.dirname(os.path.abspath(__file__)),
    )

    return LaunchDescription(
        [
            # Robot State Publisher: 发布 /robot_description 和 /tf
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_desc}],
            ),
            # Joint State Publisher GUI: 提供一个GUI来控制关节
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
            ),
            # RViz2: 可视化工具
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
            ),
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
