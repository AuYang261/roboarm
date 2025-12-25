import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pydantic import TypeAdapter
from llm.dataclass import DetectedFromLLM
import cv2
import numpy as np
import concurrent.futures
import yaml
from arm.arm_control import Arm

from llm.llm_detect import LLMDetect, json2box, draw_boxes_on_frame


def main():
    config_yaml = yaml.safe_load(
        open(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
            encoding="utf-8",
        )
    )
    offset = config_yaml["catch_offset"]
    default_gripper_aside_pos = config_yaml["default_gripper_aside_pos"]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    llm_detect = LLMDetect()
    arm = Arm()
    arm.move_to_home(gripper_angle_deg=80)
    frame_draw = None
    instructions = [
        "抓取最近的积木",
        "抓取红色积木",
        "抓取最右边的红色积木",
        "抓取最右边的黄色积木",
        "抓取最上面的蓝色积木",
        "抓取最远的蓝色积木",
    ]
    future = None
    box = None
    while True:
        if future is None or future.done():
            arm.move_to(default_gripper_aside_pos, 80)
            instruction = np.random.choice(instructions)
            print("Instruction:", instruction)
            response_task, frame = llm_detect.detect_scene(
                prompt_key="user_instruction_prompt",
                replace_map={"{user_instruction}": instruction},
                schema=TypeAdapter(DetectedFromLLM).json_schema(),
            )
            if response_task and frame is not None:
                while True:
                    response, done = llm_detect.llm_api.await_task(
                        response_task, blocking=False
                    )
                    if response:
                        print("LLM Response:", response)
                        box = json2box(
                            response, img_w=frame.shape[1], img_h=frame.shape[0]
                        )
                        print("检测到的目标:", box)
                        if box:
                            # 将图像坐标转换为机械臂坐标系
                            # 考虑旋转180度
                            target_x, target_y = arm.pixel2pos(
                                frame.shape[1] - box.box_center_x,
                                frame.shape[0] - box.box_center_y,
                            )
                            gripper_angle_rad = arm.gripper_angle_by_longer(
                                box.box_center_x,
                                box.box_center_y,
                                box.box_width,
                                box.box_height,
                                box.box_rotation_deg,
                            )
                            future = executor.submit(
                                arm.catch_and_place,
                                # 夹爪向外偏移一些，避免刚好顶到物体
                                target_x + offset * np.cos(gripper_angle_rad),
                                target_y + offset * np.sin(-gripper_angle_rad),
                                gripper_angle_rad,
                                [0.2, 0.0],
                            )
                    frame_draw = draw_boxes_on_frame(
                        boxes=[box] if box else [],
                        frame=frame,
                    )
                    cv2.imshow("LLM Detection", frame_draw)
                    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                        arm.disconnect_arm()
                        cv2.destroyAllWindows()
                        exit()
                    if done:
                        break
            else:
                print(f"No response({response_task}) or frame({frame}) available.")
        if frame_draw is not None:
            cv2.imshow("LLM Detection", frame_draw)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break
    arm.disconnect_arm()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
