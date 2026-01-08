import os
from pydoc import text
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from config_getter import get_config_value
from pydantic import TypeAdapter
from llm.dataclass import DetectedFromLLM
from llm.audio2text import get_audio_text
import cv2
import numpy as np
import concurrent.futures
from queue import Queue
from arm.arm_control import Arm
from threading import Thread

from llm.llm_detect import LLMDetect, json2box, draw_boxes_on_frame


def catch_by_audio():
    text = get_audio_text(
        appid=get_config_value("APPID"),
        api_key=get_config_value("APIKey"),
        api_secret=get_config_value("APISecret"),
    )


def catch_by_instruction(
    frame: cv2.typing.MatLike, instruction: str, queue_output: Queue
):
    """根据和画面指令阻塞获取检测结果，执行抓取动作，并将检测结果放入队列中"""
    offset = get_config_value("catch_offset")
    default_gripper_aside_pos = get_config_value("default_gripper_aside_pos")
    llm_detect = LLMDetect()
    arm = Arm()
    # arm.move_to_home(gripper_angle_deg=80)
    arm.move_to(default_gripper_aside_pos, 80)
    response_task = llm_detect.detect_frame(
        frame,
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
                box = json2box(response, img_w=frame.shape[1], img_h=frame.shape[0])
                print("检测到的目标:", box)
                if box:
                    # 考虑旋转180度
                    box.box_center_x = frame.shape[1] - box.box_center_x
                    box.box_center_y = frame.shape[0] - box.box_center_y
                    queue_output.put(box)
                    # 将图像坐标转换为机械臂坐标系
                    target_x, target_y = arm.pixel2pos(
                        box.box_center_x,
                        box.box_center_y,
                    )
                    gripper_angle_rad = arm.gripper_angle_by_longer(
                        box.box_center_x,
                        box.box_center_y,
                        box.box_width,
                        box.box_height,
                        box.box_rotation_deg,
                    )
                    arm.catch_and_place(
                        target_x + offset * np.cos(gripper_angle_rad),
                        target_y + offset * np.sin(-gripper_angle_rad),
                        gripper_angle_rad,
                        [0.2, 0.0],
                    )
            if done:
                break
    else:
        print(f"No response({response_task}) or frame({frame}) available.")
    arm.disconnect_arm()


def main():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    instructions = [
        "抓取最近的积木",
        "抓取红色积木",
        "抓取最右边的红色积木",
        "抓取最右边的黄色积木",
        "抓取最上面的蓝色积木",
        "抓取最远的蓝色积木",
    ]
    future = None
    box_queue = Queue()
    frame = None
    cam = Camera()

    def consumption_thread():
        nonlocal frame
        box = None
        while True:
            if not box_queue.empty():
                box = box_queue.get(block=False)
                if box:
                    print("消费到的目标:", box)
            if frame is None:
                continue
            frame_draw = draw_boxes_on_frame(
                boxes=[box] if box else [],
                frame=frame,
            )
            cv2.imshow("LLM Detection", frame_draw)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                cv2.destroyAllWindows()
                break

    thread = Thread(target=consumption_thread)
    thread.start()
    while True:
        frame = cam.get_frames().get("color", None)
        if frame is None:
            print("Failed to grab frame")
            continue
        if future is None or future.done():
            future = executor.submit(
                catch_by_instruction,
                frame,
                instructions[0],
                # instructions[np.random.randint(0, len(instructions))],
                box_queue,
            )
        if not thread.is_alive():
            break
    cam.close()


if __name__ == "__main__":
    main()
