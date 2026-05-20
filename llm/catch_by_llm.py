import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from utils.config_getter import get_config_value
from pydantic import TypeAdapter
from llm.dataclass import DetectedBox, DetectedFromLLM
from llm.audio2text import get_audio_text
import cv2
import numpy as np
import concurrent.futures
import time
from queue import Queue
from arm.arm_base import Arm
from threading import Thread
from typing import Callable, Optional

from llm.llm_detect import LLMDetect, json2box, draw_boxes_on_frame
from utils.cv2_display import (
    show_image,
    poll_key,
    set_mouse_callback,
    destroy_all_windows,
)

CATCH_STATS_FILE = os.path.join(os.path.dirname(__file__), "catch_stats.json")

# 每次启动从0开始统计
catch_stats = {"total": 0, "success": 0, "fail": 0, "success_rate": "", "history": []}


def record_catch_result(instruction: str, target_name: str, success: bool):
    """记录一次抓取结果并立即写入文件"""
    catch_stats["total"] += 1
    if success:
        catch_stats["success"] += 1
    else:
        catch_stats["fail"] += 1
    rate = round(catch_stats["success"] / catch_stats["total"] * 100, 2)
    catch_stats["success_rate"] = f"{rate}%"
    catch_stats["history"].append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instruction": instruction,
            "target": target_name,
            "success": success,
        }
    )
    with open(CATCH_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(catch_stats, f, ensure_ascii=False, indent=2)
    print(
        f"[统计] 总计: {catch_stats['total']}, 成功: {catch_stats['success']}, "
        f"失败: {catch_stats['fail']}, 成功率: {rate}%"
    )


arm = Arm()
box_queue = Queue()
frame = None


def consumption_thread():
    global frame, box_queue
    idx = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal idx
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left button clicked at ({x}, {y})")
            # 手动点击目标，替换大模型识别结果，可靠性要求高的时候启用
            # names = "8周年快乐"
            # box_queue.put(
            #     DetectedBox(
            #         class_name=names[idx],
            #         box_center_x=x,
            #         box_center_y=y,
            #         box_width=100,
            #         box_height=150,
            #     )
            # )
            # idx += 1

    box = None
    window_name = "Camera"
    set_mouse_callback(window_name, mouse_callback)

    while True:
        if not box_queue.empty():
            box = box_queue.get(block=False)
        if frame is None:
            continue
        frame_draw = draw_boxes_on_frame(
            boxes=[box] if box else [],
            frame=frame,
        )
        show_image(window_name, frame_draw)
        if poll_key(1) & 0xFF == 27:  # Press 'ESC' to exit
            destroy_all_windows()
            arm.move_to_home()
            arm.disconnect_arm()
            break


def catch_by_audio():
    global frame, box_queue
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    audio_future = None
    text = None
    cam = Camera()

    thread = Thread(target=consumption_thread)
    thread.start()
    while True:
        frame = cam.get_frames().get("color", None)
        if frame is None:
            print("Failed to grab frame")
            continue
        if future is None or future.done():
            if audio_future and audio_future.done():
                text = audio_future.result()
                audio_future = None
                print("Audio to Text:", text)
            if text:
                future = executor.submit(
                    catch_by_instruction,
                    frame,
                    text,
                    box_queue,
                )
                # 消费完指令后清空
                text = None
            # 不在抓取进行中，且没有抓取指令时，继续获取语音指令
            if (future is None or future.done()) and (
                audio_future is None or audio_future.done()
            ):
                audio_future = audio_executor.submit(
                    get_audio_text,
                )
                box_queue.put(None)  # 清空当前目标框
        if not thread.is_alive():
            break
    cam.close()
    exit(0)


def catch_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    success_callback: Optional[Callable[[], None]] = None,
):
    """根据和画面指令阻塞获取检测结果，执行抓取动作，并将检测结果放入队列中"""
    try:
        global arm
        llm_detect = LLMDetect()
        print("Instruction:", instruction)
        place_pos = get_config_value("place_pos")
        offset = get_config_value("catch_offset")
        default_gripper_aside_pos = get_config_value(
            "default_gripper_aside_pos", raise_if_missing=False
        )
        if default_gripper_aside_pos is not None:
            arm.move_to(default_gripper_aside_pos)
        time.sleep(0.5)
        print("LLM Detecting...")
        start = time.time()
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
                    print(f"Detect used {time.time()-start}s")
                    # print("LLM Response:", response)
                    box = json2box(response, img_w=frame.shape[1], img_h=frame.shape[0])
                    print("检测到的目标:", box)
                    if box:
                        # 手动点击目标，替换大模型识别结果，可靠性要求高的时候启用
                        # box = queue_output.get(block=True)
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
                        found = False
                        class_place_pos = [0.1, 0.0]
                        for name, pos in place_pos.items():
                            for keyword in pos.get("keywords", []):
                                if keyword in box.class_name.lower():
                                    print(f"放置到'{name}'区域")
                                    class_place_pos = pos.get("pos", class_place_pos)
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            print("未知积木，放置到默认区域")
                        catch_success = arm.catch_and_place(
                            target_x + offset * np.cos(gripper_angle_rad),
                            target_y + offset * np.sin(-gripper_angle_rad),
                            gripper_angle_rad,
                            class_place_pos,
                        )
                        record_catch_result(instruction, box.class_name, catch_success)
                        if catch_success and success_callback:
                            success_callback()
                if done:
                    break
        else:
            print(f"No response({response_task}) or frame({frame}) available.")
    except Exception as e:
        print("Exception: ", e)


def catch_by_text_instruction():
    global frame, box_queue
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    instructions = [
        "抓取最近的积木",
        "抓取红色积木",
        "抓取最右边的红色积木",
        "抓取最右边的黄色积木",
        "抓取最上面的蓝色积木",
        "抓取最远的蓝色积木",
        "抓取最右边的积木",
    ]
    future = None
    cam = Camera()

    thread = Thread(target=consumption_thread)
    thread.start()
    while True:
        frame = cam.get_frames().get("color", None)
        if frame is None:
            print("Failed to grab frame")
            continue
        if future is None or future.done():
            if len(instructions) > 0:
                instruction = instructions[0]
                # instruction = instructions[np.random.randint(0, len(instructions))]
                future = executor.submit(
                    catch_by_instruction,
                    frame,
                    instruction,
                    box_queue,
                    lambda: instructions.remove(instruction),
                )
            else:
                default_gripper_aside_pos = get_config_value(
                    "default_gripper_aside_pos", raise_if_missing=False
                )
                if default_gripper_aside_pos is not None:
                    future = executor.submit(
                        arm.move_to,
                        default_gripper_aside_pos,
                    )
        if not thread.is_alive():
            break
    cam.close()


if __name__ == "__main__":
    catch_by_text_instruction()
    # catch_by_audio()
