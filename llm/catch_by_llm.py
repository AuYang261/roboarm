import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from config_getter import get_config_value
from pydantic import TypeAdapter
from llm.dataclass import DetectedFromLLM
from llm.audio2text import get_audio_text
import cv2
import numpy as np
import concurrent.futures
import time
from queue import Queue
from arm.arm_control import Arm
from threading import Thread
from typing import Callable, Optional

from llm.llm_detect import LLMDetect, json2box, draw_boxes_on_frame

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
llm_detect = LLMDetect()

def catch_by_audio():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    audio_future = None
    box_queue = Queue()
    frame = None
    text = None
    cam = Camera()

    def consumption_thread():
        nonlocal frame
        box = None
        while True:
            if not box_queue.empty():
                box = box_queue.get(block=False)
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


def catch_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    success_callback: Optional[Callable[[], None]] = None,
):
    """根据和画面指令阻塞获取检测结果，执行抓取动作，并将检测结果放入队列中"""
    try:
        global arm, llm_detect
        print("Instruction:", instruction)
        class_pos = get_config_value("class_pos")
        offset = get_config_value("catch_offset")
        default_gripper_aside_pos = get_config_value("default_gripper_aside_pos")
        arm.move_to(default_gripper_aside_pos, 80)
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
                        if "红" in box.class_name or "red" in box.class_name.lower():
                            print("红色积木，放置到红色区域")
                            place_pos = class_pos.get("red_block")
                        elif (
                            "黄" in box.class_name or "yellow" in box.class_name.lower()
                        ):
                            print("黄色积木，放置到黄色区域")
                            place_pos = class_pos.get("yellow_block")
                        elif "蓝" in box.class_name or "blue" in box.class_name.lower():
                            print("蓝色积木，放置到蓝色区域")
                            place_pos = class_pos.get("blue_block")
                        elif (
                            "绿" in box.class_name or "green" in box.class_name.lower()
                        ):
                            print("绿色积木，放置到绿色区域")
                            place_pos = class_pos.get("green_block")
                        else:
                            print("未知积木，放置到默认区域")
                            place_pos = class_pos.get("blue_block")
                        catch_success = arm.catch_and_place(
                            target_x + offset * np.cos(gripper_angle_rad),
                            target_y + offset * np.sin(-gripper_angle_rad),
                            gripper_angle_rad,
                            place_pos,
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
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    instructions = [
        # "抓取蓝色积木",
        # "抓取黄色积木",
        # "抓取红色积木",
        # "抓取绿色积木",
        "抓取最近的积木",
        "抓取红色积木",
        "抓取最右边的红色积木",
        "抓取最右边的黄色积木",
        "抓取最上面的蓝色积木",
        "抓取最远的蓝色积木",
        "抓取最右边的积木",
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
            if len(instructions) > 0:
                # instruction = instructions[0]
                instruction = instructions[np.random.randint(0, len(instructions))]
                future = executor.submit(
                    catch_by_instruction,
                    frame,
                    instruction,
                    box_queue,
                    lambda: instructions.remove(instruction),
                )
            else:
                future = executor.submit(
                    arm.move_to,
                    get_config_value("default_gripper_aside_pos"),
                )
        if not thread.is_alive():
            break
    cam.close()


if __name__ == "__main__":
    catch_by_text_instruction()
    # catch_by_audio()
