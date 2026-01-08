import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
import cv2
from PIL import Image, ImageDraw, ImageFont
from pydantic import TypeAdapter
import base64
from openai.types.chat.chat_completion import ChatCompletion
from concurrent import futures
import numpy as np
from typing import Any

from llm.llm_api import LLMAPI, extract_json_from_markdown
from llm.dataclass import DetectedBox, DetectedFromLLM


font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)


class LLMDetect:
    def __init__(self):
        self.llm_api = LLMAPI()
        self.camera = None

    def detect_scene(
        self,
        prompt_key: str,
        replace_map: dict[str, str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> tuple["futures.Future[ChatCompletion]|None", cv2.typing.MatLike | None]:
        if self.camera is None:
            self.camera = Camera(color=True, depth=False)
        frames = self.camera.get_frames()
        for _ in range(10):
            frames = self.camera.get_frames()
        color_frame = frames.get("color", None)
        if color_frame is None:
            print("Failed to grab frame")
            return None, None

        return (
            self.detect_frame(color_frame, prompt_key, replace_map, schema),
            color_frame,
        )

    def detect_frame(
        self,
        frame: cv2.typing.MatLike,
        prompt_key: str,
        replace_map: dict[str, str] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> "futures.Future[ChatCompletion]|None":
        # 旋转180度以适应摄像头安装方向，需要根据实际安装情况调整
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        _, img_encoded = cv2.imencode(".jpg", frame)
        image_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        response_task = self.llm_api.chat_img_async(
            image_base64=image_base64,
            prompt_key=prompt_key,
            replace_map=replace_map,
            schema=schema,
        )
        return response_task


def draw_boxes_on_frame(
    boxes: list[DetectedBox],
    frame: cv2.typing.MatLike,
) -> cv2.typing.MatLike:
    annotated_frame = frame.copy()
    pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for box in boxes:
        x_center, y_center = box.box_center_x, box.box_center_y
        width_box, height_box = box.box_width, box.box_height
        x1 = int((x_center - width_box / 2))
        y1 = int((y_center - height_box / 2))
        x2 = int((x_center + width_box / 2))
        y2 = int((y_center + height_box / 2))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 20), box.class_name, fill="red", font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def json2box(json_str: str, img_w: int, img_h: int) -> DetectedBox | None:
    try:
        box: DetectedFromLLM = TypeAdapter(DetectedFromLLM).validate_json(
            extract_json_from_markdown(json_str)
        )
    except Exception as e:
        print(f"解析/校验 JSON 失败: {e}")
        return None
    return box.to_detected_box(img_w, img_h) if box.is_valid() else None


def json2boxes(json_str: str, img_w: int, img_h: int) -> list[DetectedBox]:
    try:
        boxes: list[DetectedFromLLM] = TypeAdapter(list[DetectedFromLLM]).validate_json(
            extract_json_from_markdown(json_str)
        )
    except Exception as e:
        print(f"解析/校验 JSON 失败: {e}")
        return []
    return [box.to_detected_box(img_w, img_h) for box in boxes if box.is_valid()]


if __name__ == "__main__":
    llm_detect = LLMDetect()
    frame_draw = None
    while True:
        response_task, frame = llm_detect.detect_scene(
            prompt_key="block_detect_prompt",
            schema=TypeAdapter(list[DetectedFromLLM]).json_schema(),
        )
        if response_task and frame is not None:
            while True:
                response, done = llm_detect.llm_api.await_task(
                    response_task, blocking=False
                )
                if response:
                    boxes = json2boxes(
                        response, img_w=frame.shape[1], img_h=frame.shape[0]
                    )
                    print("检测到的目标:", boxes)
                    frame_draw = draw_boxes_on_frame(
                        boxes=boxes,
                        frame=frame,
                    )
                elif frame_draw is None:
                    frame_draw = frame
                cv2.imshow("LLM Detection", frame_draw)
                if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                    exit()
                if done:
                    break
        else:
            print(f"No response({response_task}) or frame({frame}) available.")
