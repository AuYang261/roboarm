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

from llm.llm_api import LLMAPI, extract_json_from_markdown
from llm.dataclass import DetectedBox, DetectedFromLLM


font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)


class LLMDetect:
    def __init__(self):
        self.llm_api = LLMAPI()
        self.camera = Camera()

    def detect_scene(
        self,
        prompt_key: str,
    ) -> tuple["futures.Future[ChatCompletion]|None", cv2.typing.MatLike | None]:
        frames = self.camera.get_frames()
        color_frame = frames.get("color", None)
        if color_frame is None:
            print("Failed to grab frame")
            return None, None

        _, img_encoded = cv2.imencode(".jpg", color_frame)
        image_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        response_task = self.llm_api.chat_img_async(
            image_base64=image_base64,
            prompt_key=prompt_key,
            schema=TypeAdapter(list[DetectedFromLLM]).json_schema(),
        )
        return response_task, color_frame


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


def json2boxes(json_str: str, img_w: int, img_h: int) -> list[DetectedBox]:
    try:
        boxes: list[DetectedFromLLM] = TypeAdapter(list[DetectedFromLLM]).validate_json(
            extract_json_from_markdown(json_str)
        )
    except Exception as e:
        print(f"解析/校验 JSON 失败: {e}")
        return []
    return [item.to_detected_box(img_w, img_h) for item in boxes]


if __name__ == "__main__":
    llm_detect = LLMDetect()
    frame_draw = None
    while True:
        response_task, frame = llm_detect.detect_scene(prompt_key="block_detect_prompt")
        if response_task and frame is not None:
            while True:
                response, done = LLMAPI.await_task(response_task, blocking=False)
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
