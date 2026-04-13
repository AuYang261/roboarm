import os
import sys
import json
import time
import threading
from collections import Counter
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from config_getter import get_config_value
import cv2
from PIL import Image, ImageDraw
from pydantic import TypeAdapter
import base64
from openai.types.chat.chat_completion import ChatCompletion
from concurrent import futures
import numpy as np
from typing import Any

from llm.llm_api import LLMAPI, extract_json_from_markdown, inline_schema_refs, font
from llm.dataclass import DetectedBox, DetectedFromLLM

# 每次检测并行发送的请求数，结果取平均以提高准确率（不大时不增加响应时间）
BATCH_SIZE = 1


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
        if get_config_value("RotationCam2Arm"):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        _, img_encoded = cv2.imencode(".jpg", frame)
        image_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        # batch size为1则简单地发单次请求返回
        if BATCH_SIZE == 1:
            return self.llm_api.chat_img_async(
                image_base64=image_base64,
                prompt_key=prompt_key,
                replace_map=replace_map,
                schema=schema,
            )

        # 通过公开 API 并行发送 BATCH_SIZE 次请求
        sub_tasks: list[futures.Future[ChatCompletion]] = []
        for _ in range(BATCH_SIZE):
            task = self.llm_api.chat_img_async(
                image_base64=image_base64,
                prompt_key=prompt_key,
                replace_map=replace_map,
                schema=schema,
            )
            if task is not None:
                sub_tasks.append(task)
        if not sub_tasks:
            return None

        # 用后台线程等待所有子任务完成并聚合，返回单个 Future
        aggregated_future: futures.Future[Any] = futures.Future()

        def _wait_and_aggregate() -> None:
            try:
                json_strs: list[str | None] = []
                for t in sub_tasks:
                    try:
                        completion = t.result()  # 阻塞等待单个子任务
                        if completion.choices:
                            json_strs.append(completion.choices[0].message.content)
                        else:
                            json_strs.append(None)
                    except Exception as e:
                        print(f"批量请求中某次失败: {e}")
                        json_strs.append(None)

                is_multi = isinstance(schema, dict) and schema.get("type") == "array"
                aggregated = (
                    _aggregate_boxes_json(json_strs)
                    if is_multi
                    else _aggregate_box_json(json_strs)
                )

                msg_mock = SimpleNamespace(content=aggregated)
                choice_mock = SimpleNamespace(message=msg_mock)
                mock = SimpleNamespace(
                    choices=[choice_mock] if aggregated is not None else []
                )
                aggregated_future.set_result(mock)
            except Exception as e:
                aggregated_future.set_exception(e)

        threading.Thread(target=_wait_and_aggregate, daemon=True).start()
        return aggregated_future


def _filter_outlier_boxes(boxes: list[DetectedFromLLM]) -> list[DetectedFromLLM]:
    """基于中位数距离剔除坐标异常的检测框。"""
    if len(boxes) <= 2:
        return boxes
    cxs = np.array([b.box_center_x for b in boxes])
    cys = np.array([b.box_center_y for b in boxes])
    cx_med, cy_med = float(np.median(cxs)), float(np.median(cys))
    dists = np.sqrt((cxs - cx_med) ** 2 + (cys - cy_med) ** 2)
    threshold = max(float(np.median(dists)) * 2, 0.05)
    keep = [b for b, d in zip(boxes, dists) if d <= threshold]
    return keep if keep else boxes


def _aggregate_box_json(json_strs: list[str | None]) -> str | None:
    """聚合多次单目标检测响应，去除异常值取均值，返回归一化坐标的 JSON 字符串。"""
    valid: list[DetectedFromLLM] = []
    for s in json_strs:
        if s is None:
            continue
        try:
            box = TypeAdapter(DetectedFromLLM).validate_json(
                extract_json_from_markdown(s)
            )
            if box.is_valid():
                valid.append(box)
        except Exception as e:
            print(f"解析/校验 JSON 失败: {e}")
    if not valid:
        return None
    filtered = _filter_outlier_boxes(valid)
    class_name = Counter(b.class_name for b in filtered).most_common(1)[0][0]
    return json.dumps(
        {
            "id": filtered[0].id,
            "class_name": class_name,
            "box_center_x": float(np.mean([b.box_center_x for b in filtered])),
            "box_center_y": float(np.mean([b.box_center_y for b in filtered])),
            "box_width": float(np.mean([b.box_width for b in filtered])),
            "box_height": float(np.mean([b.box_height for b in filtered])),
        }
    )


def _aggregate_boxes_json(json_strs: list[str | None]) -> str:
    """聚合多次多目标检测响应，空间聚类后各簇取均值，返回 JSON 数组字符串。"""
    all_boxes: list[DetectedFromLLM] = []
    total = sum(1 for s in json_strs if s is not None)
    for s in json_strs:
        if s is None:
            continue
        try:
            boxes = TypeAdapter(list[DetectedFromLLM]).validate_json(
                extract_json_from_markdown(s)
            )
            all_boxes.extend(b for b in boxes if b.is_valid())
        except Exception as e:
            print(f"解析/校验 JSON 失败: {e}")
    if not all_boxes:
        return "[]"

    CLUSTER_DIST = 0.1  # 归一化坐标下同一目标跨响应的最大允许偏移
    remaining = list(all_boxes)
    result = []
    while remaining:
        seed = remaining.pop(0)
        cluster, outliers = [seed], []
        for b in remaining:
            if b.class_name == seed.class_name:
                dist = float(
                    np.sqrt(
                        (b.box_center_x - seed.box_center_x) ** 2
                        + (b.box_center_y - seed.box_center_y) ** 2
                    )
                )
                (cluster if dist <= CLUSTER_DIST else outliers).append(b)
            else:
                outliers.append(b)
        remaining = outliers
        # 只保留在超过半数请求中均出现的目标（过滤幻觉）
        if len(cluster) >= max(1, total // 2):
            result.append(
                {
                    "id": cluster[0].id,
                    "class_name": cluster[0].class_name,
                    "box_center_x": float(np.mean([b.box_center_x for b in cluster])),
                    "box_center_y": float(np.mean([b.box_center_y for b in cluster])),
                    "box_width": float(np.mean([b.box_width for b in cluster])),
                    "box_height": float(np.mean([b.box_height for b in cluster])),
                }
            )
    return json.dumps(result)


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
        start = time.time()
        response_task, frame = llm_detect.detect_scene(
            prompt_key="block_detect_prompt",
            schema=inline_schema_refs(TypeAdapter(list[DetectedFromLLM]).json_schema()),
        )
        if response_task and frame is not None:
            while True:
                response, done = llm_detect.llm_api.await_task(
                    response_task, blocking=False
                )
                if response:
                    print(f"Detect used {time.time()-start}s")
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
