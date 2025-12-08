import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from turtle import width
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI, AsyncOpenAI
import base64
import time
import yaml
import toml
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import Any
import asyncio
from pydantic import TypeAdapter
import re
import threading
from concurrent import futures

from llm.dataclass import DetectedFromLLM


class LLMAPI:

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1"):
        config_yaml = yaml.safe_load(
            open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
                encoding="utf-8",
            )
        )
        prompts_file = config_yaml["prompts_file"]
        self.prompts = toml.load(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), prompts_file)
        )["prompts"]
        self.client = OpenAI(
            base_url=base_url,
            api_key=config_yaml["openrouter_api_key"],
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=config_yaml["openrouter_api_key"],
        )

        # 启动后台事件循环线程
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def chat_img(
        self,
        image_base64: str,
        prompt_key: str,
        model: str = "google/gemini-3-pro-preview",
        debug: bool = False,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> str | None:
        start_time = time.time()
        prompt = self.prompts.get(prompt_key, None)
        if prompt is None:
            print(f"Prompt key '{prompt_key}' not found.")
            return None
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=temperature,
            response_format=(
                {"type": "json_object"}
                if schema is None
                else {
                    "type": "json_schema",
                    "json_schema": {"name": "detection_output", "schema": schema},
                }
            ),
        )
        if debug:
            print(f"Time taken: {time.time() - start_time} seconds")
        if completion.choices is None or len(completion.choices) == 0:
            print("No choices returned from the model.")
            return None
        return completion.choices[0].message.content

    def chat_img_async(
        self,
        image_base64: str,
        prompt_key: str,
        model: str = "google/gemini-3-pro-preview",
        debug: bool = False,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> "futures.Future[ChatCompletion] | None":
        start_time = time.time()
        prompt = self.prompts.get(prompt_key, None)
        if prompt is None:
            print(f"Prompt key '{prompt_key}' not found.")
            return None
        completion_coroutine = self.async_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=temperature,
            response_format=(
                {"type": "json_object"}
                if schema is None
                else {
                    "type": "json_schema",
                    "json_schema": {"name": "detection_output", "schema": schema},
                }
            ),
        )
        if debug:
            print(f"Send time taken: {time.time() - start_time} seconds")
        return asyncio.run_coroutine_threadsafe(completion_coroutine, self._loop)

    @staticmethod
    def await_task(
        task: "futures.Future[ChatCompletion]",
        blocking: bool = False,
    ) -> tuple[str | None, bool]:
        """
        等待异步聊天协程完成并返回结果。
        返回(结果str/失败None，是否结束)，后者仅在非阻塞模式下有意义。
        """
        if blocking:
            start_time = time.time()
            completion = task.result()
            print(f"Await time taken: {time.time() - start_time} seconds")
        else:
            if not task.done():
                return None, False
            else:
                completion = task.result()
        if completion.choices is None or len(completion.choices) == 0:
            print("No choices returned from the model.")
            return None, True
        return completion.choices[0].message.content, True


def extract_json_from_markdown(text: str) -> str:
    """
    从可能包含 Markdown 代码围栏的文本中提取 JSON 内容。
    支持 ```json ... ``` 或 ``` ... ```，返回内部内容（去掉围栏）。
    若未找到围栏，原样返回。
    """
    # 优先匹配标注语言的 fenced code block
    m = re.search(r"```(?:json|JSON)\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # 退化匹配非标注语言的 fenced code block
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text


if __name__ == "__main__":
    llm_api = LLMAPI()

    image_path = os.path.join(os.path.dirname(__file__), "test.png")
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    result = None
    while result is None:
        task = llm_api.chat_img_async(
            image_base64,
            "block_detect_prompt",
            debug=True,
            schema=TypeAdapter(list[DetectedFromLLM]).json_schema(),
        )
        if task is None:
            continue
        result, _ = llm_api.await_task(task, blocking=True)
        print(result)

    # 绘制结果
    try:
        boxes: list[DetectedFromLLM] = TypeAdapter(list[DetectedFromLLM]).validate_json(
            extract_json_from_markdown(result)
        )
    except Exception as e:
        print(f"解析/校验 JSON 失败: {e}")
        exit(1)
    print(f"Detected {len(boxes)} boxes.")
    img = Image.open(image_path)
    width, height = img.size
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)
    for box in boxes:
        box = box.to_detected_box(img_w=width, img_h=height)
        label = box.class_name
        x_center, y_center = box.box_center_x, box.box_center_y
        width_box, height_box = box.box_width, box.box_height
        draw = ImageDraw.Draw(img)
        x1 = int((x_center - width_box / 2))
        y1 = int((y_center - height_box / 2))
        x2 = int((x_center + width_box / 2))
        y2 = int((y_center + height_box / 2))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 20), label, fill="red", font=font)

    cv2.imshow("result", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
