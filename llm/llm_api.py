import json
import os
from turtle import width
from networkx import star_graph
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI, AsyncOpenAI
import base64
import time
import yaml
import toml
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from typing import Any, Coroutine
import asyncio


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

    def chat_img(
        self,
        image_base64: str,
        prompt_key: str,
        model: str = "google/gemini-3-pro-preview",
        debug: bool = False,
        schema_key: str | None = None,
        temperature: float = 0.1,
    ) -> str | None:
        start_time = time.time()
        prompt = self.prompts.get(prompt_key, None)
        if prompt is None:
            print(f"Prompt key '{prompt_key}' not found.")
            return None
        schema = None
        if schema_key is not None:
            schema = self.prompts.get(schema_key, None)
            if schema is None:
                print(f"Warning: Schema key '{schema_key}' not found.")
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
        schema_key: str | None = None,
        temperature: float = 0.1,
    ) -> Coroutine[Any, Any, ChatCompletion] | None:
        start_time = time.time()
        prompt = self.prompts.get(prompt_key, None)
        if prompt is None:
            print(f"Prompt key '{prompt_key}' not found.")
            return None
        schema = None
        if schema_key is not None:
            schema = self.prompts.get(schema_key, None)
            if schema is None:
                print(f"Warning: Schema key '{schema_key}' not found.")
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
        return completion_coroutine

    def await_chat_coroutine(
        self, coroutine: Coroutine[Any, Any, ChatCompletion]
    ) -> str | None:
        start_time = time.time()
        completion = asyncio.run(coroutine)
        print(f"Await time taken: {time.time() - start_time} seconds")
        if completion.choices is None or len(completion.choices) == 0:
            print("No choices returned from the model.")
            return None
        return completion.choices[0].message.content


if __name__ == "__main__":
    llm_api = LLMAPI()

    image_path = "object_detect/dataset_process/dataset/备份-双积木/img_012.png"
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    result = None
    while result is None:
        result_coroutine = llm_api.chat_img_async(
            image_base64,
            "block_detect_prompt",
            debug=True,
            schema_key="block_detect_prompt_json_schema",
        )
        if result_coroutine is None:
            continue
        result = llm_api.await_chat_coroutine(result_coroutine)
        print(result)
    json_content = json.loads(result)
    img = Image.open(image_path)
    width, height = img.size
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)
    for item in json_content:
        box = item.get("pos")
        if box is None or len(box) != 2:
            continue
        label = item.get("label", "unknown")
        x_mid, y_mid = box["x_mid"], box["y_mid"]
        draw = ImageDraw.Draw(img)
        draw.circle((x_mid * width, y_mid * height), radius=5, fill="red")
        draw.text((x_mid * width + 5, y_mid * height - 5), label, fill="red", font=font)
    cv2.imshow("result", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
