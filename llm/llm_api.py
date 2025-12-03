import os
from openai import OpenAI
import base64
import requests
import json
import random
import time
from PIL import Image, ImageDraw, ImageFont


class LLMAPI:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate_image_description(self, image_url: str, prompt: str) -> str | None:
        start_time = time.time()
        completion = self.client.chat.completions.create(
            model="google/gemini-3-pro-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return completion.choices[0].message.content
