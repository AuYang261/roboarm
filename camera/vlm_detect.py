import os
from openai import OpenAI
import base64
import requests
import json
import random
import time
from PIL import Image, ImageDraw, ImageFont

image_path = "pic/chess.png"
json_output_path = "detection_output.json"
txt_output_path = "detection_output.txt"

def test1():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("openrouter_api_key"),
    )
    start_time = time.time()
    completion = client.chat.completions.create(
        # extra_headers={
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        # },
        extra_body={},
        # model="qwen/qwen2.5-vl-32b-instruct:free",
        model="qwen/qwen2.5-vl-72b-instruct",
        messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this  and image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
                    }
                ]
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(completion.choices[0].message.content)

prompt_object_detection_cn = """
你是图像物体检测的专家。给定一张图片，识别图像中存在的物体，并以结构化的JSON格式提供它们的详细信息。
请分析图片并返回一个JSON数组，其中每个元素包含以下字段，表示每个检测到的物体：
- "id"：检测到的物体的唯一标识符。
- "label"：检测到的物体的名称（例如：“树干”、“树叶”）。
- "bounding_box"：包含检测到的物体周围边界框坐标的对象，具有以下字段：
  - "x_mid":物体中心点x坐标
  - "y_mid"物体中心点y坐标。
- "nearby:": 一个包含与检测到的物体接近的物体标签的列表。
仅提供JSON数组作为输出，不要添加任何额外的文本或解释。
当前的常见物体标签包括但不限于：棋盘，兵，象，王，后等。
"""

def test2():
    # kimi-latest-128k
    client = OpenAI(
        api_key = os.getenv("KIMI_API_KEY"), 
        base_url = "https://api.moonshot.cn/v1",
    )
    
    # 对图片进行base64编码
    image_path = "pic/chess.png"
    with open(image_path, 'rb') as f:
        img_base = base64.b64encode(f.read()).decode('utf-8')
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="moonshot-v1-8k-vision-preview", 
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_object_detection_cn
                    }
                ]
            }
        ]
    )
    # print(response.choices[0].message.content)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(response.choices[0].message.content)
    
    json_content = response.choices[0].message.content
    try:
        detections = json.loads(json_content)
        # print("JSON 格式正确")
    except json.JSONDecodeError as e:
        
        # 尝试修正 JSON 格式错误（根据具体错误进行调整）
        print("json_content:", json_content)
        
        print("JSON 格式错误:", e)
        
        # 删除 json 最后一个表项
        last_comma_index = json_content.rfind(',')
        if last_comma_index != -1:
            json_content = json_content[:last_comma_index] + json_content[last_comma_index + 1:]
        
    
    # 将 content 保存为 JSON 文件
    try:
        json_output_path = "detection_output_kimi.json"
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json_file.write(response.choices[0].message.content)
    except Exception as e:
        print("保存 JSON 文件时出错:", e)
        # 把偶难道 txt 文件保存下来
        txt_output_path = "detection_output_kimi.txt"
        with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(response.choices[0].message.content)

if __name__ == "__main__":
    test2()