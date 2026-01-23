import asyncio
from math import inf
import sys
import os
import shutil
from pydantic import TypeAdapter
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from dataclasses import dataclass
from llm.llm_detect import LLMDetect, draw_boxes_on_frame, json2box, DetectedFromLLM
from llm.audio2text import audio_file2text
from llm.dataclass import DetectedBox
import cv2
import numpy as np
import json


@dataclass
class BoxPoints:
    point1: tuple[float, float]
    point2: tuple[float, float]
    point3: tuple[float, float]
    point4: tuple[float, float]
    label: str

    def __init__(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        point3: tuple[float, float],
        point4: tuple[float, float],
        label: str,
    ):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4
        self.label = label


def json_from_labelme2box(
    labelme_json: dict,
) -> list[BoxPoints]:
    boxes: list[BoxPoints] = []
    for shape in labelme_json.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        if len(points) != 4 or len(label) == 0:
            continue
        # 用cv2找能包含四个点的最小旋转矩形
        # pts = cv2.boxPoints(cv2.minAreaRect(np.array(points, dtype=np.float32)))
        # center_x = float(np.mean(pts[:, 0]))
        # center_y = float(np.mean(pts[:, 1]))
        # width = float(np.linalg.norm(pts[0] - pts[1]))  # 取相邻两点的距离作为宽度
        # height = float(np.linalg.norm(pts[1] - pts[2]))  # 取相邻两点的距离作为高度
        # angle = cv2.minAreaRect(np.array(points, dtype=np.float32))[2]
        # # cv2.minAreaRect返回的角度范围是[-90, 0)，需要转换为[0, 180)
        # if width < height:
        #     angle = angle + 90.0
        # boxes.append(
        #     DetectedBox(
        #         class_name=label,
        #         box_center_x=center_x,
        #         box_center_y=center_y,
        #         box_width=width,
        #         box_height=height,
        #         box_rotation_deg=angle,
        #     )
        # )
        boxes.append(BoxPoints(points[0], points[1], points[2], points[3], label))
    return boxes


async def testcase_pic_llm_detect(
    pic_path: Path,
    labelme_json: dict,
    boxes: list[BoxPoints],
    instruct_audio: Path,
) -> tuple[float, str, DetectedBox | None]:
    """
    测试一张图片和对应的labelme标注文件，将真实box与LLM检测结果对比
    返回(交并比IOU,识别的指令,识别的框)，若有多个box则返回最高IOU
    IOU为-inf表示LLM检测失败，不计入统计
    """
    pic = cv2.imread(pic_path.as_posix())
    llm_detect = LLMDetect()
    instruction = await asyncio.to_thread(audio_file2text, instruct_audio.as_posix())
    # 旋转180度以适应摄像头安装方向，需要根据实际安装情况调整
    pic_rotated = cv2.rotate(pic, cv2.ROTATE_180)
    response_task = llm_detect.detect_frame(
        pic_rotated,
        prompt_key="user_instruction_prompt",
        replace_map={"{user_instruction}": instruction},
        schema=TypeAdapter(DetectedFromLLM).json_schema(),
    )
    if response_task is None:
        return -inf, "", None
    # 把“阻塞等待”放到线程里，避免卡住 event loop
    response, _ = await asyncio.to_thread(
        llm_detect.llm_api.await_task, response_task, True
    )
    if response is None:
        return -inf, "", None
    box_llm = json2box(response, img_w=pic_rotated.shape[1], img_h=pic_rotated.shape[0])
    if not box_llm:
        return 0.0, "", None
    iou_list = []
    for points in boxes:
        # 计算IOU，points为任意四边形，box_llm为矩形
        # 计算交集
        pts_true = np.array(
            [points.point1, points.point2, points.point3, points.point4],
            dtype=np.float32,
        )
        rect_llm = (
            box_llm.box_center_x - box_llm.box_width / 2,
            box_llm.box_center_y - box_llm.box_height / 2,
            box_llm.box_center_x + box_llm.box_width / 2,
            box_llm.box_center_y + box_llm.box_height / 2,
        )
        iou = calculate_iou(pts_true, rect_llm)
        iou_list.append(iou)
    # 将框画在图片上以便调试
    pic_with_boxes = draw_boxes_on_frame([box_llm], pic)
    # 绘制真实框
    for points in boxes:
        pts = np.array(
            [points.point1, points.point2, points.point3, points.point4],
            dtype=np.int32,
        )
        cv2.polylines(
            pic_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2
        )
        cv2.putText(
            pic_with_boxes,
            points.label,
            (int(pts[0][0]), int(pts[0][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    # 保存结果图片，含中文所以不能用cv2.imwrite
    out_path = (
        pic_path.parent
        / "debug"
        / pic_path.name.replace(".png", f"{instruct_audio.name}.png")
    )

    ext = out_path.suffix if out_path.suffix else ".png"
    ok, buf = cv2.imencode(ext, pic_with_boxes)
    if not ok:
        raise RuntimeError(f"cv2.imencode 失败: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf.tobytes())
    return max(iou_list) if iou_list else -inf, instruction, box_llm


def calculate_iou(
    pts_true: np.ndarray, rect_llm: tuple[float, float, float, float]
) -> float:
    """计算任意四边形与矩形的IOU"""
    # 计算四边形的边界框
    x_min_true = np.min(pts_true[:, 0])
    x_max_true = np.max(pts_true[:, 0])
    y_min_true = np.min(pts_true[:, 1])
    y_max_true = np.max(pts_true[:, 1])

    x_min_llm, y_min_llm, x_max_llm, y_max_llm = rect_llm

    # 计算交集边界框
    x_min_inter = max(x_min_true, x_min_llm)
    y_min_inter = max(y_min_true, y_min_llm)
    x_max_inter = min(x_max_true, x_max_llm)
    y_max_inter = min(y_max_true, y_max_llm)

    # 计算交集面积
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # 计算各自面积
    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    area_llm = (x_max_llm - x_min_llm) * (y_max_llm - y_min_llm)

    # 计算并集面积
    union_area = area_true + area_llm - inter_area

    # 计算IOU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def get_blue_block_instruction(
    points_list: list[BoxPoints],
) -> list[BoxPoints]:
    """从标注中找到所有蓝色积木的坐标"""
    results = []
    for points in points_list:
        if points.label == "blue_block":
            results.append(points)
    return results


def get_nearest_block_instruction(
    points_list: list[BoxPoints],
) -> list[BoxPoints]:
    """从标注中找到最近的积木的坐标，假设最近的是y坐标最小的"""
    if not points_list:
        return []
    # 求重心y坐标最小的积木
    nearest_points = min(
        points_list,
        key=lambda p: np.mean(
            np.array([p.point1, p.point2, p.point3, p.point4], dtype=np.float32)[:, 1]
        ),
    )
    return [nearest_points]


async def main():
    pic_dataset_path = Path(__file__).parent / Path(r"dataset") / Path(r"pic")
    debug_dir = pic_dataset_path / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    audio_dataset_path = Path(__file__).parent / Path(r"dataset")
    # 每条语音指令对应的处理函数
    audio2func = {
        audio_dataset_path / "抓取蓝色积木.m4a": get_blue_block_instruction,
        audio_dataset_path / "抓取最近的积木.m4a": get_nearest_block_instruction,
    }
    tasks = []
    for item in os.listdir(pic_dataset_path):
        if item.endswith(".json"):
            continue
        pic_path = pic_dataset_path / item
        if pic_path.is_dir():
            continue
        labelme_path = pic_path.with_suffix(".json")
        if not labelme_path.exists():
            print(f"缺少标注文件: {labelme_path}")
            continue
        with open(labelme_path.as_posix(), "r", encoding="utf-8") as f:
            labelme_json = json.load(f)
        points_list = json_from_labelme2box(labelme_json)
        for audio_path, func in audio2func.items():
            instruct_points = func(points_list)
            if not instruct_points:
                print(
                    f"图片: {pic_path.name}, 指令: {audio_path.name}, 未找到对应的目标坐标，跳过"
                )
                continue

            async def run_one(
                pic_path: Path,
                audio_path: Path,
                instruct_points: list[BoxPoints],
                labelme_json: dict,
            ):
                iou, instruction, box_llm = await testcase_pic_llm_detect(
                    pic_path=pic_path,
                    labelme_json=labelme_json,
                    boxes=instruct_points,
                    instruct_audio=audio_path,
                )
                return pic_path, audio_path, instruct_points, iou, instruction, box_llm

            tasks.append(
                run_one(
                    pic_path=pic_path,
                    audio_path=audio_path,
                    instruct_points=instruct_points,
                    labelme_json=labelme_json,
                )
            )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ious = []
    for r in results:
        if isinstance(r, BaseException):
            print(f"任务失败: {r}")
            continue
        pic_path, audio_path, instruct_points, iou, instruction, box_llm = r
        if iou == -inf:
            print(
                f"图片: {pic_path.name}, 指令: {audio_path.name}, 目标坐标: {instruct_points}, 识别的指令: {instruction}, LLM检测失败，跳过"
            )
            continue
        print(
            f"图片: {pic_path.name}, 指令: {audio_path.name}, 目标坐标: {instruct_points}, 识别的指令: {instruction}, 识别的框: {box_llm}, IOU: {iou:.4f}"
        )
        ious.append(iou)
    if ious:
        print("IOU 统计信息:")
        pr("mean", float(np.mean(ious)))
        pr("min", float(np.min(ious)))
        pr("25%", float(np.percentile(ious, 25)))
        pr("median", float(np.median(ious)))
        pr("75%", float(np.percentile(ious, 75)))
        pr("99%", float(np.percentile(ious, 99)))
        pr("max", float(np.max(ious)))
        pr("std", float(np.std(ious)))

        iou_threshold = 0.5
        ap50 = sum(1 for iou in ious if iou >= iou_threshold) / len(ious)
        pr("mAP50", ap50)

        # mAP50-95
        iou_ranges = np.arange(0.5, 1.0, 0.05)
        ap_list = []
        for iou_threshold in iou_ranges:
            ap = sum(1 for iou in ious if iou >= iou_threshold) / len(ious)
            ap_list.append(ap)
        map50_95 = float(np.mean(ap_list))
        pr("mAP50-95", map50_95)


def pr(label: str, value: float):
    print(f"{label:>10}: {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
