import os
import sys
import json
from datetime import datetime
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera


"""
    使用camera/re_position.py的轮廓提取代码，得到如下输出：
    Mask 区域
        Point coordinates: (334, 218)
        Point coordinates: (331, 277)
        Point coordinates: (360, 343)
        Point coordinates: (360, 399)
        Point coordinates: (398, 463)
        Point coordinates: (617, 460)
        Point coordinates: (477, 243)
        Point coordinates: (439, 243)
        Point coordinates: (390, 209)
        Point coordinates: (329, 219)
"""
_ARM_MASK_POINTS = np.array(
    [
        [334, 218],
        [331, 277],
        [360, 343],
        [360, 399],
        [398, 463],
        [617, 460],
        [477, 243],
        [439, 243],
        [390, 209],
        [329, 219],
    ],
    dtype=np.int32,
)

def _resolve_path(path: str, must_exist: bool = True) -> str | None:
    if path is None:
        return None

    candidates = [
        path,
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path),
    ]

    checked = []
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        if os.path.exists(candidate):
            return candidate

    if must_exist:
        raise FileNotFoundError(f"文件不存在: {path}")
    return None

def _normalize_xyaxes(xyaxes: np.ndarray) -> np.ndarray:
    xyaxes = np.asarray(xyaxes, dtype=np.float64).reshape(6)
    x_axis = xyaxes[:3]
    y_axis = xyaxes[3:]

    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    if np.linalg.norm(y_axis) < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(fallback, x_axis)) > 0.95:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = fallback - np.dot(fallback, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)

    return np.concatenate([x_axis, y_axis])

def _xyaxes_to_rotation(xyaxes: np.ndarray) -> np.ndarray:
    xyaxes = _normalize_xyaxes(xyaxes)
    x_axis = xyaxes[:3]
    y_axis = xyaxes[3:]
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])

def _rotation_to_quat(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = np.trace(rotation)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / quat_norm

def _parse_camera_params(xml_path: str, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for cam in root.iter("camera"):
        if cam.get("name") == camera_name:
            pos = np.fromstring(cam.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
            xyaxes = np.fromstring(cam.get("xyaxes", "1 0 0 0 1 0"), sep=" ", dtype=np.float64)
            if pos.size != 3:
                raise ValueError(f"camera {camera_name} 的 pos 配置不合法")
            if xyaxes.size != 6:
                raise ValueError(f"camera {camera_name} 的 xyaxes 配置不合法")
            return pos, _normalize_xyaxes(xyaxes)
    raise ValueError(f"未在 {xml_path} 中找到相机 {camera_name}")

def _write_camera_params(xml_path: str, camera_name: str, pos: np.ndarray, xyaxes: np.ndarray) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for cam in root.iter("camera"):
        if cam.get("name") == camera_name:
            cam.set("pos", " ".join(f"{v:.6f}" for v in np.asarray(pos, dtype=np.float64).tolist()))
            cam.set(
                "xyaxes",
                " ".join(f"{v:.6f}" for v in _normalize_xyaxes(xyaxes).tolist()),
            )
            tree.write(xml_path, encoding="utf-8", xml_declaration=False)
            return
    raise ValueError(f"未在 {xml_path} 中找到相机 {camera_name}")

def _load_real_frame(base_img_path: str | None, camera: Camera | None) -> np.ndarray:
    if base_img_path is not None:
        frame = cv2.imread(base_img_path)
        if frame is None:
            raise ValueError(f"无法读取图像: {base_img_path}")
        return frame

    if camera is None:
        raise RuntimeError("未提供历史图像，也未初始化真实相机")

    for _ in range(30):
        frames = camera.get_frames()
        frame = frames.get("color") if isinstance(frames, dict) else None
        if frame is not None:
            return frame
    raise RuntimeError("无法从真实相机获取彩色图像帧")

def _extract_arm_contour(frame: np.ndarray, mask_points: np.ndarray | None = None) -> dict:
    work_frame = frame.copy()
    if mask_points is not None and len(mask_points) >= 3:
        mask = np.zeros(work_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points.astype(np.int32)], 255)
        work_frame = cv2.bitwise_and(work_frame, work_frame, mask=mask)

    gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edged,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edged, (x1, y1), (x2, y2), 0, 3)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "contour": None,
            "center": None,
            "area": 0.0,
            "mask": np.zeros_like(gray),
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))

    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        center = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    else:
        center = None

    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    return {
        "contour": contour,
        "center": center,
        "area": area,
        "mask": contour_mask,
    }

def _score_contours(real_info: dict, sim_info: dict) -> tuple[float, dict]:
    if real_info["contour"] is None or sim_info["contour"] is None:
        return float("inf"), {
            "iou": 0.0,
            "center_distance": float("inf"),
            "center_score": float("inf"),
            "area_score": float("inf"),
            "shape_score": float("inf"),
        }

    if real_info["center"] is None or sim_info["center"] is None:
        return float("inf"), {
            "iou": 0.0,
            "center_distance": float("inf"),
            "center_score": float("inf"),
            "area_score": float("inf"),
            "shape_score": float("inf"),
        }

    real_mask = real_info["mask"] > 0
    sim_mask = sim_info["mask"] > 0
    union = np.logical_or(real_mask, sim_mask).sum()
    inter = np.logical_and(real_mask, sim_mask).sum()
    iou = float(inter / union) if union > 0 else 0.0

    height, width = real_info["mask"].shape[:2]
    diag = max(float(np.hypot(width, height)), 1.0)
    center_distance = float(
        np.linalg.norm(np.asarray(real_info["center"]) - np.asarray(sim_info["center"]))
    )
    center_score = center_distance / diag
    area_score = abs(sim_info["area"] - real_info["area"]) / max(real_info["area"], 1.0)
    shape_score = float(
        cv2.matchShapes(
            real_info["contour"],
            sim_info["contour"],
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
    )

    score = 2.0 * (1.0 - iou) + 1.2 * center_score + 0.8 * area_score + 0.2 * shape_score
    return score, {
        "iou": iou,
        "center_distance": center_distance,
        "center_score": center_score,
        "area_score": float(area_score),
        "shape_score": shape_score,
        "real_center": [float(real_info["center"][0]), float(real_info["center"][1])],
        "sim_center": [float(sim_info["center"][0]), float(sim_info["center"][1])],
        "real_area": float(real_info["area"]),
        "sim_area": float(sim_info["area"]),
    }

def _render_camera_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_id: int,
) -> np.ndarray:
    render_camera = mujoco.MjvCamera()
    render_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    render_camera.fixedcamid = camera_id
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, render_camera)
    image = renderer.render()
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return img


def _draw_comparison(
    real_frame: np.ndarray,
    sim_frame: np.ndarray,
    real_info: dict,
    sim_info: dict,
    score: float,
    metrics: dict,
    label: str = "",
) -> np.ndarray:
    """绘制真实/模拟对比图：左侧真实帧+轮廓，右侧模拟帧+轮廓，底部评分信息"""
    real_vis = real_frame.copy()
    sim_vis = sim_frame.copy()

    # 在真实帧上绘制轮廓（绿色）和中心点
    if real_info["contour"] is not None:
        cv2.drawContours(real_vis, [real_info["contour"]], -1, (0, 255, 0), 2)
    if real_info["center"] is not None:
        cx, cy = int(real_info["center"][0]), int(real_info["center"][1])
        cv2.circle(real_vis, (cx, cy), 6, (0, 255, 0), -1)

    # 在模拟帧上绘制轮廓（蓝色）和中心点
    if sim_info["contour"] is not None:
        cv2.drawContours(sim_vis, [sim_info["contour"]], -1, (255, 100, 0), 2)
    if sim_info["center"] is not None:
        cx, cy = int(sim_info["center"][0]), int(sim_info["center"][1])
        cv2.circle(sim_vis, (cx, cy), 6, (255, 100, 0), -1)

    # 添加标题
    cv2.putText(real_vis, "Real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(sim_vis, "Sim", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

    # 拼接左右
    h1 = real_vis.shape[0]
    h2 = sim_vis.shape[0]
    max_h = max(h1, h2)
    if h1 < max_h:
        real_vis = cv2.copyMakeBorder(real_vis, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if h2 < max_h:
        sim_vis = cv2.copyMakeBorder(sim_vis, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    combined = np.hstack([real_vis, sim_vis])

    # 底部评分信息面板
    panel_h = 80
    total_w = combined.shape[1]
    panel = np.zeros((panel_h, total_w, 3), dtype=np.uint8)

    iou = metrics.get("iou", 0.0)
    center_dist = metrics.get("center_distance", float("inf"))
    area_score = metrics.get("area_score", float("inf"))
    shape_score = metrics.get("shape_score", float("inf"))

    line1 = f"{label}  score={score:.4f}  IoU={iou:.4f}  center_dist={center_dist:.2f}px"
    line2 = f"area_score={area_score:.4f}  shape_score={shape_score:.4f}"
    if "real_area" in metrics and "sim_area" in metrics:
        line2 += f"  real_area={metrics['real_area']:.0f}  sim_area={metrics['sim_area']:.0f}"

    cv2.putText(panel, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(panel, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return np.vstack([combined, panel])


def _contour_to_points(contour) -> list[list[int]]:
    if contour is None:
        return []
    return contour.reshape(-1, 2).astype(int).tolist()

# 臂上相机 - 简单使用像素坐标点大概估计位置信息 - 微调 urdf\meshes\mjmodel.xml 中 L60 的 gripper_cam 参数 pos="0 0.08 0" xyaxes="1 0 0 0 0.8 -0.6"
def simple_opt_arm_camera(xml_path:str = 'urdf/meshes/mjmodel_opt.xml', camera_name: str = 'gripper_cam', base_img : str = 'koch_img/koch1.jpg'):
    xml_path = _resolve_path(xml_path)
    base_img_path = _resolve_path(base_img, must_exist=False)

    camera = None
    renderer = None

    # 创建模拟场景
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"未在模型中找到相机 {camera_name}")

    try:
        # 获取相机视频流
        if base_img_path is None:
            camera = Camera(color=True, depth=False)

        # xml文件解析 - 摄像机参数获取
        init_pos, init_xyaxes = _parse_camera_params(xml_path, camera_name)

        # 臂上真实视频流加载，获取图像帧 / 加载历史保存图像 加载 koch_img/koch.jpg
        real_frame = _load_real_frame(base_img_path, camera)
        height, width = real_frame.shape[:2]
        renderer = mujoco.Renderer(model, height=height, width=width)
        output_root = os.path.join(os.path.dirname(__file__), "camera_simple_opt_output")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_root, f"{camera_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "real_frame.png"), real_frame)

        # 机械臂轮廓提取 + Mask
        
        mask_points = _ARM_MASK_POINTS.copy()

        # 提取轮廓
        real_info = _extract_arm_contour(real_frame, mask_points)
        if real_info["contour"] is None:
            raise RuntimeError("真实图像中未提取到机械臂轮廓，请重新检查 mask 点或输入图像")

        # 评估 参数效果
        def evaluate_camera_params(pos: np.ndarray, xyaxes: np.ndarray) -> tuple[float, dict, np.ndarray, np.ndarray, dict]:
            # 臂上模拟视频流加载-更新参数
            candidate_pos = np.asarray(pos, dtype=np.float64).reshape(3)
            candidate_xyaxes = _normalize_xyaxes(xyaxes)
            model.cam_pos[camera_id] = candidate_pos
            model.cam_quat[camera_id] = _rotation_to_quat(_xyaxes_to_rotation(candidate_xyaxes))

            # 从模拟中渲染当前相机视角的图像帧
            sim_frame = _render_camera_frame(model, data, renderer, camera_id)

            # 机械臂轮廓提取 + Mask
            sim_info = _extract_arm_contour(sim_frame, mask_points)

            # 轮廓对比 - 与真实场景对比
            score, metrics = _score_contours(real_info, sim_info)

            # 计算得分
            return score, metrics, candidate_xyaxes, sim_frame, sim_info

        best_pos = init_pos.copy()
        best_xyaxes = init_xyaxes.copy()
        initial_score, initial_metrics, best_xyaxes, initial_sim_frame, initial_sim_info = evaluate_camera_params(best_pos, best_xyaxes)
        best_score = initial_score
        best_metrics = initial_metrics

        initial_compare = _draw_comparison(
            real_frame,
            initial_sim_frame,
            real_info,
            initial_sim_info,
            initial_score,
            initial_metrics,
            label="initial",
        )
        cv2.imwrite(os.path.join(output_dir, "initial_compare.png"), initial_compare)

        pos_step = np.array([0.01, 0.02, 0.01], dtype=np.float64)
        axis_step = 0.08

        # 循环处理 ：修改参数 - 评估 - 更新参数
        for _ in range(12):
            improved = False
            candidates: list[tuple[np.ndarray, np.ndarray]] = []

            for idx in range(3):
                for direction in (-1.0, 1.0):
                    candidate_pos = best_pos.copy()
                    candidate_pos[idx] += direction * pos_step[idx]
                    candidates.append((candidate_pos, best_xyaxes.copy()))

            for idx in range(6):
                for direction in (-1.0, 1.0):
                    candidate_xyaxes = best_xyaxes.copy()
                    candidate_xyaxes[idx] += direction * axis_step
                    candidates.append((best_pos.copy(), candidate_xyaxes))

            for candidate_pos, candidate_xyaxes in candidates:
                score, metrics, normalized_xyaxes, sim_frame, sim_info = evaluate_camera_params(candidate_pos, candidate_xyaxes)

                # 保存/打印 每轮对比信息
                compare_img = _draw_comparison(
                    real_frame,
                    sim_frame,
                    real_info,
                    sim_info,
                    score,
                    metrics,
                    label="eval",
                )
                eval_idx = len(
                    [name for name in os.listdir(output_dir) if name.startswith("compare_") and name.endswith(".png")]
                )
                compare_path = os.path.join(output_dir, f"compare_{eval_idx:03d}.png")
                cv2.imwrite(compare_path, compare_img)

                metrics_record = {
                    "label": "eval",
                    "score": float(score),
                    "pos": [float(v) for v in candidate_pos.tolist()],
                    "xyaxes": [float(v) for v in normalized_xyaxes.tolist()],
                    "metrics": metrics,
                    "real_contour": _contour_to_points(real_info["contour"]),
                    "sim_contour": _contour_to_points(sim_info["contour"]),
                    "image_path": compare_path,
                }
                with open(os.path.join(output_dir, f"compare_{eval_idx:03d}.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics_record, f, ensure_ascii=False, indent=2)

                print(f"eval pos=\"{' '.join(f'{v:.6f}' for v in candidate_pos.tolist())}\" "
                      f"xyaxes=\"{' '.join(f'{v:.6f}' for v in candidate_xyaxes.tolist())}\" "
                      f"score={score:.6f}")
                if score + 1e-9 < best_score:
                    best_score = score
                    best_pos = candidate_pos
                    best_xyaxes = normalized_xyaxes
                    best_metrics = metrics
                    improved = True

                    # 保存 best score 的参数
                    _write_camera_params(xml_path, camera_name, best_pos, best_xyaxes)

            if not improved:
                pos_step *= 0.5
                axis_step *= 0.5
                if np.max(pos_step) < 0.001 and axis_step < 0.005:
                    break

        _write_camera_params(xml_path, camera_name, best_pos, best_xyaxes)

        result = {
            "xml_path": xml_path,
            "camera_name": camera_name,
            "output_dir": output_dir,
            "initial_score": float(initial_score),
            "best_score": float(best_score),
            "best_pos": [float(v) for v in best_pos.tolist()],
            "best_xyaxes": [float(v) for v in best_xyaxes.tolist()],
            "metrics": best_metrics,
        }

        print(f"[{camera_name}] initial_score={initial_score:.6f}, best_score={best_score:.6f}")
        print(f"[{camera_name}] output_dir=\"{output_dir}\"")
        print(f"[{camera_name}] pos=\"{' '.join(f'{v:.6f}' for v in best_pos.tolist())}\"")
        print(f"[{camera_name}] xyaxes=\"{' '.join(f'{v:.6f}' for v in best_xyaxes.tolist())}\"")
        
        # 保存 best socre 的图片和对比信息到文件夹中
        core, metrics, normalized_xyaxes, sim_frame, sim_info = evaluate_camera_params(best_pos, best_xyaxes)
        best_compare = _draw_comparison(
            real_frame,
            sim_frame,
            real_info,
            sim_info,
            core,
            metrics,
            label="best",
        )
        cv2.imwrite(os.path.join(output_dir, "best_compare.png"), best_compare)
        
        return result
    finally:
        if renderer is not None:
            renderer.close()
        if camera is not None:
            camera.close()


# TODO 外部相机流程 - 简单使用像素坐标点大概估计位置信息
def simple_opt_external_camera():
    pass

if __name__ == "__main__":
    simple_opt_arm_camera()
