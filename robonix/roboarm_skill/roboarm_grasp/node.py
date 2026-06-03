"""
Roboarm Grasp Skill Node

Wraps the existing YOLO + LLM detection + robotic arm grasping pipeline
as a Robonix Skill with MCP tools.

Tools:
  - detect_and_grasp(instruction) — LLM-based detection + grasp (catch_by_llm.py)
  - classify_and_grasp(repeat)     — YOLO classification + grasp (classification/catch_with_arm.py)
  - move_home()                    — return arm to home position
"""

import json
import os
import sys
import threading
import time
import traceback

# -- Add roboarm project to Python path --
_ROBOARM_PATH = os.environ.get("ROBOARM_PATH", "/home/xjy/roboarm")
if _ROBOARM_PATH not in sys.path:
    sys.path.insert(0, _ROBOARM_PATH)

from mcp.server.fastmcp import FastMCP
from robonix_api import Skill

# ---------------------------------------------------------------------------
# Skill provider
# ---------------------------------------------------------------------------

_NAMESPACE = "robonix/skill/roboarm_grasp"

skill = Skill(
    id="roboarm_grasp",
    namespace=_NAMESPACE,
)

# ---------------------------------------------------------------------------
# FastMCP app + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("roboarm-grasp")

# Lazy-initialized globals (set on first tool call so bootstrap is fast)
_arm = None
_camera = None
_yolo_models = None
_init_lock = threading.Lock()


def _ensure_hardware():
    """Late-init arm, camera, and YOLO models on first use."""
    global _arm, _camera, _yolo_models
    if _arm is not None:
        return
    with _init_lock:
        if _arm is not None:
            return
        from arm.arm_base import Arm
        from camera.camera_api import Camera
        from object_detect.detect import load_model
        from utils.config_getter import get_config_value

        _arm = Arm()
        _camera = Camera(color=True, depth=False)

        # Load YOLO models for classification
        model_paths = get_config_value("classification_YOLO_model_path", [])
        model_paths = [
            os.path.join(_ROBOARM_PATH, p.lstrip("/").lstrip("\\")) for p in model_paths
        ]
        print(f"[roboarm_skill] Loading YOLO models: {model_paths}", flush=True)
        _yolo_models = [load_model(p) for p in model_paths]
        print("[roboarm_skill] hardware + YOLO models initialized", flush=True)
        print(f"[roboarm_skill] {model_paths}", flush=True)


# ---------------------------------------------------------------------------
# Tool: detect_and_grasp  (LLM-based, from llm/catch_by_llm.py)
# ---------------------------------------------------------------------------


@mcp.tool()
async def detect_and_grasp(instruction: str) -> str:
    """使用大模型(LLM)视觉识别并抓取指定物体。

    根据自然语言指令，通过摄像头拍照并调用大模型进行目标检测，
    然后控制机械臂执行抓取和分类放置。

    Args:
        instruction: 自然语言抓取指令，如"抓取红色积木"、"抓取最左边的蓝色方块"
    """
    try:
        _ensure_hardware()
        from queue import Queue
        from llm.catch_by_llm import catch_by_instruction

        frame_data = _camera.get_frames()
        frame = frame_data.get("color", None)
        if frame is None:
            return json.dumps(
                {"status": "error", "reason": "无法获取摄像头画面"}, ensure_ascii=False
            )

        result = catch_by_instruction(frame, instruction, Queue(), arm=_arm)
        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        traceback.print_exc()
        return json.dumps(
            {"status": "error", "error": str(exc), "instruction": instruction},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool: classify_and_grasp  (YOLO-based, from classification/catch_with_arm.py)
# ---------------------------------------------------------------------------


@mcp.tool()
async def classify_and_grasp(repeat: int = 1) -> str:
    """使用YOLO模型识别画面中所有物体，分类并逐一抓取放置。

    拍照后运行YOLO目标检测，识别画面中所有已知物体，
    按配置的类别放置位置执行抓取并分类放置。

    Args:
        repeat: 重复抓取轮数，每轮抓取画面上所有已识别物体，默认1次
    """
    try:
        _ensure_hardware()
        from classification.catch_with_arm import classify_and_grasp_objects
        from utils.config_getter import get_config_value

        if not _yolo_models:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "YOLO模型未加载，检查classification_YOLO_model_path配置",
                },
                ensure_ascii=False,
            )

        default_conf_thres = get_config_value("default_conf_thres")
        default_gripper_aside_pos = get_config_value(
            "default_gripper_aside_pos", raise_if_missing=False
        )

        total_grasped = 0
        results_by_round = []

        for round_idx in range(max(1, repeat)):
            if default_gripper_aside_pos is not None:
                _arm.move_to(default_gripper_aside_pos)
                time.sleep(0.3)

            frame_data = _camera.get_frames()
            frame = frame_data.get("color", None)
            if frame is None:
                return json.dumps(
                    {"status": "error", "reason": "无法获取摄像头画面"},
                    ensure_ascii=False,
                )

            results = classify_and_grasp_objects(
                _arm, frame, _yolo_models, default_conf_thres
            )

            round_detected = []
            round_grasped = 0
            for (_u, _v, _w, _h, _r), score, _class_id, class_name, grasp_success in results:
                entry = {"class": class_name, "score": float(score)}
                if grasp_success is None:
                    entry["skipped"] = True
                    entry["reason"] = "too close to place position"
                else:
                    entry["grasp_success"] = grasp_success
                    if grasp_success:
                        round_grasped += 1
                        total_grasped += 1
                round_detected.append(entry)

            print(
                f"[roboarm_skill] classify round {round_idx + 1}: "
                f"{len(results)} objects, {round_grasped} grasped",
                flush=True,
            )

            results_by_round.append(
                {
                    "round": round_idx + 1,
                    "detected": len(results),
                    "grasped": round_grasped,
                    "objects": round_detected,
                }
            )

        return json.dumps(
            {
                "status": "success",
                "method": "yolo",
                "total_rounds": max(1, repeat),
                "total_grasped": total_grasped,
                "rounds": results_by_round,
            },
            ensure_ascii=False,
        )

    except Exception as exc:
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool: move_home
# ---------------------------------------------------------------------------


@mcp.tool()
async def move_home() -> str:
    """控制机械臂回到初始安全位置（张开夹爪）。"""
    try:
        _ensure_hardware()
        _arm.move_to_home(gripper_open_0to1=1)
        return json.dumps(
            {"status": "success", "action": "move_home"}, ensure_ascii=False
        )
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool metadata for capability declarations
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "detect_and_grasp",
        "description": "使用大模型(LLM)视觉识别并抓取指定物体。"
        "输入自然语言指令（如'抓取红色积木'），"
        "通过摄像头+LLM检测目标，控制机械臂完成抓取和分类放置。",
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "抓取指令，例如：'抓取红色积木'、'抓取最左边的蓝色方块'",
                }
            },
            "required": ["instruction"],
        },
    },
    {
        "name": "classify_and_grasp",
        "description": "使用YOLO模型识别画面中所有物体，分类并逐一抓取放置。"
        "适合已知类别物体的批量分类抓取场景，不依赖大模型。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repeat": {
                    "type": "integer",
                    "description": "重复抓取轮数，默认1次",
                }
            },
            "required": [],
        },
    },
    {
        "name": "move_home",
        "description": "控制机械臂回到初始安全位置（张开夹爪）。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. Attach MCP app to skill (must happen before bootstrap)
    skill.use_mcp_app(mcp)

    # 2. Bootstrap: register with Atlas, start gRPC + MCP servers, heartbeat
    print("[roboarm_skill] bootstrapping...", flush=True)
    skill.bootstrap()
    print(
        f"[roboarm_skill] MCP server listening on port {skill._mcp_port}",
        flush=True,
    )

    # 3. Manually declare MCP capabilities
    endpoint = skill.mcp_endpoint
    for tool in _TOOLS:
        try:
            skill.declare_mcp(
                contract_id=f"{_NAMESPACE}/{tool['name']}",
                endpoint=endpoint,
                input_schema_json=json.dumps(tool["input_schema"]),
                description=tool["description"],
            )
            print(
                f"[roboarm_skill] declared MCP capability: {tool['name']}", flush=True
            )
        except Exception as exc:
            print(f"[roboarm_skill] declare {tool['name']} failed: {exc}", flush=True)

    # 4. Wait for termination
    print("[roboarm_skill] ready", flush=True)
    import signal

    stop = threading.Event()

    def _on_signal(signum, frame):
        print(f"[roboarm_skill] received signal {signum}, shutting down", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    stop.wait()


if __name__ == "__main__":
    main()
