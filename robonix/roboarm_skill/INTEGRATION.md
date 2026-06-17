# Roboarm Skill 接入 Robonix 指南

将已有的 YOLO + LLM 机械臂抓取项目接入 Robonix 框架，使其可通过 `rbnx chat` 用自然语言控制。

## 前置条件

- Robonix 已安装（`rbnx` 命令可用）
- roboarm 项目位于 `/home/xjy/roboarm/`，内含 `config.yaml` 和 `.venv`
- 摄像头和机械臂硬件已连接
- 需配置 `VLM_BASE_URL`、`VLM_API_KEY`、`VLM_MODEL` 环境变量

## 架构概览

```
rbnx chat (Pilot, VLM agent)
    ↓ MCP
roboarm_grasp skill node (FastMCP HTTP)
    ↓ Python import
roboarm 项目 (arm_base, camera_api, YOLO, LLM detect)
    ↓
真实硬件 (机械臂 + 摄像头)
```

skill node 是薄层，只负责 Robonix 生命周期管理和结果格式化。所有检测和抓取逻辑由 roboarm 项目提供。

## 第一步：改造 roboarm 项目，提取可复用接口（已完成，非必需，也可以在skill的实现里重新实现逻辑）

原 roboarm 代码中，核心逻辑与 demo 循环混杂。需将核心逻辑提取为独立函数，供外部调用。

### 1.1 改造 `classification/catch_with_arm.py`

提取 `classify_and_grasp_objects()` 函数，封装 YOLO 检测 + 坐标转换 + 放置位置解析 + 抓取执行：

```python
def classify_and_grasp_objects(arm, frame, models, conf_thres: float) -> list[dict]:
    """对一帧图像运行 YOLO 检测，逐个抓取识别到的物体。

    Args:
        arm: Arm() 实例。
        frame: BGR 图像 (numpy array)。
        models: YOLO 模型列表。
        conf_thres: 置信度阈值。

    Returns:
        [(u, v, w, h, r), score, class_id, class_name, grasp_success]
        grasp_success: True=成功, False=失败, None=跳过(距离放置位置太近)
    """
```

`main()` 函数简化为只做显示循环，内部调用 `classify_and_grasp_objects()`。

### 1.2 改造 `llm/catch_by_llm.py`

**延迟初始化 arm**：模块级 `arm = None`，由 `main()` 或外部调用者注入。

`catch_by_text_instruction()` 和 `catch_by_audio()` 内部自行创建 `Arm()` 实例。

**给 `catch_by_instruction()` 添加 `arm` 参数和返回值**：

```python
def catch_by_instruction(
    frame: cv2.typing.MatLike,
    instruction: str,
    queue_output: Queue,
    success_callback: Optional[Callable[[], None]] = None,
    arm=None,                          # 新增：可注入 Arm 实例
):
    ...
    # 返回值示例
    return {
        "status": "success" | "failed",
        "target": box.class_name,
        "instruction": instruction,
        "method": "llm",
        "grasp_success": True | False,
    }
```

## 第二步：创建 skill 包目录结构

在 Robonix 仓库中创建：

```
examples/roboarm_skill/
├── package_manifest.yaml    # 包元数据
├── CAPABILITY.md            # 能力说明文档
├── scripts/
│   ├── build.sh             # 构建脚本（rbnx codegen）
│   └── start.sh             # 启动脚本
└── roboarm_grasp/
    └── node.py              # skill 节点主代码
```

## 第三步：编写 skill 节点 (node.py)

核心要点：

1. **延迟初始化硬件**：Arm、Camera、YOLO 模型在首次工具调用时才加载，加快 bootstrap 速度。

```python
_arm = None
_camera = None
_yolo_models = None
_init_lock = threading.Lock()

def _ensure_hardware():
    global _arm, _camera, _yolo_models
    if _arm is not None:
        return
    with _init_lock:
        if _arm is not None:
            return
        _arm = Arm()
        _camera = Camera(color=True, depth=False)
        _yolo_models = [load_model(p) for p in model_paths]
```

2. **工具函数直接调用 roboarm 接口**，不重复实现逻辑：

```python
@mcp.tool()
async def detect_and_grasp(instruction: str) -> str:
    _ensure_hardware()
    from llm.catch_by_llm import catch_by_instruction

    frame = _camera.get_frames().get("color")
    result = catch_by_instruction(frame, instruction, Queue(), arm=_arm)
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
async def classify_and_grasp(repeat: int = 1) -> str:
    _ensure_hardware()
    from classification.catch_with_arm import classify_and_grasp_objects

    for round_idx in range(max(1, repeat)):
        frame = _camera.get_frames().get("color")
        results = classify_and_grasp_objects(_arm, frame, _yolo_models, conf_thres)
        # 格式化结果...
    return json.dumps(result, ensure_ascii=False)
```

3. **声明 MCP 能力**：在 `main()` 中 bootstrap 后手动 `declare_mcp()`。

## 第四步：编写 package_manifest.yaml

```yaml
manifestVersion: 1
build: bash scripts/build.sh
start: bash scripts/start.sh
package:
  name: com.robonix.skill.roboarm_grasp
  version: 0.1.0
  vendor: robonix
  description: YOLO + LLM detection + robotic arm grasping skill
  license: MulanPSL-2.0
capabilities: []
depends: []
```

## 第五步：编写构建和启动脚本

**build.sh**：运行 `rbnx codegen` 生成 gRPC stub。

```bash
#!/usr/bin/env bash
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
rbnx codegen -p "$PKG"
echo "[roboarm_skill] build done"
```

**start.sh**：使用 roboarm 的 venv Python（含有 camera、YOLO、arm 依赖），并将 robonix-api 加入 PYTHONPATH。

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "[roboarm_skill] starting..."

ROBOARM_PATH="${ROBOARM_PATH:-/home/xjy/roboarm}"
PYTHON="${ROBOARM_PATH}/.venv/bin/python3"

ROBONIX_API="$(rbnx path robonix-api)"
export PYTHONPATH="${ROBONIX_API}:${PYTHONPATH:-}"

exec "$PYTHON" -m roboarm_grasp.node
```

## 第六步：创建部署清单

在 `examples/roboarm_deploy/robonix_manifest.yaml` 中引用 skill 包：

```yaml
manifestVersion: 1
name: roboarm_deploy
env:
  LOG: "INFO"
system:
  atlas:
    listen: 127.0.0.1:50051
  executor:
    listen: 127.0.0.1:50061
  pilot:
    listen: 127.0.0.1:50071
    vlm:
      upstream: ${VLM_BASE_URL}
      api_key: ${VLM_API_KEY}
      model: ${VLM_MODEL}
      api_format: openai
skill:
  - name: roboarm_grasp
    path: ../roboarm_skill
```

## 第七步：构建和启动

```bash
# 设置环境变量（LLM 抓取功能需要）
export VLM_BASE_URL="https://your-llm-api/v1"
export VLM_API_KEY="your-api-key"
export VLM_MODEL="your-model-name"

# 构建 skill
cd examples/roboarm_skill
rbnx clean -p .
rbnx build -p .

# 启动全部服务（atlas + executor + pilot + skill）
cd ../roboarm_deploy
rbnx boot
# 或分别启动各模块，方便调试，以下每行命令单独一个终端
robonix-atlas
rbnx start -p /home/xjy/robonix/examples/roboarm_skill
robonix-executor
robonix-pilot
```

## 第八步：验证

```bash
# 查看注册的节点
rbnx nodes

# 查看可用工具
rbnx tools

# 自然语言控制机械臂
# 单次对话
rbnx ask "执行物品分类任务"
# chat需要有Liaison组件
rbnx chat
```

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| ROBOARM_PATH | /home/xjy/roboarm | roboarm 项目根目录 |
| ROBONIX_ATLAS | 127.0.0.1:50051 | Atlas 控制平面地址 |
| VLM_BASE_URL | - | Pilot 使用的 LLM API 地址 |
| VLM_API_KEY | - | LLM API 密钥 |
| VLM_MODEL | - | LLM 模型名称 |

## 常见问题

**Q: `ModuleNotFoundError: No module named 'mcp'`**

需要安装 MCP 依赖。由于使用 roboarm 的 venv，需在其中安装：
```bash
/home/xjy/roboarm/.venv/bin/python3 -m pip install mcp fastmcp uvicorn
```

**Q: `ModuleNotFoundError: No module named 'robonix_contracts_pb2_grpc'`**

build.sh 未正确执行 codegen。确保 `rbnx codegen -p "$PKG"` 正常运行，且 `grpcio-tools` 已安装：
```bash
/home/xjy/roboarm/.venv/bin/python3 -m pip install grpcio-tools
```

**Q: 机械臂连接失败**

确认 config.yaml 中机械臂配置正确，硬件已上电并连接。

**Q: 摄像头无画面**

确认摄像头已连接，`config.yaml` 中相机配置正确。
