# Being-H05 远程推理方案 2 — UDP 推流 + 远程推理 + 动作回传

> **目标**: 将相机视频流推送到服务器，在服务器端运行 Being-H05 VLA 推理，将动作序列通过 UDP 回传客户端执行。
>
> **传输方式**: 纯 UDP 数据报，无自定义协议头，无分片重组，无控制通道。
>
> **当前实现范围**: 客户端（推流 + 执行）。服务端部分仅定义接口和数据格式。
>
> **网络拓扑**:
> ```
> ┌─────────────────────────────┐      UDP       ┌─────────────────────────────┐
> │  客户端 (192.168.2.5)        │ ◄──────────►   │  服务端 (192.168.2.12)       │
> │                             │                 │                             │
> │  ┌─────────┐  ┌──────────┐  │  Top JPEG :5002 │  ┌──────────┐  ┌──────────┐ │
> │  │ Top Cam  │  │ Wrist Cam│──┼──────────────►┼──│ 帧缓存    │  │ BeingH05 │ │
> │  │ (idx=1)  │  │ (idx=3)  │  │  Wrist JPEG:5003│  │          │  │ 推理引擎  │ │
> │  └─────────┘  └──────────┘  │──────────────►┼──│          │──│          │ │
> │                             │                 │  └──────────┘  └──────────┘ │
> │  ┌──────────────────────┐   │  Action JSON    │                             │
> │  │ Koch Robot (COM6)    │◄──┼─────── :5000 ──┼── 动作序列 (固定格式)        │
> │  │ my_awesome_follower_arm│ │                 │                             │
> │  │ calib: ./my_calibration│ │                 │                             │
> │  └──────────────────────┘   │                 │                             │
> └─────────────────────────────┘                 └─────────────────────────────┘
> ```

---

## 一、整体架构设计

### 1.1 系统分层（极简三层）

```
┌──────────────────────────────────────────────────┐
│                   应用层 (Application)             │
│  client_main.py            server_main.py         │
│  启动/停止/主循环            启动/停止/推理循环      │
├──────────────────────────────────────────────────┤
│                   传输层 (Transport)               │
│  udp_sender.py (发视频)      udp_receiver.py      │
│  udp_receiver.py (收动作)     udp_sender.py        │
│  纯 UDP sendto/recvfrom     纯 UDP sendto/recvfrom│
├──────────────────────────────────────────────────┤
│                   设备层 (Device)                  │
│  camera_streamer.py         inference_engine.py   │
│  robot_executor.py          BeingH05模型封装       │
│  OpenCV采集/JPEG编码         模型加载/推理          │
└──────────────────────────────────────────────────┘
```

> **关键简化**: 去掉了协议层（不再有自定义包头、分片、CRC、序列号、多通道复用）。所有数据以完整 UDP 数据报直接传输。

### 1.2 端口分配

| 方向 | 数据内容 | 发送方端口 | 接收方端口 | 说明 |
|------|---------|-----------|-----------|------|
| 上行 | 顶部相机 JPEG | 客户端任意 | 服务端 `5002` | 一个 UDP 数据报 = 一帧完整 JPEG |
| 上行 | 腕部相机 JPEG | 客户端任意 | 服务端 `5003` | 一个 UDP 数据报 = 一帧完整 JPEG |
| 下行 | 动作序列 JSON | 服务端任意 | 客户端 `5000` | 一个 UDP 数据报 = 一个完整 ActionChunk |

> 每路数据一个独立端口，无需任何头部标记即可区分来源。

### 1.3 任务描述传递

任务描述通过以下方式之一传递（不在传输层处理）：
- 方式 A（推荐）: 服务端启动时通过 `--task` 命令行参数指定
- 方式 B: 客户端和服务端各配置相同的 `--task` 参数

---

## 二、传输数据格式

### 2.1 上行：视频帧（客户端 → 服务端）

**每个 UDP 数据报 = 一个完整的 JPEG 文件字节流，无任何额外头部。**

```
┌──────────────────────────────────────────────┐
│           JPEG File Bytes (完整)              │
│    640×480, quality=80, ~30-60 KB            │
│    单个 UDP 数据报承载，无需分片               │
└──────────────────────────────────────────────┘
```

- 顶部相机帧 → 发送到 `192.168.2.12:5002`
- 腕部相机帧 → 发送到 `192.168.2.12:5003`
- 编码参数: `cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])`
- 单帧大小约 30-60 KB，远小于 UDP 理论最大 65507 字节，局域网无分片风险

### 2.2 下行：动作序列（服务端 → 客户端）

**每个 UDP 数据报 = 一个完整的 JSON 文本，UTF-8 编码，无任何额外头部。**

```json
{
    "sequence_id": 42,
    "chunk_size": 16,
    "inference_time_ms": 85.3,
    "action": {
        "arm_joint_position": [
            [0.15, -0.42, 0.88, -1.23, 0.05],
            [0.16, -0.41, 0.87, -1.22, 0.06],
            [0.17, -0.40, 0.86, -1.21, 0.07],
            "... 共 16 步，每步 5 个关节值"
        ],
        "gripper_position": [
            [0.73],
            [0.72],
            [0.72],
            "... 共 16 步，每步 1 个夹爪值"
        ]
    }
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `sequence_id` | `int` | 推理序列号，递增，用于客户端检测丢包 |
| `chunk_size` | `int` | 动作块长度，固定 16 |
| `inference_time_ms` | `float` | 本次推理耗时（毫秒） |
| `action.arm_joint_position` | `float[16][5]` | 16 步 × 5 关节 (shoulder_pan/lift/elbow_flex/wrist_flex/wrist_roll) |
| `action.gripper_position` | `float[16][1]` | 16 步 × 1 夹爪 |

**关节顺序（与 KochFollower 一致）**:
```
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
                 ↑── arm_joint_position (5) ──↑   ↑── gripper (1) ──↑
```

**值范围**:
- 前 5 个关节: 角度值（度），由 `MotorNormMode.DEGREES` 输出
- 夹爪: 归一化值 [0, 100]，由 `MotorNormMode.RANGE_0_100` 输出

---

## 三、客户端模块设计

### 3.1 模块清单

| 模块 | 文件 | 职责 |
|------|------|------|
| UDP 发送 | `client/udp_sender.py` | 将 JPEG 字节发送到服务端指定端口 |
| UDP 接收 | `client/udp_receiver.py` | 从服务端接收动作 JSON，非阻塞 |
| 相机采集 | `client/camera_streamer.py` | OpenCV 双相机读取 + JPEG 编码 |
| 机械臂控制 | `client/robot_executor.py` | 连接/断开、动作队列、执行、安全检查 |
| 主控制器 | `client/client_main.py` | 组装各模块、主循环、帧率控制 |

### 3.2 客户端主循环流程

```
1. 初始化
   ├─ 创建 UDP socket (接收动作: 0.0.0.0:5000)
   ├─ 连接两台相机 (Top idx=1, Wrist idx=3, 640×480)
   ├─ 连接机械臂 (COM6, my_awesome_follower_arm, ./my_calibration)
   └─ 打印启动信息

2. 主循环 (15Hz)
   ┌─────────────────────────────────────────────────────────┐
   │  loop_start = time.perf_counter()                       │
   │                                                         │
   │  // 采集 + 编码（主线程顺序执行）                           │
   │  ├─ 读取 Top Camera → JPEG 编码                          │
   │  │    → sendto(192.168.2.12:5002, jpeg_top_bytes)       │
   │  └─ 读取 Wrist Camera → JPEG 编码                        │
   │       → sendto(192.168.2.12:5003, jpeg_wrist_bytes)     │
   │                                                         │
   │  // 处理接收（非阻塞）                                     │
   │  ├─ recvfrom() 尝试接收动作 JSON                          │
   │  │   ├─ 有数据 → 解析 → enqueue_action_chunk()           │
   │  │   └─ 无数据 → 跳过                                    │
   │  └─ 检测 sequence_id 跳跃 → 日志告警（丢包检测）          │
   │                                                         │
   │  // 执行动作                                              │
   │  └─ 从动作队列 pop 下一步 → execute_action()              │
   │       → 安全检查 (ensure_safe_goal_position)            │
   │       → DynamixelBus.sync_write("Goal_Position", ...)   │
   │       → 队列空且超时 500ms → emergency_stop()           │
   │                                                         │
   │  // 可选: 显示画面 (OpenCV imshow)                        │
   │                                                         │
   │  fps_control(loop_start, 15)                            │
   └─────────────────────────────────────────────────────────┘

3. 停止 (Ctrl+C)
   ├─ 断开相机
   ├─ 断开机械臂
   └─ 关闭 UDP socket
```

### 3.3 动作队列调度

```
服务端返回 ActionChunk JSON (16步)
  ↓ json.loads → 拼接 arm_joint (16,5) + gripper (16,1) → (16, 6)
  ↓
客户端 ActionQueue (deque, maxlen=32)
  ↓  pop 每步 → {"shoulder_pan.pos": v0, ..., "gripper.pos": v5}
  ↓
安全检查 (ensure_safe_goal_position)
  ↓
DynamixelBus.sync_write("Goal_Position", goal_pos)
```

**预取逻辑**: 队列深度 < 8 时，说明服务器推理节奏可加快（服务端通过观察丢包/时延自行决定）。

### 3.4 安全检查

- 单步步长限制: `max_relative_target`（关节 ≤ 5.0°, 夹爪 ≤ 20.0%）
- 关节限位: 使用标定文件 `range_min` / `range_max`
- 急停: 动作队列空 + 超时 500ms → `emergency_stop()` → 释放力矩
- 最大队列深度 32（防止积压过期动作）

---

## 四、服务端推理接口设计（接口规范 + 数据格式）

> ⚠️ 服务端部分**不在当前实现范围内**，以下为接口规范定义，
> 供后续服务端实现参考。

### 4.1 服务端模块清单（规划）

| 模块 | 文件 | 职责 |
|------|------|------|
| UDP 接收 | `server/udp_receiver.py` | 接收两路视频帧，维护每路最新帧缓存 |
| UDP 发送 | `server/udp_sender.py` | 将动作 JSON 发送到客户端 |
| 推理引擎 | `server/inference_engine.py` | BeingH05 模型加载 + 推理 |
| 主服务 | `server/server_main.py` | 服务启动、推理循环 |

### 4.2 推理引擎接口 (InferenceEngine)

```python
class InferenceEngine:
    """Being-H05 VLA 推理引擎——服务端核心"""

    def __init__(self, config: InferenceConfig):
        """
        参数:
            config.model_path: str          # Being-H05 模型检查点路径
            config.data_config_name: str    # "koch_posttrain"
            config.dataset_name: str        # "koch_posttrain"
            config.embodiment_tag: str      # "koch"
            config.device: str              # "cuda"
            config.n_action_steps: int      # 16 (chunk_size)
            config.task: str                # 任务描述
        """

    def load_model(self) -> None:
        """加载 Being-H05 模型权重到 GPU。耗时约 10~30s。"""

    def infer(self, top_jpeg: bytes, wrist_jpeg: bytes) -> str:
        """
        单次推理——输入两路 JPEG 帧字节流，输出动作序列 JSON 字符串。

        输入:
            top_jpeg: bytes       # 顶部相机 JPEG 字节流
            wrist_jpeg: bytes     # 腕部相机 JPEG 字节流

        输出:
            str                   # 动作序列 JSON 字符串（格式见 §2.2）

        异常:
            InferenceTimeoutError: 推理超时 (>500ms)
            ModelNotReadyError:    模型未加载
        """

    def is_ready(self) -> bool:
        """模型是否就绪"""

    def unload_model(self) -> None:
        """卸载模型释放 GPU 显存"""
```

### 4.3 服务端主循环流程（规划）

```
1. 初始化
   ├─ 创建 2 个 UDP socket 绑定 0.0.0.0:5002 (top) 和 :5003 (wrist)
   ├─ 加载 Being-H05 模型
   └─ 打印就绪信息

2. 推理循环
   ┌─────────────────────────────────────────────────────────┐
   │  // 接收视频帧（非阻塞循环）                                │
   │  ├─ recvfrom(:5002) → 有数据 → 更新 latest_top_jpeg      │
   │  └─ recvfrom(:5003) → 有数据 → 更新 latest_wrist_jpeg    │
   │                                                         │
   │  // 推理触发（两路帧均非空时）                               │
   │  if latest_top_jpeg is not None                         │
   │     and latest_wrist_jpeg is not None:                  │
   │      action_json = engine.infer(                        │
   │          latest_top_jpeg,                               │
   │          latest_wrist_jpeg                              │
   │      )                                                  │
   │      sendto(192.168.2.5:5000, action_json.encode())     │
   │      sequence_id += 1                                   │
   └─────────────────────────────────────────────────────────┘

3. 停止 (Ctrl+C)
   ├─ 卸载模型
   └─ 关闭 UDP sockets
```

### 4.4 服务端配置参数

```python
@dataclass
class ServerConfig:
    # 网络
    server_host: str = "0.0.0.0"
    top_port: int = 5002           # 接收顶部相机 JPEG
    wrist_port: int = 5003         # 接收腕部相机 JPEG
    client_host: str = "192.168.2.5"
    client_port: int = 5000        # 发送动作 JSON

    # 模型
    model_path: str = "D:/study/code/lerobot/model-h05"
    device: str = "cuda"

    # 推理
    n_action_steps: int = 16
    task: str = "pick the green toy and place into box"

    # 帧管理
    frame_stale_timeout_ms: int = 500  # 帧过期时间
```

---

## 五、数据流时序

```
Client (192.168.2.5)                    Server (192.168.2.12)
  │                                         │
  │ ═══════ 主循环开始 (15Hz) ═══════════════  │
  │                                         │
  │── top_jpeg ──► :5002                    │── 更新 latest_top
  │── wrist_jpeg ► :5003                    │── 更新 latest_wrist
  │                                         │── infer(top, wrist)
  │◄─ action_json ── :5000 ────────────────│
  │                                         │
  │── 解析 JSON → 入队 (16步)                │
  │── 执行 step 0..15                       │
  │                                         │
  │── top_jpeg ──► :5002                    │
  │── wrist_jpeg ► :5003                    │
  │                                         │── infer(top, wrist)
  │◄─ action_json ── :5000 ────────────────│
  │                                         │
  │  ... (循环直到 Ctrl+C)                   │
  └─────────────────────────────────────────┘
```

> 客户端持续以固定频率（15Hz）推送两路视频帧，服务端收到新帧后触发推理，返回动作序列。两端完全解耦——客户端不关心服务端何时推理，服务端不关心客户端执行进度。

---

## 六、关键设计决策

### 6.1 为什么去掉协议层

| 原设计 | 简化后 | 理由 |
|--------|--------|------|
| 自定义 16B 包头 + 12B 视频子头 | 无头部，纯数据报 | 每路数据独占端口，端口号即"通道标识" |
| 应用层分片 (MTU 1400) | 不分片 | JPEG 帧 30-60KB，UDP 最大 65507B，完全装得下 |
| CRC16 校验 | 不校验 | 局域网 UDP 校验和已足够；JPEG 解码失败 = 自然丢帧 |
| 序列号/ACK/重传 | 不重传 | 视频允许丢帧；动作通过 `sequence_id` 仅做丢包检测（日志告警） |
| CONTROL 握手/心跳 | 去掉 | 客户端和服务端独立启动，各自配置参数 |

### 6.2 端口分离 vs 单端口复用

选择端口分离的原因:
- 端口号即语义: `5002` = top, `5003` = wrist, `5000` = action
- 无需任何字节级标记，数据报内容就是纯数据
- 服务端可直接按端口区分数据流，代码更简洁
- 便于独立调试（可用 netcat 单独发送/监听某一端口）

### 6.3 JPEG 压缩参数

- 分辨率: 640×480（与 Being-H05 训练数据一致）
- 质量: 80（视觉损失可忽略，30-60KB/帧）
- 双路 @ 15fps: ~0.9-1.8 MB/s (7-14 Mbps)，局域网 100Mbps 绰绰有余

---

## 七、文件结构

```
d:/study/code/lerobot/
├── client/                          # ★ 客户端（当前实现）
│   ├── __init__.py
│   ├── udp_sender.py                # UDP 发送工具（send JPEG to server）
│   ├── udp_receiver.py              # UDP 接收工具（recv action JSON）
│   ├── camera_streamer.py           # 双相机采集 + JPEG 编码
│   ├── robot_executor.py            # 机械臂控制 + 动作执行
│   └── client_main.py               # 客户端主入口
│
├── server/                          # 服务端（接口定义，后续实现）
│   ├── __init__.py
│   ├── udp_receiver.py              # 接收两路视频帧
│   ├── udp_sender.py                # 发送动作 JSON 到客户端
│   ├── inference_engine.py          # BeingH05 推理封装
│   └── server_main.py               # 服务端主入口
│
├── Being-H05推理方案2.md            # ★ 本文档
├── TODOV2.md                        # ★ 客户端实现 TODO 清单
├── lerobot_record_vla.py            # 现有单机脚本（保留不动）
├── Being-H05/                       # Being-H05 源码（不动）
└── src/lerobot/                     # LeRobot 框架（不动）
```

---

## 八、与现有代码的关系

| 现有模块 | 新方案中的角色 | 变更 |
|---------|-------------|------|
| `lerobot_record_vla.py` | 保留作为本地调试脚本 | 不修改 |
| `src/lerobot/policies/beingh_koch/` | 服务端推理引用 | 不修改 |
| `src/lerobot/robots/koch_follower/` | 客户端 robot_executor 引用 | 不修改 |
| `src/lerobot/cameras/opencv/` | 客户端 camera_streamer 引用 | 不修改 |
| `my_calibration/` | 客户端 robot_executor 加载 | 不修改 |
| `Being-H05/` | 服务端推理引擎加载 | 不修改 |

---

## 九、客户端启动命令

```bash
conda activate D:\study\code\lerobot\condaenv
python client/client_main.py \
    --server-host 192.168.2.12 \
    --top-port 5002 \
    --wrist-port 5003 \
    --action-port 5000 \
    --robot-port COM6 \
    --robot-id my_awesome_follower_arm \
    --calibration-dir ./my_calibration \
    --top-camera 1 \
    --wrist-camera 3 \
    --camera-width 640 \
    --camera-height 480 \
    --fps 15 \
    --jpeg-quality 80 \
    --display
```

## 十、服务端启动命令（规划）

```bash
python server/server_main.py \
    --host 0.0.0.0 \
    --top-port 5002 \
    --wrist-port 5003 \
    --client-host 192.168.2.5 \
    --client-port 5000 \
    --model-path D:/study/code/lerobot/model-h05 \
    --device cuda \
    --n-action-steps 16 \
    --task "pick the green toy and place into box"
```
