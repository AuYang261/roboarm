# Being-H05 远程推理方案 2 — UDP 推流 + 远程推理 + 动作回传

> **目标**: 将相机视频流推送到服务器，在服务器端运行 Being-H05 VLA 推理，将动作序列通过 UDP 回传客户端执行。
>
> **传输方式**: 纯 UDP 数据报，无自定义协议头，无分片重组，无控制通道。
>
> **当前实现范围**: 客户端（推流 + 执行）。服务端部分仅定义接口和数据格式。
>
> **实现路线**:
> ```
> ① 本地 UDP 视频回环验证  →  ② Action.json 回放驱动机械臂  →  ③ 接入服务端实时推理
>  ┌──────────────┐            ┌──────────────────────┐          ┌──────────────────┐
>  │相机→JPEG→UDP │            │ 相机→JPEG→UDP 推流    │          │ 相机→JPEG→UDP 推流│
>  │→UDP→解码→显示│            │ action.json→Robot执行 │          │ 服务端推理→动作回传│
>  │(localhost)   │            │ (localhost/remote)    │          │ (remote server)  │
>  └──────────────┘            └──────────────────────┘          └──────────────────┘
>   里程碑 1: 管道通              里程碑 2: 机器人动               里程碑 3: 远程推理闭环
> ```
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
| 相机采集 | `client/camera_streamer.py` | OpenCV 双相机读取 + JPEG 编码线程 |
| UDP 发送 | `client/udp_sender.py` | 将 JPEG 字节发送到服务端指定端口 |
| UDP 接收 | `client/udp_receiver.py` | 从服务端接收动作 JSON，非阻塞 |
| Action 回放 | `client/action_replay.py` | 从 action.json 加载预录动作序列（无服务端时的数据源） |
| 本地回环测试 | `client/local_video_loop.py` | 本地 UDP 视频回环验证（里程碑 1） |
| 机械臂控制 | `client/robot_executor.py` | Koch 机械臂连接/动作执行/安全检查 |
| 主控制器 | `client/client_main.py` | 组装各模块、主循环、支持回放/推理双模式 |

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

### 3.3 两种运行模式

#### 模式 A: Action 回放模式（当前可实现，无需服务端）

```
pick_orange_simple_shape/data/chunk-000/action.json
  ↓ ActionReplay.load()
  ↓ jsonl 逐行解析: {"action": [6 floats], "frame_index": N, ...}
  ↓ iter_actions() → 逐帧返回 np.ndarray (6,)
  ↓
RobotExecutor.execute_action_array(action)
  ↓ 安全检查 (ensure_safe_goal_position)
  ↓ DynamixelBus.sync_write("Goal_Position", goal_pos)
```

**特点**: 视频正常推流到服务端端口，但机械臂动作来自本地 JSON 文件。适合独立调试机械臂执行管道。

#### 模式 B: 服务端推理模式（需服务端就绪后切换）

```
UdpReceiver.recv_json() → ActionChunk JSON
  ↓ 解析 action.arm_joint_position (16,5) + action.gripper_position (16,1)
  ↓ np.concatenate → (16, 6)
  ↓
客户端 ActionQueue (deque, maxlen=32)
  ↓  pop 每步 → RobotExecutor.execute_action_array()
```

运行命令: `python client/client_main.py --no-action-replay`

### 3.4 本地 UDP 视频回环测试

位于 `client/local_video_loop.py`，在 localhost 上验证完整的"采集→编码→UDP发送→UDP接收→解码→显示"管道。

```
CameraStreamer (top=1, wrist=3)
  │  read_latest + JPEG encode
  ▼
UdpSender → 127.0.0.1:15002 (top)
UdpSender → 127.0.0.1:15003 (wrist)
  │
UdpReceiver ← bind 0.0.0.0:15002 (top)
UdpReceiver ← bind 0.0.0.0:15003 (wrist)
  │  cv2.imdecode
  ▼
Display + 延迟统计
```

运行命令: `python client/local_video_loop.py --duration 10 --display`

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
│   ├── local_video_loop.py          # 本地 UDP 视频回环测试 (里程碑1)
│   ├── action_replay.py             # action.json 回放加载器
│   ├── robot_executor.py            # 机械臂控制 + 动作执行
│   └── client_main.py               # 客户端主入口 (回放/推理双模式)
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
├── pick_orange_simple_shape/        # 预录数据集（ActionReplay 数据源）
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

### 里程碑 1: 本地 UDP 视频回环测试

```bash
condaenv\python.exe client\local_video_loop.py \
    --top-camera 1 \
    --wrist-camera 3 \
    --width 640 \
    --height 480 \
    --fps 15 \
    --duration 10 \
    --display
```

### 里程碑 2: Action 回放模式（预录动作驱动机械臂）

```bash
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
    --action-replay \
    --action-json pick_orange_simple_shape/data/chunk-000/action.json \
    --action-interval 0.05 \
    --display
```

### 里程碑 3: 服务端推理模式（需服务端就绪）

```bash
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
    --no-action-replay \
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

---

## 十一、实现记录

> 实现日期: 2026-06-14
> 实现范围: 客户端推流 + 执行 (V2-01 ~ V2-06)

### V2-01: 双相机采集与 JPEG 编码

**文件**: `client/camera_streamer.py`

**实现要点**:
- 基于 LeRobot `OpenCVCamera` + `OpenCVCameraConfig`，设置 `color_mode=ColorMode.BGR` 避免不必要的颜色转换
- 后台读取线程由 OpenCVCamera 内部管理，`read_latest()` 非阻塞
- `cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])` 输出完整 JPEG 字节流
- 640×480 JPEG quality=80 单帧约 5.4KB（黑帧）~30-60KB（实景帧）
- 两路相机独立连接，wrist 连接失败时自动清理 top 连接

**遇到的问题**:
- Windows MSMF 硬件变换兼容性: 需要在 import cv2 前设置 `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`
- 相机索引确认: top=1, wrist=3（用户提供）

### V2-02: UDP 发送与接收工具模块

**文件**: `client/udp_sender.py`, `client/udp_receiver.py`

**实现要点**:
- `UdpSender`: 不绑定固定本地端口（OS 自动分配），通过 `sendto()` 发送完整数据报
- `UdpReceiver`: 绑定指定端口，`settimeout(10ms)` 实现非阻塞轮询，`recv_json()` 自动解析 JSON
- `SequenceTracker`: 追踪 `sequence_id` 连续性，检测丢包并记录日志
- `SO_REUSEADDR` 设置允许快速重启

**遇到的问题**:
- 未遇到技术问题，测试通过（本地回环 send/receive JSON 和 bytes 正常）

### V2-03: 本地 UDP 视频回环测试（里程碑 1）

**文件**: `client/local_video_loop.py`

**实现要点**:
- 主线程: CameraStreamer 采集 → JPEG 编码 → UdpSender 发送到 localhost:15002/15003
- 接收线程: UdpReceiver 接收 → `cv2.imdecode` 解码 → 验证尺寸 → 存入共享缓冲区
- 主线程同时负责显示（带状态叠加层）和帧率控制
- `LoopStats` 数据类追踪: 发送/接收帧数、延迟样本、丢帧率、解码错误
- 循环结束后打印完整统计并运行验证检查

**遇到的问题**:
- Windows GBK 编码: 日志中使用 Unicode 字符（✅/❌/→）导致 `UnicodeEncodeError`，替换为 ASCII 等效字符 `[PASS]`/`[FAIL]`/`->`
- `cv2.imdecode` 返回 `None` 时需检查数据尺寸，增加 `img.shape` 校验

### V2-04: Action 回放加载器（里程碑 2 数据源）

**文件**: `client/action_replay.py`

**实现要点**:
- JSONL 逐行解析：`pick_orange_simple_shape/data/chunk-000/action.json`（315步，4个 episode）
- 每行包含 `action` (6 floats) 和 `observation.state` (6 floats)，单位：度
- 关节顺序与 KochFollower 一致: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- `get_action_dict()` 转换为 KochFollower.send_action 兼容格式
- 数据验证: frame_index 单调性、动作值范围检查

**数据统计**（来自实际加载）:
```
Shoulder pan:   [-7.25,  37.44]
Shoulder lift: [-91.86,   9.88]
Elbow flex:    [-82.21,  28.21]
Wrist flex:    [-68.21,  -0.71]
Wrist roll:    [ -1.64,  12.72]
Gripper:       [ 30.52,  50.41]
```

**遇到的问题**:
- 文件路径: 默认路径 `pick_orange_simple_shape/...` 是相对于 project root，需要通过 `Path(__file__).resolve().parent.parent` 解析
- 数据中只有 1 个 episode（episode_index=0），不是预期的 4 个 — 因为 parquet 文件有 4 个但 action.json 只有 episode 0 的动作数据

### V2-05: 机械臂执行控制器

**文件**: `client/robot_executor.py`

**实现要点**:
- 接口设计参考 `arm_base.py` 的 Arm 基类模式: `set_arm_angles(angles_deg, gripper_0to1)`, `get_arm_angles()`, `set_gripper()`, `move_to_home()`, `enable_torque()/disable_torque()`, `emergency_stop()`
- 底层基于 LeRobot `KochFollower` + `DynamixelMotorsBus`
- 创建 `KochFollowerConfig(cameras={})` 不绑定相机（由 CameraStreamer 独立管理）
- 夹爪转换: gripper_0to1 ∈ [0,1] ↔ Koch 内部 [0, 100]
- 动作执行支持多种格式: `execute_action_array()` (numpy 6,), `execute_action_dict()` (Koch 格式), `execute_action_sequence()` (批量)
- 安全检查: 复用 `ensure_safe_goal_position` 和标定文件 `range_min`/`range_max`

**遇到的问题**:
- 仅验证了导入和标定文件加载，未连接真实机械臂（需要 COM6 和实际硬件）

### V2-06: 客户端主控制器

**文件**: `client/client_main.py`

**实现要点**:
- 两种运行模式通过 `--action-replay`/`--no-action-replay` 切换
- 模式 A: 视频推流 + action.json 回放驱动机械臂
- 模式 B: 视频推流 + UDP 接收服务端推理结果
- 主循环 15Hz: 采集→编码→UDP发送(上行) + 获取动作→执行(下行) + 显示
- 启动时打印完整配置横幅
- Ctrl+C 优雅停止，按顺序释放: 机械臂(先)→相机→UDP→显示

**遇到的问题**:
- 未连接真实机械臂进行实际运行测试

### V2-06a: 机械臂回放脚本

**文件**: `client/replay_actions.py`

独立回放脚本——不依赖相机和 UDP，仅从 action.json 读取动作序列驱动机械臂。

**遇到的问题与修复**:

#### 问题 1: IndexError — angles 与 gripper 维度不匹配

**症状**:
```
IndexError: list index out of range
  at: f"Delta: {[f'{first_action[i] - (angles[i] if angles else 0):.1f}' for i in range(6)]}"
```

**原因**: `RobotExecutor.get_arm_angles()` 返回 `angles` 只有 5 个元素（前 5 个关节），`gripper` 单独返回。代码中用 `range(6)` 对 `angles` 取索引 5 时越界。`action.json` 中 `action` 数组是 6 维（5 joints + 1 gripper），但 `get_arm_angles()` 遵循 `arm_base.py` 模式将夹爪独立返回。

**修复**: 将 delta 计算拆为两部分：前 5 维从 `angles` 计算差值，第 6 维（gripper）从 `gripper` 变量计算（注意 `gripper` 是 [0,1]，需乘 100 转换为 [0,100] 后再与 action 数组的 gripper 值比较）。

#### 问题 2: Dynamixel 总线通信失败

**症状**:
```
ERROR: set_arm_angles failed: Failed to sync read 'Present_Position' 
on ids=[1,2,3,4,5,6] after 1 tries. [TxRxResult] There is no status packet!
ERROR: Step 20 execution FAILED, stopping
```

**原因**: `KochFollower.send_action()` 在 `max_relative_target is not None` 时会先调用 `sync_read("Present_Position")` 获取当前位置做限幅检查（`ensure_safe_goal_position`），然后再 `sync_write("Goal_Position")`。回放间隔设为 0.033s（30Hz）时，每次 send_action 需要一次 `sync_read` + `sync_write` 往返，Dynamixel 总线来不及响应，导致 "no status packet" 错误。运行到约第 20 步时总线缓冲区溢出彻底断开。

**修复**（两处）:
1. `MAX_RELATIVE_TARGET = None`: 跳过每步 `sync_read`。action.json 是真实录制数据，动作本身已平滑安全，不需要每步限幅检查。
2. `PLAYBACK_INTERVAL = 0.05`（20Hz）: Dynamixel 总线安全通信频率。该频率下 `sync_write` 单步有 50ms 间隔，总线不会拥堵。

**涉及的代码路径**:
- `replay_actions.py`: `MAX_RELATIVE_TARGET` / `PLAYBACK_INTERVAL` 常量
- `robot_executor.py` → `KochFollowerConfig(max_relative_target=...)`
- `src/lerobot/robots/koch_follower/koch_follower.py` `send_action()`: 第 221 行 `if self.config.max_relative_target is not None` 分支触发 `sync_read`
- `src/lerobot/motors/dynamixel.py` `DynamixelMotorsBus.sync_read()`: 底层总线通信

---

## 十二、硬件验证指令

> 所有命令在项目根目录 `d:\study\code\lerobot` 下运行。
> Python 解释器: **必须使用** `condaenv\python.exe`（不能用系统 `python`，否则缺少依赖）
> 或在 PowerShell 中先执行: `conda activate D:\study\code\lerobot\condaenv`

### 12.1 验证前准备

```bash
# 1. 确认 conda Python 可用
condaenv\python.exe -c "import cv2; print('OpenCV', cv2.__version__)"

# 2. 确认串口 COM6 存在
#    Windows: 设备管理器 → 端口(COM和LPT) → 确认 USB Serial Device (COM6)

# 3. 确认标定文件存在
dir my_calibration\my_awesome_follower_arm.json

# 4. 确认 action.json 存在
dir pick_orange_simple_shape\data\chunk-000\action.json
```

### 12.2 步骤一：相机发现

```bash
condaenv\python.exe -c "
import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
import cv2
for i in range(10):
    for backend, bid in [('DSHOW', 700), ('MSMF', 1400)]:
        cap = cv2.VideoCapture(i, bid)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f'Camera {i} [{backend}]: {frame.shape[1]}x{frame.shape[0]} OK')
            cap.release()
            break
        cap.release()
"
```

**期望输出**: 至少看到 Camera 1 和 Camera 3 各有一条 `OK`。
**如果索引不同**: 记下顶部相机和腕部相机的实际索引，后续命令中替换 `--top-camera` / `--wrist-camera` 参数。

### 12.3 步骤二：单相机画面测试（不推流）

```bash
condaenv\python.exe -c "
import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
import cv2, time

# 同时打开 camera 1 和 3
cap_top = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap_wrist = cv2.VideoCapture(3, cv2.CAP_DSHOW)

print('Press q to quit')
while True:
    ret1, f1 = cap_top.read()
    ret2, f2 = cap_wrist.read()
    if ret1: cv2.imshow('Top Camera (1)', f1)
    if ret2: cv2.imshow('Wrist Camera (3)', f2)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap_top.release(); cap_wrist.release(); cv2.destroyAllWindows()
"
```

**验证点**:
- [ ] 顶部相机画面正常（画面清晰、无花屏）
- [ ] 腕部相机画面正常
- [ ] 两路相机帧率稳定（无卡顿）

### 12.4 步骤三：本地 UDP 视频回环测试（里程碑 1）

```bash
condaenv\python.exe client\local_video_loop.py \
    --top-camera 1 \
    --wrist-camera 3 \
    --duration 10 \
    --display
```

**验证点**:
- [ ] `[PASS]: Top frames received > 0`
- [ ] `[PASS]: Wrist frames received > 0`
- [ ] `[PASS]: No decode errors`
- [ ] `drop_top` 和 `drop_wrist` < 5%
- [ ] 显示窗口能看到两路相机画面（带状态栏叠加层）

**如果相机索引不同**: 加上 `--top-camera X --wrist-camera Y`

### 12.5 步骤四：机械臂连接测试（不执行动作）

```bash
condaenv\python.exe -c "
from client.robot_executor import RobotExecutor

robot = RobotExecutor(port='COM6', robot_id='my_awesome_follower_arm',
                       calibration_dir='./my_calibration')
robot.connect()
print('Robot connected!')
print(f'Joint limits: {robot.joint_limits}')

angles, gripper = robot.get_arm_angles()
print(f'Current angles: {angles}')
print(f'Current gripper: {gripper}')

robot.disconnect()
print('Robot disconnected.')
"
```

**验证点**:
- [ ] 连接成功，无异常
- [ ] 关节限位数据正确（来自标定文件）
- [ ] 当前角度在合理范围
- [ ] 断开成功

**如果 COM6 不通**: 检查串口号，替换 `port='COMX'`

### 12.6 步骤五：单步动作回放测试（机械臂微动验证）

> ⚠️ 此步骤会驱动机器臂执行一个微小的移动。确保机器人处于安全位置且无碰撞风险。

```bash
condaenv\python.exe -c "
import time
from client.action_replay import ActionReplay
from client.robot_executor import RobotExecutor

# 加载 action.json
replay = ActionReplay('pick_orange_simple_shape/data/chunk-000/action.json')

# 获取第 1 步动作（起点位置，通常是最安全的位置）
action0 = replay.get_action(0)
print(f'Action 0: {action0}')
print('Joint names:', RobotExecutor(port='COM6').joint_names)

# 连接并执行
robot = RobotExecutor(port='COM6', robot_id='my_awesome_follower_arm',
                       calibration_dir='./my_calibration')
robot.connect()

# 先读取当前状态
angles, gripper = robot.get_arm_angles()
print(f'Before: angles={angles}, gripper={gripper}')

# 执行第 1 步（应接近当前状态，移动很小）
print('Executing action step 0...')
ok = robot.execute_action_array(action0)
print(f'Result: {\"OK\" if ok else \"FAILED\"}')

time.sleep(0.5)
angles, gripper = robot.get_arm_angles()
print(f'After:  angles={angles}, gripper={gripper}')

robot.disconnect()
"
```

**验证点**:
- [ ] `Result: OK`
- [ ] 机械臂有微小位移（或保持同位置——取决于当前状态与 action0 的差异）
- [ ] 电机关节号与标定文件一致

### 12.7 步骤六：完整回放模式运行（里程碑 2）

> ⚠️ 此步骤会执行完整的 315 步动作序列。确保机器人处于安全位置，准备随时按 Ctrl+C 停止。

```bash
condaenv\python.exe client\client_main.py \
    --server-host 192.168.2.12 \
    --top-camera 1 \
    --wrist-camera 3 \
    --robot-port COM6 \
    --robot-id my_awesome_follower_arm \
    --calibration-dir ./my_calibration \
    --camera-width 640 \
    --camera-height 480 \
    --fps 15 \
    --jpeg-quality 80 \
    --action-replay \
    --action-json pick_orange_simple_shape/data/chunk-000/action.json \
    --action-interval 0.05 \
    --display
```

**验证点**:
- [ ] 启动横幅打印完整配置信息
- [ ] 两路相机推流正常（UDP 发送无异常）
- [ ] 动作从 action.json 逐步回放，机器人平滑运动
- [ ] 显示窗口有状态栏叠加层（FPS / step / 模式）
- [ ] Ctrl+C 优雅停止，打印统计摘要
- [ ] 机器人先断开，再断开相机和 UDP

**仅测试推流（不接机械臂）**:
```bash
# 创建仅相机的轻量测试——修改 client_main.py 跳过 robot.connect()
# 临时方法: 使用 --robot-port NONE 并在代码中跳过连接
# 或者直接运行 local_video_loop.py（步骤三）
```

### 12.8 步骤七：模拟服务端 → 客户端联调

> 本测试不需要真实服务端——在一台机器上启动 mock server 验证动作接收管道。

```bash
# 终端 1: 启动 mock server（纯 Python，发送模拟动作 JSON）
condaenv\python.exe -c "
import socket, json, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
seq = 0
while True:
    action = {
        'sequence_id': seq,
        'chunk_size': 16,
        'inference_time_ms': 50.0,
        'action': {
            'arm_joint_position': [[10.0]*5]*16,
            'gripper_position': [[50.0]]*16
        }
    }
    data = json.dumps(action).encode()
    sock.sendto(data, ('192.168.2.5', 5000))
    print(f'Sent action seq={seq}')
    seq += 1
    time.sleep(1.0)
"

# 终端 2: 启动客户端（仅接收+显示，不连机械臂）
condaenv\python.exe -c "
from client.udp_receiver import UdpReceiver
import time
r = UdpReceiver('0.0.0.0', 5000, timeout_ms=100)
print('Waiting for actions...')
while True:
    obj = r.recv_json()
    if obj:
        print(f'Received: seq={obj[\"sequence_id\"]}, chunk={obj[\"chunk_size\"]}')
"
```

**验证点**:
- [ ] 终端 1 显示 `Sent action seq=0,1,2...`
- [ ] 终端 2 显示 `Received: seq=0,1,2...`
- [ ] 无丢包（在本地或局域网内 sequence_id 连续）

### 12.9 验证清单

| 步骤 | 内容 | 耗时 | 状态 |
|------|------|------|------|
| 12.1 | 环境准备 | 1 min | □ |
| 12.2 | 相机发现 | 1 min | □ |
| 12.3 | 单相机画面测试 | 2 min | □ |
| 12.4 | 本地 UDP 回环（里程碑1） | ~10s 运行 | □ |
| 12.5 | 机械臂连接测试 | 2 min | □ |
| 12.6 | 单步动作回放 | 1 min | □ |
| 12.7 | 完整回放模式（里程碑2） | ~20s 运行 | □ |
| 12.8 | 模拟服务端联调 | 5 min | □ |

### 12.10 常见问题排查

| 症状 | 可能原因 | 解决方法 |
|------|---------|---------|
| 相机 1/3 无法打开 | MSMF 后端不兼容 | 已默认使用 DSHOW，或手动 `--camera-backend 700` |
| `UnicodeEncodeError` | Windows GBK 编码 | 已全部替换为 ASCII 字符 |
| 机械臂连接失败 | COM 端口不对 | 设备管理器确认端口号 |
| 机械臂标定失败 | 标定文件路径不对 | 检查 `--calibration-dir` 和 `--robot-id` |
| 动作执行无反应 | Dynamixel 力矩未使能 | 检查 `KochFollower.connect()` 日志 |
| UDP 发送失败 | 防火墙拦截 | 临时关防火墙，或添加 Python 例外 |
| 视频延迟高 | JPEG quality 过高 | `--jpeg-quality 60` 降低质量 |
| 丢帧严重 | WiFi 不稳定 | 使用有线连接，或降低 `--camera-fps 10` |

