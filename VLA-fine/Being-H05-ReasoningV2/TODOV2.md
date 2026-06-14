# TODOV2 — Being-H05 远程推理客户端实现清单

> 本文档按依赖关系排序，仅覆盖**客户端推流 + 客户端执行**部分。
> 服务端部分仅定义接口和数据格式，不在当前实现范围内。
>
> 参考设计文档: [Being-H05推理方案2.md](./Being-H05推理方案2.md)

---

## 阶段零：基础设施（无外部依赖）

---

- TODO V2-01 UDP 发送工具模块 - 无依赖项
    封装 UDP socket 发送操作——创建 UDP socket、将数据报发送到指定 (host, port) 目标、可选的发送速率统计。每个数据报是完整独立的字节流，不添加任何协议头或分片逻辑。

    需要修改的文件：
      - D:\study\code\lerobot\client\__init__.py (新建)
      - D:\study\code\lerobot\client\udp_sender.py (新建)

    需要实现的功能 API：
      class UdpSender:
          def __init__(self, remote_host: str, remote_port: int):
              """
              创建 UDP socket 并设置目标地址。
              socket 不绑定固定本地端口（由 OS 自动分配）。
              """

          def send(self, data: bytes) -> int:
              """
              发送一个完整数据报到目标地址。
              参数:
                  data: 待发送的字节流（一个完整 JPEG 帧或 JSON 字符串）
              返回:
                  实际发送的字节数
              异常:
                  OSError: socket 发送失败
              """

          def send_json(self, obj: Any) -> int:
              """将 Python 对象序列化为 JSON 字符串并发送（便捷方法）。"""

          @property
          def bytes_sent(self) -> int:
              """累计发送字节数"""

          def close(self) -> None:
              """关闭 socket"""

          def __enter__ / __exit__ 支持 context manager

      # 便捷工厂函数
      def create_top_sender(server_host: str = "192.168.2.12",
                            top_port: int = 5002) -> UdpSender:
          """创建顶部相机视频发送器 → server:5002"""

      def create_wrist_sender(server_host: str = "192.168.2.12",
                              wrist_port: int = 5003) -> UdpSender:
          """创建腕部相机视频发送器 → server:5003"""

    设计注意点：
      - 一个 UdpSender 实例对应一个目标 (host, port)
      - 客户端需要两个实例: top_sender (→ :5002) 和 wrist_sender (→ :5003)
      - socket 使用 SOCK_DGRAM，不调用 connect()（保持无连接模式）
      - send() 是同步阻塞调用，在局域网中延迟通常 <1ms

    需要生成的测试：
      - tests/client/test_udp_sender.py：
        测试 UdpSender 初始化创建有效 socket
        测试 send 发送字节流并返回正确长度
        测试 send_json 正确序列化并发送
        测试 bytes_sent 累计计数正确
        测试 close 后 send 抛异常
        测试 context manager 自动关闭
        (mock socket 验证 sendto 参数正确)

---

- TODO V2-02 UDP 接收工具模块 - 无依赖项
    封装 UDP socket 接收操作——绑定本地端口、非阻塞接收数据报、可选超时、丢包检测（通过应用层 sequence_id 连续性）。

    需要修改的文件：
      - D:\study\code\lerobot\client\udp_receiver.py (新建)

    需要实现的功能 API：
      class UdpReceiver:
          def __init__(self, local_host: str, local_port: int,
                       timeout_ms: int = 10):
              """
              创建 UDP socket 并绑定本地地址。
              参数:
                  local_host: 绑定 IP (通常 "0.0.0.0")
                  local_port: 绑定端口 (客户端收动作用 5000)
                  timeout_ms: recvfrom 超时（毫秒），用于非阻塞轮询
              """

          def recv(self) -> bytes | None:
              """
              非阻塞尝试接收一个数据报。
              返回: 收到的字节流；超时/无数据返回 None
              异常: OSError (socket 错误)
              """

          def recv_json(self) -> dict | None:
              """
              非阻塞接收一个 JSON 数据报并解析为 dict。
              返回: 解析后的 dict；无数据或 JSON 解析失败返回 None
              """

          def recv_with_timeout(self, timeout_ms: float) -> bytes | None:
              """
              阻塞等待接收一个数据报，直到超时。
              参数:
                  timeout_ms: 最大等待时间（毫秒）
              返回: 收到的字节流；超时返回 None
              """

          @property
          def packets_received(self) -> int:
              """累计接收的数据报数量"""

          def close(self) -> None:
              """关闭 socket"""

          def __enter__ / __exit__ 支持 context manager

      class SequenceTracker:
          """简单的序列号追踪器——检测 action sequence_id 跳跃"""
          def __init__(self):
              self.last_seq: int = -1
              self.missed: int = 0        # 累计丢失数

          def update(self, seq_id: int) -> int | None:
              """
              检查序列号连续性。
              返回: None = 正常, int = 跳过的数量（丢包）
              """

    设计注意点：
      - 客户端绑定 0.0.0.0:5000 接收动作序列
      - 使用 socket.settimeout() 实现非阻塞轮询
      - JSON 解析失败时记录日志并返回 None（不抛异常）
      - SequenceTracker 仅用于日志告警，不影响执行流程
      - socket 设置 SO_REUSEADDR 允许快速重启

    需要生成的测试：
      - tests/client/test_udp_receiver.py：
        测试 UdpReceiver 初始化绑定正确端口
        测试 recv 在无数据时返回 None
        测试 recv_json 正确解析收到的 JSON
        测试 recv_json 在 JSON 格式错误时返回 None
        测试 packets_received 累计计数正确
        测试 SequenceTracker 连续 seq 返回 None
        测试 SequenceTracker 跳跃 seq 返回丢失数
        测试 close 后 recv 抛异常
        (mock socket 模拟收发)

---

## 阶段一：设备层 — 相机

---

- TODO V2-03 双相机采集与 JPEG 编码 - 依赖项：无（仅依赖 OpenCV 和现有 lerobot OpenCVCamera）
    基于 LeRobot OpenCVCamera 实现双相机同步采集封装——打开顶部相机 (index=1) 和腕部相机 (index=3)，独立后台线程异步读取帧，实时 JPEG 编码，维护每路相机的最新帧及编码缓存（线程安全）。

    需要修改的文件：
      - D:\study\code\lerobot\client\camera_streamer.py (新建)

    需要实现的功能 API：
      class CameraStreamer:
          def __init__(self, top_index: int = 1, wrist_index: int = 3,
                       width: int = 640, height: int = 480,
                       fps: int = 15, jpeg_quality: int = 80):
              """
              初始化双相机配置:
              1. top_camera = OpenCVCamera(OpenCVCameraConfig(index=1, width, height, fps))
              2. wrist_camera = OpenCVCamera(OpenCVCameraConfig(index=3, width, height, fps))
              3. JPEG 编码参数: [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
              """

          # ── 连接 ──
          def connect(self, warmup: bool = True) -> None:
              """依次连接两台相机，日志打印相机名称和实际分辨率"""

          # ── 帧读取 ──
          def read_top_frame(self) -> np.ndarray | None:
              """读取顶部相机最新原始帧 (BGR, H×W×3 uint8)"""

          def read_wrist_frame(self) -> np.ndarray | None:
              """读取腕部相机最新原始帧 (BGR, H×W×3 uint8)"""

          # ── JPEG 编码 ──
          def encode_jpeg(self, frame: np.ndarray) -> bytes:
              """
              将 BGR/OpenCV 帧编码为 JPEG 字节流。
              cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
              """

          def get_jpeg_both(self) -> dict:
              """
              同时获取两路相机的最新帧并编码为 JPEG:
              返回: {
                  "top_jpeg": bytes | None,
                  "wrist_jpeg": bytes | None,
              }
              若某路相机不可用，对应值为 None
              """

          # ── 显示 ──
          def get_display_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
              """获取两路原始帧用于显示 (BGR 格式)"""

          # ── 生命周期 ──
          def disconnect(self) -> None:
              """断开两路相机"""

          @property
          def is_connected(self) -> bool:
              """两路相机是否都已连接"""

    设计注意点：
      - 相机索引: top=1, wrist=3（OpenCV 中 1 是顶部相机，3 是腕部相机）
      - 后台线程由 OpenCVCamera 内部管理（_read_loop），不需要额外启动线程
      - read_latest() 是 OpenCVCamera 的非阻塞接口，返回最新帧
      - cv2.imencode 直接接受 BGR 格式，无需转换
      - 编码失败时返回 None 而非抛异常（保持主循环稳定）

    需要生成的测试：
      - tests/client/test_camera_streamer.py：
        测试 CameraStreamer 初始化——配置参数正确
        测试 encode_jpeg 生成有效 JPEG bytes (mock BGR frame)
        测试 get_jpeg_both 返回字典结构正确
        测试 disconnect 后 is_connected 返回 False
        (需要真实相机: 测试两路相机实际连接/读取/编码)

---

## 阶段二：设备层 — 机械臂

---

- TODO V2-04 机械臂执行控制器 - 依赖项：无（仅依赖现有 lerobot KochFollower）
    基于 LeRobot KochFollower 封装机械臂连接、动作队列管理、动作执行和安全检查——复用现有标定文件和电机总线协议，支持从 JSON ActionChunk 批量入队、单步弹出执行、目标位置限幅安全检查以及急停机制。

    需要修改的文件：
      - D:\study\code\lerobot\client\robot_executor.py (新建)

    需要实现的功能 API：
      class RobotExecutor:
          def __init__(self, port: str = "COM6",
                       robot_id: str = "my_awesome_follower_arm",
                       calibration_dir: str = "./my_calibration",
                       max_relative_target: float | None = 5.0):
              """
              初始化:
              1. 构建 KochFollowerConfig(port, id, calibration_dir, cameras={})
                 注意: cameras 为空——相机由 CameraStreamer 独立管理
              2. 创建 KochFollower 实例（不自动连接）
              3. 初始化动作队列: deque(maxlen=32)
              """

          # ── 连接 ──
          def connect(self) -> None:
              """连接机械臂，执行标定，配置电机 PID"""

          # ── 动作队列 ──
          def enqueue_action_json(self, action_json: dict) -> int:
              """
              解析服务端返回的 ActionChunk JSON 并批量入队:

              输入 action_json (来自 §2.2 的固定格式):
              {
                  "sequence_id": 42,
                  "chunk_size": 16,
                  "action": {
                      "arm_joint_position": [[...16步, 每步5值]],
                      "gripper_position": [[...16步, 每步1值]]
                  }
              }

              处理:
              1. 提取 arm_joint_position (16,5) 和 gripper_position (16,1)
              2. np.concatenate → (16, 6) ndarray
              3. 逐行入队 deque: {"shoulder_pan.pos": v0, ..., "gripper.pos": v5}
              4. 返回入队后的队列深度

              列序: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
              """

          def pop_action(self) -> dict | None:
              """
              从动作队列取出下一步动作:
              返回: {
                  "shoulder_pan.pos": float,
                  "shoulder_lift.pos": float,
                  "elbow_flex.pos": float,
                  "wrist_flex.pos": float,
                  "wrist_roll.pos": float,
                  "gripper.pos": float,
              }
              队列为空返回 None
              """

          @property
          def queue_depth(self) -> int:
              """当前动作队列中的残留步数"""

          @property
          def is_queue_empty(self) -> bool:
              """队列是否为空"""

          # ── 动作执行 ──
          def execute_action(self, action: dict) -> dict:
              """
              执行单步动作:
              1. 安全检查: ensure_safe_goal_position
                 (复用 lerobot.robots.utils.ensure_safe_goal_position)
              2. self._robot.send_action(action)
              3. 返回实际发送的动作（可能被 clip）
              """

          # ── 安全检查 ──
          def is_safe_position(self, goal_pos: dict) -> bool:
              """检查目标位置是否在标定文件定义的关节限位内"""

          def emergency_stop(self) -> None:
              """立即停止所有电机运动 (Torque_Enable = 0)"""

          def should_emergency_stop(self, queue_empty_duration_ms: float,
                                    max_empty_ms: float = 500.0) -> bool:
              """判断是否应触发急停: 队列空且持续时间超过阈值"""

          # ── 状态读取 (调试用，不发送给服务端) ──
          def get_state(self) -> dict:
              """
              读取当前关节状态:
              返回: {
                  "arm_joint_position": [float; 5],
                  "gripper_position": [float; 1],
                  "timestamp": float,
              }
              """

          # ── 生命周期 ──
          def disconnect(self) -> None:
              """断开机械臂（释放力矩）"""

          @property
          def is_connected(self) -> bool:
              """机械臂是否已连接"""

    设计注意点：
      - 复用标定文件: my_calibration/my_awesome_follower_arm.json
      - 关节顺序: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
      - 前 5 关节使用 MotorNormMode.DEGREES，夹爪使用 MotorNormMode.RANGE_0_100
      - max_relative_target=5.0 (关节 5° 步长限制)
      - 急停触发条件: 动作队列空 + 等待 > 500ms

    需要生成的测试：
      - tests/client/test_robot_executor.py：
        测试 RobotExecutor 初始化——标定文件加载成功
        测试 enqueue_action_json 正确解析 JSON 并入队 (16, 6) 动作
        测试 pop_action 按顺序弹出动作
        测试 queue_depth 正确反映队列长度
        测试 is_queue_empty 队列空时返回 True
        测试 execute_action 对越界动作进行 clip
        测试 emergency_stop 触发 Disable_Torque
        测试 should_emergency_stop 超时判断
        (需要真实机械臂: 测试连接、单步动作执行)

---

## 阶段三：主控制器

---

- TODO V2-05 客户端主控制器 - 依赖项：V2-01 (UdpSender), V2-02 (UdpReceiver), V2-03 (CameraStreamer), V2-04 (RobotExecutor)
    实现客户端主入口——组装各模块、启动相机和机械臂、运行主控制循环（采集→编码→推流 + 收动作→入队→执行）、帧率控制和延迟统计、优雅停止和异常处理。

    需要修改的文件：
      - D:\study\code\lerobot\client\client_main.py (新建)

    需要实现的功能 API：
      @dataclass
      class ClientConfig:
          # 网络
          server_host: str = "192.168.2.12"
          top_port: int = 5002
          wrist_port: int = 5003
          action_port: int = 5000

          # 相机
          top_camera_index: int = 1
          wrist_camera_index: int = 3
          camera_width: int = 640
          camera_height: int = 480
          camera_fps: int = 15
          jpeg_quality: int = 80

          # 机械臂
          robot_port: str = "COM6"
          robot_id: str = "my_awesome_follower_arm"
          calibration_dir: str = "./my_calibration"
          max_relative_target: float = 5.0

          # 控制
          control_fps: int = 15
          emergency_timeout_ms: float = 500.0
          display: bool = True

      class ClientMain:
          def __init__(self, config: ClientConfig):
              """
              组装各模块:
              1. self.top_sender = UdpSender(server_host, top_port)
              2. self.wrist_sender = UdpSender(server_host, wrist_port)
              3. self.action_receiver = UdpReceiver("0.0.0.0", action_port)
              4. self.camera = CameraStreamer(top_index, wrist_index, ...)
              5. self.robot = RobotExecutor(robot_port, robot_id, calibration_dir)
              6. self.seq_tracker = SequenceTracker()
              """

          def start(self) -> None:
              """
              1. self.camera.connect()
              2. self.robot.connect()
              3. 打印启动横幅（IP、端口、相机索引、机械臂状态）
              4. self._main_loop()
              """

          def stop(self) -> None:
              """
              1. self.robot.disconnect()
              2. self.camera.disconnect()
              3. self.top_sender.close()
              4. self.wrist_sender.close()
              5. self.action_receiver.close()
              6. 打印运行统计
              """

          def _main_loop(self) -> None:
              """
              主控制循环 (control_fps Hz):

              每个周期:
              ┌─ loop_start = time.perf_counter()

              ├─ 采集 + 推流:
              │   jpegs = self.camera.get_jpeg_both()
              │   if jpegs["top_jpeg"]:
              │       self.top_sender.send(jpegs["top_jpeg"])
              │   if jpegs["wrist_jpeg"]:
              │       self.wrist_sender.send(jpegs["wrist_jpeg"])

              ├─ 接收动作:
              │   action = self.action_receiver.recv_json()
              │   if action:
              │       self.seq_tracker.update(action["sequence_id"])
              │       self.robot.enqueue_action_json(action)

              ├─ 执行动作:
              │   next_action = self.robot.pop_action()
              │   if next_action:
              │       self.robot.execute_action(next_action)
              │       self._queue_empty_since = None
              │   else:
              │       if self._queue_empty_since is None:
              │           self._queue_empty_since = time.perf_counter()
              │       elif self.robot.should_emergency_stop(...):
              │           self.robot.emergency_stop()

              ├─ 显示 (if display):
              │   top, wrist = self.camera.get_display_frames()
              │   cv2.imshow("Top", top); cv2.imshow("Wrist", wrist)
              │   cv2.waitKey(1)

              └─ 帧率控制: precise_sleep(1/control_fps - elapsed)
              """

          def _precise_sleep(self, duration_s: float) -> None:
              """高精度 sleep（duration_s <= 0 时立即返回）"""

          def _print_status(self) -> None:
              """每秒打印一次状态: FPS, 发送/接收字节, 队列深度"""

          @property
          def stats(self) -> dict:
              """返回运行统计字典"""

    需要生成的测试：
      - tests/client/test_client_main.py：
        测试 ClientConfig 默认值正确
        测试 ClientMain 初始化——各子模块正确创建
        测试 _precise_sleep 负值立即返回
        测试 start/stop 生命周期——资源正确释放
        测试主循环 _process_incoming（mock receiver 返回 JSON）
        测试主循环 _execute_step（mock executor）
        测试 KeyboardInterrupt 触发安全停止
        (集成测试: 启动客户端→mock 服务端→验证完整流程)

---

## 阶段四：集成测试与调试辅助

---

- TODO V2-06 集成测试与模拟服务端 - 依赖项：V2-01 ~ V2-05
    编写轻量模拟服务端——用于无真实服务端时的客户端集成测试——在指定端口接收两路视频帧（仅验证数据到达）、按固定间隔返回模拟动作序列 JSON、可配置丢包率和延迟以测试客户端健壮性。

    需要修改的文件：
      - D:\study\code\lerobot\tests\client\__init__.py (新建)
      - D:\study\code\lerobot\tests\client\conftest.py (新建)
      - D:\study\code\lerobot\tests\client\mock_server.py (新建)
      - D:\study\code\lerobot\tests\client\test_integration.py (新建)

    需要实现的功能 API (mock_server.py):
      class MockServer:
          """模拟 Being-H05 推理服务端——用于客户端集成测试"""

          def __init__(self, host: str = "0.0.0.0",
                       top_port: int = 5002,
                       wrist_port: int = 5003,
                       client_host: str = "127.0.0.1",
                       client_port: int = 5000):
              """创建 3 个 UDP socket 绑定对应端口"""

          def start(self, response_interval_ms: int = 200) -> None:
              """
              启动后台线程:
              - 线程1: 接收 :5002 (top) 和 :5003 (wrist) 的视频帧
              - 线程2: 每隔 response_interval_ms 发送模拟动作 JSON 到客户端
              """

          def generate_mock_action(self) -> dict:
              """
              生成模拟动作序列:
              - arm_joint_position: 16 步正弦波轨迹
              - gripper_position: 16 步恒定值
              - sequence_id 递增
              """

          def set_loss_rate(self, rate: float) -> None:
              """设置动作丢包率 (0.0 ~ 1.0)"""

          def set_latency_ms(self, latency: int) -> None:
              """设置额外延迟 (ms)"""

          def get_received_frame_count(self, camera: str) -> int:
              """获取收到的 top/wrist 帧数"""

          def stop(self) -> None:
              """停止模拟服务端，关闭所有 socket"""

      # conftest.py fixtures:
      - mock_server_fixture: 启动/停止模拟服务端（使用随机端口避免冲突）
      - client_config_fixture: 提供测试用 ClientConfig（指向 localhost）
      - sample_action_json_fixture: 提供标准 ActionChunk JSON 样本

    需要生成的测试 (test_integration.py):
      测试完整流程——客户端启动 → 推流 → 收到动作 → 入队执行
      测试客户端 video_push——mock_server 成功接收到 top/wrist JPEG
      测试客户端 action_receive——正确解析 mock_server 发送的动作 JSON
      测试动作队列——16 步动作依次弹出执行
      测试丢包场景——mock_server 动作丢包 30% → seq_tracker 检测到跳跃
      测试延迟场景——mock_server 动作延迟 300ms → 客户端继续正常工作
      测试急停——mock_server 停止发送动作 → 客户端超时触发 emergency_stop
      测试客户端 stop——资源正确清理

---

- TODO V2-07 CLI 启动脚本与日志 - 依赖项：V2-05 (ClientMain)
    创建客户端命令行启动入口——支持命令行参数解析（tyro）、打印启动信息摘要（连接配置、设备状态），结构化日志输出（INFO 级别关键事件 + DEBUG 级别周期统计），以及 Ctrl+C 优雅停止。

    需要修改的文件：
      - D:\study\code\lerobot\client\client_main.py 的 __main__ 块（或独立 cli.py）

    需要实现的功能：
      # 命令行入口
      def main():
          """解析 CLI 参数 → ClientConfig → ClientMain → start()"""

      # 支持的命令行参数 (tyro):
      --server-host         服务端 IP (默认: 192.168.2.12)
      --top-port            顶部相机目标端口 (默认: 5002)
      --wrist-port          腕部相机目标端口 (默认: 5003)
      --action-port         动作接收端口 (默认: 5000)
      --top-camera          顶部相机 OpenCV 索引 (默认: 1)
      --wrist-camera        腕部相机 OpenCV 索引 (默认: 3)
      --camera-width        相机宽度 (默认: 640)
      --camera-height       相机高度 (默认: 480)
      --camera-fps          相机帧率 (默认: 15)
      --jpeg-quality        JPEG 质量 1-100 (默认: 80)
      --robot-port          机械臂串口 (默认: COM6)
      --robot-id            机械臂标定 ID (默认: my_awesome_follower_arm)
      --calibration-dir     标定文件目录 (默认: ./my_calibration)
      --max-relative-target 单步最大位移 (默认: 5.0)
      --fps                 控制循环频率 (默认: 15)
      --display             是否显示画面 (默认: true)
      --log-level           日志级别 (默认: INFO)

      # 启动横幅示例:
      """
      ╔══════════════════════════════════════════════════════╗
      ║  Being-H05 Remote Client v1.0.0                     ║
      ╠══════════════════════════════════════════════════════╣
      ║  Server:  192.168.2.12 (top:5002, wrist:5003)       ║
      ║  Action:  recv on 0.0.0.0:5000                      ║
      ║  Cameras: top=1, wrist=3 @ 640x480, 15fps           ║
      ║  Robot:   COM6 (my_awesome_follower_arm)             ║
      ║  Control: 15Hz, emergency timeout 500ms              ║
      ╚══════════════════════════════════════════════════════╝
      """

    设计注意点：
      - 日志使用 logging 模块，默认 INFO 级别
      - DEBUG 级别每秒输出: FPS, 发送 KB/s, 队列深度
      - 启动时打印完整配置摘要
      - Ctrl+C → 优雅停止流程（打印统计摘要）
      - 相机/机械臂连接失败时给出明确错误信息

    需要生成的测试：
      该模块为集成性质——通过 TODO V2-06 的集成测试间接覆盖

---

- TODO V2-08 可视化调试面板 - 依赖项：V2-05 (ClientMain)
    在 OpenCV 显示窗口上叠加运行信息——FPS、发送/接收速率、动作队列深度、最后一次收到动作的时间戳（用于判断服务端是否在线）。

    需要修改的文件：
      - D:\study\code\lerobot\client\debug_overlay.py (新建)

    需要实现的功能 API：
      class DebugOverlay:
          """轻量级调试信息叠加层"""

          def draw_status_bar(self, frame: np.ndarray, stats: dict) -> np.ndarray:
              """
              在视频帧底部叠加状态栏:
              - FPS: 14.8 / 15
              - Queue: [████████░░░░░░░░] 8/16
              - TX: 1.2 MB/s | Last Action: 0.05s ago
              - Server: ● LIVE  (● green=正常, ● red=超时>1s)
              返回带叠加层的帧
              """

    需要生成的测试：
      - tests/client/test_debug_overlay.py：
        测试 draw_status_bar 返回与输入同尺寸的帧
        测试 stats 为空时不会崩溃

---

## TODO 执行顺序总结

```
阶段零（UDP 基础，无外部依赖）:
  ├─ TODO V2-01  UDP 发送工具模块 (udp_sender.py)
  └─ TODO V2-02  UDP 接收工具模块 (udp_receiver.py)

阶段一（设备层 — 相机，独立）:
  └─ TODO V2-03  双相机采集与 JPEG 编码 (camera_streamer.py)

阶段二（设备层 — 机械臂，独立）:
  └─ TODO V2-04  机械臂执行控制器 (robot_executor.py)

阶段三（主控制器，依赖阶段零~二）:
  └─ TODO V2-05  客户端主控制器 (client_main.py)

阶段四（测试与辅助）:
  ├─ TODO V2-06  集成测试与模拟服务端
  ├─ TODO V2-07  CLI 启动脚本与日志
  └─ TODO V2-08  可视化调试面板
```

> **最小可运行子集（MVP）**：完成 V2-01 ~ V2-05 即可通过 `python client/client_main.py` 启动客户端，连接模拟服务端完成完整推流+执行流程测试。

---

## 优先级速查

| TODO | 名称 | 优先级 | 阶段 |
|------|------|--------|------|
| V2-01 | UDP 发送工具模块 | 🔴 高 | 零 |
| V2-02 | UDP 接收工具模块 | 🔴 高 | 零 |
| V2-03 | 双相机采集与 JPEG 编码 | 🔴 高 | 一 |
| V2-04 | 机械臂执行控制器 | 🔴 高 | 二 |
| V2-05 | 客户端主控制器 | 🔴 高 | 三 |
| V2-06 | 集成测试与模拟服务端 | 🟡 中 | 四 |
| V2-07 | CLI 启动脚本与日志 | 🟡 中 | 四 |
| V2-08 | 可视化调试面板 | 🟢 低 | 四 |

---

## 服务端接口规范（供参考，不在当前实现范围）

详见 [Being-H05推理方案2.md 第四节](./Being-H05推理方案2.md#四服务端推理接口设计接口规范--数据格式)。

核心接口签名:

```python
# 服务端推理引擎 (server/inference_engine.py)
class InferenceEngine:
    def load_model(self) -> None: ...
    def infer(self, top_jpeg: bytes, wrist_jpeg: bytes) -> str: ...
    def is_ready(self) -> bool: ...
    def unload_model(self) -> None: ...

# infer() 输出 JSON 格式 (§2.2):
{
    "sequence_id": 42,
    "chunk_size": 16,
    "inference_time_ms": 85.3,
    "action": {
        "arm_joint_position": [[...], ...],  // [16][5]
        "gripper_position": [[...], ...]      // [16][1]
    }
}
```

服务端实现后应按此接口对接客户端。
