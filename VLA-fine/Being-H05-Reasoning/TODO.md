# Being-H05 VLA → LeRobot 适配实现 TODO

> 本文档基于 [Being-H05-Feat-lerobot.md](./Being-H05-Feat-lerobot.md) 设计文档生成，按依赖关系排序。
> 排序原则：**基础模块优先 → 被依赖模块优先 → 核心功能优先 → 扩展功能次之**。

---

## 阶段一：基础设施（无外部依赖，所有后续模块的基石）

---

### TODO-01: Being-H05 源码路径配置与依赖验证 ✅ 已完成

<details>
<summary>完成摘要 — 2026-06-13</summary>

- 创建 6 个缺失的 `__init__.py`: `configs/`, `BeingH/inference/`, `BeingH/dataset/`, `BeingH/utils/`, `BeingH/benchmark/`, `BeingH/train/`
- 创建 `condaenv/Lib/site-packages/beingh.pth` 路径配置文件
- 安装 4 个缺失依赖: `numpydantic` (1.9.1), `timm` (1.0.27), `tyro` (1.0.13), `pyzmq` (27.1.0)
- 测试: `tests/policies/beingh_koch/test_imports.py` — **9/9 通过**
- 详见 [step.md](./step.md) 和 [Debug.md](./Debug.md)
</details>

---

### TODO-02: KochPosttrainDataConfig — Koch 具身在 200 维统一空间中的映射定义

- **依赖项**：TODO-01（Being-H05 源码路径可导入）
- **优先级**：🔴 高
- **prompt**：

```
在 Being-H05 的 data_config.py 中新增 KochPosttrainDataConfig 类，定义 Koch 机器人
在 200 维统一动作/状态空间中的语义槽位映射。

需要修改的文件：
  - D:\study\code\lerobot\Being-H05\configs\data_config.py
    新增 KochPosttrainDataConfig 类
    在 DATA_CONFIG_MAP 中注册 "koch_posttrain" → KochPosttrainDataConfig

需要实现的功能 API：
  class KochPosttrainDataConfig(BaseDataConfig):
      VIDEO_KEYS = ["video.above", "video.wrist"]
      VIDEO_SOURCE_COLUMNS = {
          "video.above": "observation.images.above",
          "video.wrist": "observation.images.wrist",
      }
      STATE_KEYS = ["state.arm_joint_position", "state.gripper_position"]
      ACTION_KEYS = ["action.arm_joint_position", "action.gripper_position"]
      LANGUAGE_KEYS = ["language.instruction"]

      # 200 维统一空间映射（核心定义）
      UNIFIED_MAPPING = {
          "state.arm_joint_position":  (50, 55),  # 5-dof 关节 → 统一空间 [50:55)
          "state.gripper_position":    (18, 19),  # 1-dof 夹爪 → 统一空间 [18:19)
          "action.arm_joint_position": (50, 55),
          "action.gripper_position":   (18, 19),
      }

      state_normalization_modes = {
          "state.arm_joint_position": "mean_std",
          "state.gripper_position": "mean_std",
      }
      action_normalization_modes = {
          "action.arm_joint_position": "mean_std",
          "action.gripper_position": "mean_std",
      }

      def define_modalities(self) -> Dict[str, ModalityDef]:
          # 从 parquet 列定义各模态（此处为推理时的"虚拟"定义，字段存在即可）
          ...

      def get_transforms(self) -> ModalityTransform:
          # 定义归一化 pipeline：StateActionToTensor → StateActionTransform
          ...

  - D:\study\code\lerobot\Being-H05\configs\data_config.py 末尾
    DATA_CONFIG_MAP["koch_posttrain"] = KochPosttrainDataConfig

设计注意点：
  - UNIFIED_MAPPING 中的 (start, end) 必须与 model-h05/koch_posttrain_metadata.json 中
    记录的 arm_joint_position shape=(5,) 和 gripper_position shape=(1,) 一致
  - 关节使用统一空间 dims [50:55)，因为标准 arm_joint_position 占 7 维，Koch 只用前 5 维
  - normalization_modes 使用 "mean_std"，与 metadata.json 中的 statistics 字段匹配
    （metadata 中有 mean/std/q01/q99，mean_std 模式使用 mean 和 std）

需要生成的测试：
  - tests/policies/beingh_koch/test_koch_data_config.py：
    测试 KochPosttrainDataConfig 可正常实例化
    测试 UNIFIED_MAPPING 维度与 metadata.json 一致
    测试 get_transforms() 返回的 pipeline 结构正确
    测试 define_modalities() 返回的模态定义键名完整
```

---

## 阶段二：LeRobot 侧配置与模型封装

---

### TODO-03: BeingHKochConfig — LeRobot 策略配置类

- **依赖项**：TODO-02（KochPosttrainDataConfig 已可用）
- **优先级**：🔴 高
- **prompt**：

```
创建 BeingHKochConfig 类，继承 PreTrainedConfig，作为 LeRobot 框架识别
policy.type=beingh_koch 的配置入口。

需要新建的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\__init__.py
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\configuration_beingh_koch.py

需要实现的功能 API：
  @PreTrainedConfig.register_subclass("beingh_koch")
  @dataclass
  class BeingHKochConfig(PreTrainedConfig):
      # ── 必需路径参数 ──
      pretrained_path: str = ""              # Being-H05 模型检查点路径

      # ── Being-H05 数据配置参数 ──
      data_config_name: str = "koch_posttrain"  # Being-H05 DataConfig 名称
      dataset_name: str = "koch_posttrain"       # 对应 metadata.json 的 dataset key
      embodiment_tag: str = "koch"               # 具身标签

      # ── 观测/动作维度 ──
      n_obs_steps: int = 1
      chunk_size: int = 16                   # 与 action_chunk_length 一致
      n_action_steps: int = 16

      # ── 归一化配置 ──
      normalization_mapping: dict = field(default_factory=lambda: {
          "VISUAL": NormalizationMode.IDENTITY,
          "STATE": NormalizationMode.MEAN_STD,
          "ACTION": NormalizationMode.MEAN_STD,
      })

      # ── 推理参数 ──
      enable_rtc: bool = True
      num_inference_timesteps: int | None = None  # None=使用模型默认值(4)
      use_mpg: bool | None = None                 # None=使用模型默认值(false)
      device: str = "cuda"

      # ── 图像预处理 ──
      force_image_size: int = 224             # ViT 输入尺寸
      resize_imgs_with_padding: tuple | None = None  # None=不额外 resize

      # ── 语言指令模板 ──
      instruction_template: str = (
          "According to the instruction '{task_description}', "
          "what's the micro-step actions in the next {k} steps?"
      )

      # ── 训练参数（预留给后续微调）──
      freeze_vision_encoder: bool = True
      train_expert_only: bool = True

      def __post_init__(self):
          super().__post_init__()
          # 验证 pretrained_path 存在且包含必要文件
          # 验证 chunk_size 与 n_action_steps 的一致性

      def validate_features(self) -> None:
          # 定义输入特征：2 个摄像头 + 6 维状态
          # 定义输出特征：6 维动作

      @property
      def observation_delta_indices(self) -> list: return [0]
      @property
      def action_delta_indices(self) -> list: return list(range(self.chunk_size))
      @property
      def reward_delta_indices(self) -> None: return None

需要修改的文件：
  - 无（新建文件，通过 @register_subclass 自动注册）

需要生成的测试：
  - tests/policies/beingh_koch/test_configuration_beingh_koch.py：
    测试 BeingHKochConfig 可正常实例化
    测试 pretrained_path 指向不存在的目录时 __post_init__ 应报错
    测试 validate_features 生成的特征字典键名正确（observation.images.above 等）
    测试 PreTrainedConfig.get_choice_class("beingh_koch") 返回 BeingHKochConfig
    测试 normalization_mapping 的默认值
```

---

### TODO-04: BeingHWrapper — BeingHPolicy 的薄封装层

- **依赖项**：TODO-02（KochPosttrainDataConfig）、TODO-03（BeingHKochConfig）
- **优先级**：🔴 高
- **prompt**：

```
创建 BeingHWrapper 类，封装 BeingHPolicy 的初始化、推理调用和资源管理。
这是连接 LeRobot 框架与 Being-H05 原生推理引擎的桥梁。

需要新建的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\beingh_wrapper.py

需要实现的功能 API：
  class BeingHWrapper:
      def __init__(self, config: BeingHKochConfig):
          """
          1. 验证 config.pretrained_path 存在
          2. 验证关键文件存在: model.safetensors, config.json,
             koch_posttrain_metadata.json, tokenizer.json
          3. 构建 instruction_template
          4. 调用 BeingHPolicy(...) 初始化模型:
             - model_path = config.pretrained_path
             - data_config_name = config.data_config_name  # "koch_posttrain"
             - dataset_name = config.dataset_name
             - embodiment_tag = config.embodiment_tag
             - instruction_template = config.instruction_template
             - max_view_num = -1  # 使用所有相机视图
             - use_fixed_view = False
             - enable_rtc = config.enable_rtc
             - num_inference_timesteps = config.num_inference_timesteps
             - use_mpg = config.use_mpg
             - device = config.device
          5. 保存 beingh_policy 引用
          6. 提取 unified_mapping 和 metadata
          """

      def get_action(self, observations: dict) -> dict:
          """
          直接委托给 self.beingh_policy.get_action(observations)
          输入: 符合 BeingH 格式的观测字典
          输出: {"action.arm_joint_position": [[...],...], "action.gripper_position": [[...],...]}
          """

      def get_unified_mapping(self) -> dict:
          """返回 Koch 的 UNIFIED_MAPPING"""

      def reset(self):
          """重置推理状态（如有必要）"""

      def close(self):
          """清理 GPU 资源"""

      @property
      def device(self) -> torch.device:
          """返回当前设备"""

      @property
      def action_chunk_length(self) -> int:
          """返回 action_chunk_length (固定为 16)"""

设计注意点：
  - BeingHPolicy 初始化时 data_config_name 必须与 TODO-02 中注册的名称一致
  - BeingHPolicy.__init__ 中会调用 _load_metadata(Path(model_path))，需要确保
    koch_posttrain_metadata.json 在 model_path 根目录下
  - BeingHPolicy 使用 AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    因此 tokenizer 文件必须在 model_path 目录中
  - 模型加载过程（BeingH from_pretrained + safetensors load）较慢（~10-30s），
    建议在 __init__ 中打印进度日志
  - device 参数控制模型加载到 cuda 还是 cpu

需要生成的测试：
  - tests/policies/beingh_koch/test_beingh_wrapper.py：
    测试 BeingHWrapper 初始化——传入有效的 BeingHKochConfig
    测试 BeingHWrapper 初始化——pretrained_path 不存在时报错
    测试 get_unified_mapping() 返回正确的映射字典
    测试 action_chunk_length 属性返回 16
    测试 close() 方法（mock torch.cuda.empty_cache）
    注意：需要 mock BeingHPolicy 以避免实际加载大模型
```

---

### TODO-05: BeingHKochPolicy — PreTrainedPolicy 核心实现

- **依赖项**：TODO-03（BeingHKochConfig）、TODO-04（BeingHWrapper）
- **优先级**：🔴 高
- **prompt**：

```
实现 BeingHKochPolicy 类，继承 PreTrainedPolicy，作为 LeRobot 策略核心。
关键要求：实现 select_action 接口，与 lerobot-record 的 record_loop 兼容。

需要新建的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\modeling_beingh_koch.py

需要实现的功能 API：
  class BeingHKochPolicy(PreTrainedPolicy):
      config_class = BeingHKochConfig
      name = "beingh_koch"

      def __init__(self, config: BeingHKochConfig, **kwargs):
          """
          1. super().__init__(config)
          2. config.validate_features()
          3. self.beingh = BeingHWrapper(config)  ← 模型加载在这
          4. self.reset()  ← 初始化动作队列
          """

      def reset(self):
          """
          重置动作队列 _queues = {ACTION: deque(maxlen=n_action_steps)}
          注意：deque maxlen 为 config.n_action_steps (默认 16)
          """

      @torch.no_grad()
      def select_action(self, batch: dict[str, Tensor]) -> Tensor:
          """
          单步动作选择（与 record_loop 的主循环兼容）：
          1. self.eval()
          2. 从 _queues[ACTION] 取出缓存的动作
          3. 若队列为空，调用 _get_action_chunk(batch) 填充队列
          4. 返回 _queues[ACTION].popleft()  ← shape (action_dim,)

          返回: Tensor (action_dim,) 即 Koch 6 维动作
          """

      @torch.no_grad()
      def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
          """
          整块动作预测（可选，供外部直接使用）：
          1. 调用 _get_action_chunk(batch)
          2. 返回 (1, chunk_size, action_dim) Tensor
          """

      def _get_action_chunk(self, batch: dict) -> Tensor:
          """
          核心推理方法：
          1. 从 batch 提取观测数据
          2. 转换为 BeingH 所需的观测字典格式
          3. 调用 self.beingh.get_action(observations) → action_dict
          4. 从 action_dict 提取并拼接 Koch 6 维动作:
             actions = concat(
               action_dict["action.arm_joint_position"],  # (chunk, 5)
               action_dict["action.gripper_position"],     # (chunk, 1)
             )  → (chunk, 6)
          5. unsqueeze(0) → (1, chunk, 6)
          6. 返回 Tensor
          """

      def _prepare_observations(self, batch: dict) -> dict:
          """
          将 LeRobot 格式的 batch 转换为 BeingH 期望的观测字典：

          输入 batch:
          {
            "observation.images.above": Tensor (B, H, W, 3) uint8,
            "observation.images.wrist": Tensor (B, H, W, 3) uint8,
            "observation.state": Tensor (B, 6),  # [sp, sl, ef, wf, wr, gripper]
            "task": "pick the green toy...",
          }

          输出:
          {
            "video.above": np.ndarray (H, W, 3) uint8,  ← 去掉 batch 维度
            "video.wrist": np.ndarray (H, W, 3) uint8,
            "state.arm_joint_position": np.ndarray (5,),  ← 前 5 维
            "state.gripper_position": np.ndarray (1,),    ← 第 6 维
            "language.instruction": ["pick the green toy..."],
          }

          注意:
          - Tensor → numpy: .cpu().numpy()
          - 图像需要保持 uint8 原始格式（BeingH 内部会做 transform）
          - 状态值是物理值（电机位置），不需要提前归一化（BeingH 内部做）
          """

      def forward(self, batch, noise=None, time=None, reduction="mean"):
          """
          训练前向传播（后续微调时实现，当前返回 NotImplementedError）
          """

设计注意点：
  - select_action 的返回值 shape 为 (action_dim,)，即 (6,)
  - _queues 的调度逻辑参考 SmolVLAPolicy：
    - 入队: self._queues[ACTION].extend(actions.transpose(0, 1)[:n_action_steps])
    - 出队: self._queues[ACTION].popleft()
  - batch["observation.state"] 的列序必须与 KochFollower 的 motor 顺序一致:
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
  - 状态值为归一化后的电机位置（KochFollower 的 MotorNormMode.RANGE_M100_100
    输出范围 [-100, 100]，夹爪为 RANGE_0_100 输出范围 [0, 100]）

需要生成的测试：
  - tests/policies/beingh_koch/test_modeling_beingh_koch.py：
    测试 BeingHKochPolicy 初始化（mock BeingHWrapper）
    测试 reset() 清空动作队列
    测试 select_action —— 队列非空时直接弹出缓存
    测试 select_action —— 队列空时触发 _get_action_chunk
    测试 _prepare_observations —— 键名映射正确
    测试 _prepare_observations —— 状态维度拆分正确 (6 → 5+1)
    测试 _get_action_chunk —— 动作拼接维度正确 (5+1→6)
    测试 forward —— 当前应抛出 NotImplementedError
```

---

## 阶段三：处理器（Processor）层

---

### TODO-06: 自定义 ProcessorStep 实现

- **依赖项**：TODO-02（UNIFIED_MAPPING 已定义）
- **优先级**：🔴 高
- **prompt**：

```
实现三个自定义 ProcessorStep，分别处理状态填充、图像预处理和动作拼接。

需要新建的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\processor_beingh_koch.py

需要实现的类：

┌──────────────────────────────────────────────────────────────────┐
│ 1. BeingHStatePaddingProcessorStep                               │
│    继承: ComplementaryDataProcessorStep（修改 complementary_data）│
│    注册: @ProcessorStepRegistry.register(name="beingh_state_padding")│
│                                                                   │
│    功能: 将 Koch 6 维状态向量映射到 BeingH 200 维统一状态空间      │
│                                                                   │
│    输入: observation.state Tensor (B, 6)                          │
│    处理:                                                          │
│      - 创建 zeros (B, 200)                                        │
│      - zeros[:, 50:55] = state[:, 0:5]  ← arm_joint_position     │
│      - zeros[:, 18:19] = state[:, 5:6]  ← gripper_position       │
│    输出: observation.state Tensor (B, 200)                        │
│                                                                   │
│    配置参数:                                                      │
│      state_mapping: dict[str, tuple[int, int]]                    │
│        = {"arm_joint_position": (50,55), "gripper_position": (18,19)}│
│                                                                   │
│    方法:                                                          │
│      complementary_data(self, data) -> dict                       │
│      transform_features(self, features) -> features               │
├──────────────────────────────────────────────────────────────────┤
│ 2. BeingHImagePreprocessorStep                                    │
│    继承: RenameObservationsProcessorStep 或自定义                 │
│    注册: @ProcessorStepRegistry.register(name="beingh_image_prep")│
│                                                                   │
│    功能: 将 LeRobot 的图像数据从 Tensor 转为 BeingH 期望的格式     │
│                                                                   │
│    输入: observation.images.{key} Tensor (B, H, W, 3) uint8       │
│    处理:                                                          │
│      - Tensor → numpy array (H, W, 3) uint8                       │
│      - 可选: resize 到 force_image_size (224x224)                 │
│      - 键名不变（BeingH 内部通过 VIDEO_KEYS 匹配）                 │
│    输出: 保持 numpy 格式（BeingH 的 _modality_transform 内部       │
│           会用 StateActionToTensor 转为 torch.Tensor）             │
│                                                                   │
│    配置参数:                                                      │
│      force_image_size: int = 224                                  │
│      image_keys: list[str] = ["observation.images.above",         │
│                               "observation.images.wrist"]          │
│                                                                   │
│    方法:                                                          │
│      complementary_data(self, data) -> dict                       │
│      transform_features(self, features) -> features               │
├──────────────────────────────────────────────────────────────────┤
│ 3. BeingHActionUnsliceProcessorStep                               │
│    继承: 标准的 ProcessorStep（处理 PolicyAction 的 action 字段）  │
│    注册: @ProcessorStepRegistry.register(name="beingh_action_unslicer")│
│                                                                   │
│    功能: 从 BeingH 返回的 action_dict 拼接 Koch 6 维动作，         │
│          并根据 Koch 电机范围进行 clamp                            │
│                                                                   │
│    输入: PolicyAction.action Tensor (chunk, total_action_dim)     │
│          其中 total_action_dim 可能为 6 或需要从多 key 拼接        │
│    处理:                                                          │
│      - 拼接 arm_joint (5-d) + gripper (1-d) → (chunk, 6)         │
│      - Clamp: joints ∈ [-100, 100], gripper ∈ [0, 100]           │
│    输出: PolicyAction.action Tensor (chunk, 6)                    │
│                                                                   │
│    配置参数:                                                      │
│      action_keys: list[str] = ["action.arm_joint_position",       │
│                                "action.gripper_position"]          │
│      joint_range: tuple = (-100, 100)                             │
│      gripper_range: tuple = (0, 100)                              │
│                                                                   │
│    方法:                                                          │
│      forward(self, data: PolicyAction) -> PolicyAction            │
│      transform_features(self, features) -> features               │
└──────────────────────────────────────────────────────────────────┘

注意: 以上三个自定义 ProcessorStep 参考 SmolVLANewLineProcessor 的注册方式
（使用 @ProcessorStepRegistry.register(name="...")）

需要生成的测试：
  - tests/policies/beingh_koch/test_processor_steps.py：
    测试 BeingHStatePaddingProcessorStep —— 6→200 维度映射正确
    测试 BeingHStatePaddingProcessorStep —— 填充的零不影响非 Koch 维度
    测试 BeingHImagePreprocessorStep —— Tensor→numpy 转换正确
    测试 BeingHActionUnsliceProcessorStep —— 5+1→6 拼接正确
    测试 BeingHActionUnsliceProcessorStep —— clamp 生效（超出范围值被截断）
```

---

### TODO-07: Pre/Post Processor Pipeline 组装

- **依赖项**：TODO-03（BeingHKochConfig）、TODO-06（自定义 ProcessorStep）
- **优先级**：🔴 高
- **prompt**：

```
实现 make_beingh_koch_pre_post_processors() 函数，组装完整的预处理和后处理流水线。

需要修改的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\processor_beingh_koch.py
    （与 TODO-06 同一个文件，新增加工厂函数）

需要实现的功能 API：
  def make_beingh_koch_pre_post_processors(
      config: BeingHKochConfig,
      dataset_stats: dict | None = None,
  ) -> tuple[
      PolicyProcessorPipeline[dict, dict],       # preprocessor
      PolicyProcessorPipeline[PolicyAction, PolicyAction],  # postprocessor
  ]:
      """
      构造预处理和后处理流水线。

      预处理器 Pipeline (输入: LeRobot batch dict → 输出: 处理后的 dict):
        Step 1: RenameObservationsProcessorStep
          - rename_map: config 中定义的 observation key 映射
          - 将 LeRobot 键名映射为 BeingH/策略内部键名

        Step 2: BeingHStatePaddingProcessorStep
          - 将 Koch 6 维状态填充为 BeingH 200 维统一状态

        Step 3: BeingHImagePreprocessorStep
          - 图像格式转换（Tensor → numpy），可选 resize

        Step 4: AddBatchDimensionProcessorStep
          - 添加 batch 维度（如已有则跳过）

        Step 5: TokenizerProcessorStep (可选，如果语言指令未 tokenize)
          - 使用 Being-H05 的 tokenizer 对 task 描述进行 tokenization
          - 注意: BeingH 在 get_action 内部也会 tokenize，此处可能需要绕过
          - 若 BeingHWrapper.get_action 内部已处理，则此步骤可为 pass-through

        Step 6: DeviceProcessorStep
          - 数据移至 config.device

      后处理器 Pipeline (输入: PolicyAction → 输出: PolicyAction):
        Step 1: BeingHActionUnsliceProcessorStep
          - 拼接 5+1 → 6 维 Koch 动作，clamp 到有效范围

        Step 2: DeviceProcessorStep
          - 数据移至 CPU

      返回:
        (PolicyProcessorPipeline, PolicyProcessorPipeline)
      """

设计注意点：
  - 参考 SmolVLA 的 make_smolvla_pre_post_processors() 实现模式
  - preprocessor pipeline name: POLICY_PREPROCESSOR_DEFAULT_NAME
  - postprocessor pipeline name: POLICY_POSTPROCESSOR_DEFAULT_NAME
  - postprocessor 的 to_transition / to_output 转换器使用
    policy_action_to_transition / transition_to_policy_action（与 SmolVLA 一致）
  - TokenizerProcessorStep 需要谨慎处理——BeingH 内部自带 tokenizer，
    若在 preprocessor 中提前 tokenize 会导致双重 tokenization
    策略: 在 preprocessor 中保留原始 task 字符串，在 BeingHWrapper 内部处理 tokenization

需要生成的测试：
  - tests/policies/beingh_koch/test_processor_pipeline.py：
    测试 make_beingh_koch_pre_post_processors 返回两个 Pipeline 对象
    测试 preprocessor pipeline 的步骤数量和顺序
    测试 postprocessor pipeline 的步骤数量和顺序
    测试 preprocessor pipeline 端到端处理 mock batch
    测试 postprocessor pipeline 端到端处理 mock PolicyAction
```

---

## 阶段四：框架注册与集成

---

### TODO-08: LeRobot Factory 注册与动态发现验证

- **依赖项**：TODO-03（BeingHKochConfig）、TODO-05（BeingHKochPolicy）、TODO-07（processor pipeline）
- **优先级**：🔴 高
- **prompt**：

```
验证 BeingHKochPolicy 可通过 LeRobot 的 factory 动态发现机制正常加载。

需要检查/修改的文件：
  - D:\study\code\lerobot\src\lerobot\policies\factory.py
    检查 _get_policy_cls_from_policy_name() 和 _make_processors_from_policy_config()
    是否已通过动态 import 支持 3rd-party policy。
    当前代码已有此逻辑（通过 PreTrainedConfig.get_known_choices()），但需验证。

  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\__init__.py
    确保导出 BeingHKochConfig, BeingHKochPolicy, make_beingh_koch_pre_post_processors

需要验证的流程：
  1. PreTrainedConfig.get_known_choices() 包含 "beingh_koch"
  2. get_policy_class("beingh_koch") 返回 BeingHKochPolicy
  3. make_policy_config("beingh_koch", pretrained_path="...") 返回 BeingHKochConfig 实例
  4. make_pre_post_processors(policy_cfg, dataset_stats=...) 返回 (pre, post) pipeline

若动态发现不生效，则需在 factory.py 中显式添加：
  # get_policy_class()
  elif name == "beingh_koch":
      from lerobot.policies.beingh_koch.modeling_beingh_koch import BeingHKochPolicy
      return BeingHKochPolicy

  # make_policy_config()
  elif policy_type == "beingh_koch":
      return BeingHKochConfig(**kwargs)

如果显式添加，需要同步修改：
  - D:\study\code\lerobot\src\lerobot\policies\factory.py

需要生成的测试：
  - tests/policies/beingh_koch/test_factory_integration.py：
    测试 PreTrainedConfig.get_known_choices() 包含 "beingh_koch"
    测试 get_policy_class("beingh_koch") 返回正确的类
    测试 make_policy_config("beingh_koch") 返回 BeingHKochConfig
    测试 make_pre_post_processors 正常返回 pipeline（使用默认 config）
```

---

### TODO-09: CLI 推理启动脚本 lerobot_record_vla.py

- **依赖项**：TODO-05（BeingHKochPolicy）、TODO-07（processor）、TODO-08（factory）
- **优先级**：🟡 中
- **prompt**：

```
创建独立的推理启动脚本，支持不通过 lerobot-record 框架直接运行 Being-H05 推理，
便于调试和快速测试。

需要新建的文件：
  - D:\study\code\lerobot\lerobot_record_vla.py

需要实现的功能：

  1. 命令行参数解析（使用 dataclass + tyro 或 argparse）：
     --model_path       Being-H05 模型路径 (默认: D:/study/code/lerobot/model-h05)
     --robot_port       Koch 机械臂 COM 端口 (默认: COM6)
     --robot_id         机械臂标定 ID (默认: my_awesome_follower_arm)
     --calibration_dir  标定文件目录 (默认: ./my_calibration)
     --top_camera_index 顶部相机 opencv index (默认: 0)
     --wrist_camera_index 臂上相机 opencv index (默认: 4)
     --task             任务描述 (默认: "pick the green toy and place into box")
     --fps              控制频率 (默认: 30)
     --display_data     是否显示画面 (默认: true)
     --record           是否录制数据集 (默认: false)
     --dataset_repo_id  数据集 repo_id
     --dataset_root     数据集保存根目录

  2. 核心循环（参考 lerobot_record.py 的 record_loop 结构）：
     while True:
       # (a) 获取机器人观测
       obs = robot.get_observation()

       # (b) 构建 batch 字典
       batch = {
         "observation.images.above": obs["above"],
         "observation.images.wrist": obs["wrist"],
         "observation.state": obs_state_vector,
         "task": task_description,
       }

       # (c) 策略推理
       action = beingh_policy.select_action(batch)  # (6,) Tensor

       # (d) 发送给机器人
       robot_action = {
         "shoulder_pan.pos": action[0].item(),
         "shoulder_lift.pos": action[1].item(),
         "elbow_flex.pos":    action[2].item(),
         "wrist_flex.pos":    action[3].item(),
         "wrist_roll.pos":    action[4].item(),
         "gripper.pos":       action[5].item(),
       }
       robot.send_action(robot_action)

       # (e) 可选: 记录到数据集
       if record:
           dataset.add_frame(frame)

       # (f) 帧率控制
       precise_sleep(1/fps - elapsed)

  3. 集成 BeingHKochPolicy（非 lerobot-record 框架路径）：
     config = BeingHKochConfig(
       pretrained_path=args.model_path,
       device="cuda",
     )
     policy = BeingHKochPolicy(config)

  4. 集成 KochFollower 机器人：
     robot_config = KochFollowerConfig(
       port=args.robot_port,
       cameras={...},
     )
     robot = KochFollower(robot_config)
     robot.connect()

注意:
  - 该脚本不依赖 lerobot-record 的 RecordConfig 和完整 dataset 框架
  - 若 --record 为 true，需要创建 LeRobotDataset 实例
  - 需要 import Being-H05 和 lerobot 两个代码库
  - 键盘中断 (Ctrl+C) 时需安全断开机器人连接并清理资源

需要生成的测试：
  - 该脚本为集成性质，不适合单元测试
  - 建议在 TODO-11 的集成测试中覆盖实际运行场景
```

---

## 阶段五：测试与验证

---

### TODO-10: 单元测试套件

- **依赖项**：TODO-03 ~ TODO-08 全部完成
- **优先级**：🟡 中
- **prompt**：

```
编写完整的单元测试套件，覆盖各模块的核心功能。

需要新建的文件：
  - D:\study\code\lerobot\tests\policies\beingh_koch\__init__.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\conftest.py
    提供共享的 fixtures: mock_model_path, mock_beingh_policy, sample_batch 等
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_imports.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_koch_data_config.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_configuration_beingh_koch.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_beingh_wrapper.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_modeling_beingh_koch.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_processor_steps.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_processor_pipeline.py
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_factory_integration.py

测试策略：
  - 使用 unittest.mock 或 pytest-mock mock 掉 BeingHPolicy（避免加载大模型）
  - mock BeingHPolicy.get_action() 返回固定的示例输出
  - 使用 model-h05 目录的真实 config.json 和 metadata.json（不需要加载模型权重）
  - 测试数据使用人造的小批量样本

conftest.py 应提供的关键 fixtures:
  - mock_beingh_policy_fixture: mock BeingHPolicy，返回固定的 action_dict
  - sample_batch_fixture: LeRobot 格式的样本 batch 字典
  - beingh_koch_config_fixture: 有效的 BeingHKochConfig 实例
  - koch_metadata_fixture: 从 model-h05/koch_posttrain_metadata.json 加载的真实 metadata

测试覆盖率目标: ≥80% 的核心逻辑代码（不含模型本身）
```

---

### TODO-11: 端到端集成测试

- **依赖项**：TODO-09（CLI 脚本）、TODO-10（单元测试）
- **优先级**：🟡 中
- **prompt**：

```
编写端到端集成测试，验证完整推理链路。

需要新建/修改的文件：
  - D:\study\code\lerobot\tests\policies\beingh_koch\test_integration.py

测试场景：

  1. 模型加载测试（需要 GPU，可标记为 slow）:
     - 使用真实 model-h05 加载 BeingHKochPolicy
     - 验证模型加载成功且 device 正确
     - 验证 reset() 后 _queues 为空

  2. 单步推理测试（需要 GPU）:
     - 加载模型
     - 构建模拟 Koch 观测（使用标定文件中的 rest position）
     - 调用 select_action
     - 验证输出 shape 为 (6,)
     - 验证输出值在合理范围内（joints ∈ [-100,100], gripper ∈ [0,100]）

  3. Chunk 推理测试（需要 GPU）:
     - 调用 predict_action_chunk
     - 验证输出 shape 为 (1, 16, 6)
     - 验证 16 步动作中有变化（非全相等）

  4. 动作队列测试:
     - 连续调用 select_action 16 次
     - 验证每次返回值不同（从 chunk 中逐步消费）
     - 验证第 17 次调用触发新推理（队列为空后重新填充）

  5. lerobot-record 集成测试（使用 mock robot）:
     - 模拟 lerobot-record 的完整调用链
     - policy = make_policy(cfg, ds_meta)
     - preprocessor, postprocessor = make_pre_post_processors(...)
     - 验证 preprocessor(batch) 输出格式正确
     - 验证 postprocessor(policy_action) 输出格式正确

运行建议:
  - 标记 GPU 测试为 @pytest.mark.slow + @pytest.mark.gpu
  - CI 中可以使用 mock 版本的测试，GPU 测试由开发者手动运行
```

---

## 阶段六：扩展功能

---

### TODO-12: BeingHKochPolicy 训练支持（forward 实现）

- **依赖项**：TODO-05（BeingHKochPolicy 推理已实现）
- **优先级**：🟢 低
- **prompt**：

```
实现 BeingHKochPolicy.forward() 方法以支持 LeRobot 训练框架进行微调。

需要修改的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\modeling_beingh_koch.py

需要实现的功能 API：
  def forward(self, batch, noise=None, time=None, reduction="mean"):
      """
      训练前向传播，计算 Flow Matching 损失。

      1. 调用 _prepare_observations(batch) 转换观测格式
      2. 构建 BeingH 训练所需的 packed_inputs
      3. 调用 beingh.forward() 计算 action_loss
      4. 返回 loss, loss_dict

      注意：
      - Being-H05 使用 Flow Matching 训练，需要生成 noise 和 time
      - BeingH.forward() 需要完整 packed_inputs（包括 packed_text_ids,
        packed_position_ids, split_lens, attn_modes 等）
      - 训练时 use_flow_matching=True，使用 L1 或 MSE loss
      - 需要处理 padded_action 和 padded_action_mask
      """

需要额外实现：
  - _prepare_packed_inputs_for_training(): 构建 BeingH 训练用的 packed 输入
  - get_optim_params(): 返回需要优化的参数（freeze VLM，只训练 expert + projection layers）

需要生成的测试：
  - tests/policies/beingh_koch/test_training.py：
    测试 forward 在 mock 数据上返回 loss 和 loss_dict
    测试 loss 为标量且可反向传播
    测试 freeze_vision_encoder=True 时 vit 参数不可训练
```

---

### TODO-13: 多任务推理与 Metadata Variant 支持

- **依赖项**：TODO-04（BeingHWrapper）
- **优先级**：🟢 低
- **prompt**：

```
支持 Being-H05 的多任务推理能力——根据 task description 自动选择对应的
归一化统计量（metadata variant），提高特定任务的推理精度。

需要修改的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\beingh_wrapper.py
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\configuration_beingh_koch.py

需要实现的功能：

  1. BeingHKochConfig 新增字段:
     metadata_variant: str | None = None
       # None: 自动选择（默认使用第一个 variant）
       # "merged": 使用合并统计量
       # "pick_orange_simple_shape": 使用特定任务的统计量
     stats_selection_mode: str = "auto"
       # "auto": 自动选择（优先 task-level）
       # "task": 使用 task-specific stats
       # "embodiment": 使用 embodiment-level stats

  2. BeingHWrapper.__init__ 中传递 metadata_variant 给 BeingHPolicy:
     BeingHPolicy(
       ...
       metadata_variant=config.metadata_variant,
       stats_selection_mode=config.stats_selection_mode,
     )

  3. 添加方法:
     def get_available_variants(self) -> list[str]:
         """返回 metadata 中可用的 variant 名称列表"""

设计注意点:
  - model-h05/koch_posttrain_metadata.json 中包含 10+ 个 task variants
  - 每个 variant 有独立的 statistics（mean/std/q01/q99）
  - 使用 task-specific stats 可能提高该特定任务的推理精度
  - 当前默认使用 merged stats（所有任务数据混合计算的统计量）

需要生成的测试：
  - tests/policies/beingh_koch/test_variant_support.py：
    测试 metadata_variant="merged" 正常加载
    测试 metadata_variant="pick_orange_simple_shape" 正常加载
    测试不存在的 variant 名称时 fallback 到默认
    测试 get_available_variants() 返回非空列表
```

---

### TODO-14: RTC（Real-Time Chunking）高级功能完善

- **依赖项**：TODO-05（BeingHKochPolicy）
- **优先级**：🟢 低
- **prompt**：

```
完善 RTC 推理的高级功能，包括可调节的 inference_delay、前缀锁定可视化等。

需要修改的文件：
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\modeling_beingh_koch.py
  - D:\study\code\lerobot\src\lerobot\policies\beingh_koch\configuration_beingh_koch.py

需要实现的功能：

  1. 可调节的 inference_delay:
     - BeingHKochConfig 新增: inference_delay: int = 5
     - 控制 RTC 前缀锁定的步数（前 N 步复用上一 chunk）

  2. prev_chunk 状态管理:
     - 在 BeingHKochPolicy 中维护 prev_chunk 缓存
     - 每次推理后更新 prev_chunk（取当前 chunk 的后 (chunk_len - delay) 步）

  3. RTC 步进逻辑:
     def _step_rtc_action(self, prev_chunk, delay):
         """
         1. 如果 delay >= chunk_size: 触发新推理
         2. 新推理时传入 prev_chunk + inference_delay
         3. BeingH 内部通过 prefix_locking 锁定前缀
         4. 返回新 chunk，更新 prev_chunk
         """

  4. RTC 性能指标:
     - 统计 RTC 命中率（复用前缀的比例）
     - 统计推理延迟（每次完整推理的耗时）
     - 提供 get_rtc_stats() 方法

需要生成的测试：
  - tests/policies/beingh_koch/test_rtc.py：
    测试 inference_delay=5 时前 5 步复用 prev_chunk
    测试 prev_chunk 在推理后正确更新
    测试 RTC 模式下 select_action 的行为正确
```

---

## TODO 执行顺序总结

```
阶段一（基础设施，无外部依赖）:
  ├─ ✅ TODO-01  Being-H05 源码路径配置与依赖验证
  └─ ✅ TODO-02  KochPosttrainDataConfig（Being-H05 侧）

阶段二（LeRobot 侧配置与模型封装，依赖阶段一）:
  ├─ ✅ TODO-03  BeingHKochConfig（LeRobot 配置类）
  ├─ ✅ TODO-04  BeingHWrapper（BeingHPolicy 封装）
  └─ ✅ TODO-05  BeingHKochPolicy（核心策略实现）

阶段三（Processor 层，依赖阶段一+二）:
  ├─ ✅ TODO-06  自定义 ProcessorStep 实现
  └─ ✅ TODO-07  Pre/Post Processor Pipeline 组装

阶段四（框架注册与集成，依赖阶段二+三）:
  ├─ ✅ TODO-08  LeRobot Factory 注册
  └─ ✅ TODO-09  CLI 推理启动脚本

阶段五（测试与验证）:
  ├─ ✅ TODO-10  单元测试套件（56 项，在实现过程中同步完成）
  └─ ⚠️ TODO-11  端到端集成测试（需 Linux/Triton 解决推理速度）

阶段六（扩展功能，按需实现）:
  ├─ □ TODO-12  训练支持（forward 实现）
  ├─ □ TODO-13  多任务推理与 Variant 支持
  └─ □ TODO-14  RTC 高级功能完善
```

> **Windows 推理验证结论**: 模型加载 ✅ | 代码流程 ✅ | 推理速度 ⚠️（需 Triton/FlashAttention，建议 Linux/WSL2）

---

## 优先级速查

| TODO | 名称 | 优先级 | 阶段 |
|------|------|--------|------|
| TODO-01 | 源码路径配置与依赖验证 | 🔴 高 | 一 |
| TODO-02 | KochPosttrainDataConfig | 🔴 高 | 一 |
| TODO-03 | BeingHKochConfig | 🔴 高 | 二 |
| TODO-04 | BeingHWrapper | 🔴 高 | 二 |
| TODO-05 | BeingHKochPolicy | 🔴 高 | 二 |
| TODO-06 | 自定义 ProcessorStep | 🔴 高 | 三 |
| TODO-07 | Processor Pipeline 组装 | 🔴 高 | 三 |
| TODO-08 | Factory 注册 | 🔴 高 | 四 |
| TODO-09 | CLI 推理脚本 | 🟡 中 | 四 |
| TODO-10 | 单元测试套件 | 🟡 中 | 五 |
| TODO-11 | 端到端集成测试 | 🟡 中 | 五 |
| TODO-12 | 训练支持 | 🟢 低 | 六 |
| TODO-13 | 多任务 Variant 支持 | 🟢 低 | 六 |
| TODO-14 | RTC 高级功能 | 🟢 低 | 六 |

> **最小可运行子集**（MVP）：完成 TODO-01 ~ TODO-08 即可通过 `lerobot-record --policy.type=beingh_koch` 运行推理。
