# Being-H05 VLA → LeRobot 适配方案与设计文档

## 一、概述

### 1.1 目标

将 Being-H05 VLA（Vision-Language-Action）模型的推理输出转换为 LeRobot 框架的标准格式，使 Being-H05 能够像 SmolVLA 一样作为 `policy.type=beingh_koch` 被 `lerobot-record` 脚本正常调用。

### 1.2 系统拓扑

```
┌────────────────────────────────────────────────────────────────────────┐
│                         lerobot-record                                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────┐  │
│  │ Koch     │   │ Observation  │   │ BeingH05     │   │ Action    │  │
│  │ Follower │──▶│ Preprocessor │──▶│ Policy       │──▶│ Post-     │  │
│  │ (COM6)   │   │              │   │ (VLA Model)  │   │ processor │  │
│  └──────────┘   └──────────────┘   └──────────────┘   └───────────┘  │
│       ▲                                                      │         │
│       │                                                      ▼         │
│       │              ┌──────────────┐              ┌──────────────┐  │
│       └──────────────┤ Robot.send   │◀─────────────┤ Robot Action │  │
│                      │ _action()    │              │ Processor    │  │
│                      └──────────────┘              └──────────────┘  │
│                                                         │              │
│                                                  ┌──────▼─────────┐  │
│                                                  │  Save to       │  │
│                                                  │  LeRobotDataset│  │
│                                                  └────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心挑战

Being-H05 模型输出的是 **200 维统一动作空间（Unified Action Space）**的归一化值，而 LeRobot 的 Koch 机器人期望的是 **6 个独立电机**（5 个关节 + 1 个夹爪）的绝对目标位置值。转换链路需要完成三个核心步骤：

1. **切片映射**：从 200 维中提取 Koch 相关的 6 维
2. **反归一化**：将归一化值（[-1,1] 或 z-score）还原为物理值
3. **格式转换**：输出 LeRobot 框架期望的 `RobotAction` 字典格式

---

## 二、数据流与格式分析

### 2.1 Being-H05 模型输出端

**模型架构关键参数**（来自 `model-h05/config.json`）：
| 参数 | 值 | 含义 |
|------|-----|------|
| `gen_action_type` | `prop_hidden` | 从 proprioception hidden states 解码动作 |
| `action_chunk_length` | 16 | 每次预测 16 步动作序列 |
| `use_flow_matching` | true | 使用 Flow Matching 扩散生成 |
| `num_inference_timesteps` | 4 | 扩散去噪步数 |
| `use_expert` | true | 使用 Action Expert（MoE 架构） |
| `unified_action_dim` | 200 | 统一动作空间维度 |

**BeingHPolicy.get_action() 输出格式**（来自 `Being-H05/BeingH/inference/beingh_policy.py`）：
```python
# 返回字典结构（key 为 "action.xxx"，value 为 numpy list）
result = {
    "action.arm_joint_position": [[j1, j2, j3, j4, j5], ...],  # (chunk_length, 5)
    "action.gripper_position":   [[g1], ...],                    # (chunk_length, 1)
    # RTC 模式额外返回：
    "action_unified": [[...200-dim...], ...],                    # (chunk_length, 200)
}
```

**内部处理流程**（`BeingHPolicy.get_action()` 的关键步骤）：
```
原始观测 (np arrays)
  │
  ▼
_modality_transform.apply()  ← 归一化 + 旋转转换
  │
  ▼
_prepare_packed_inputs()  ← 构建 packed sequence（vision tokens + state + prompt）
  │
  ▼
model.get_action()  ← LLM + Flow Matching 推理
  │  返回: action_pred: (B * chunk_length, unified_action_dim=200)
  │
  ▼
unified_mapping 切片  ← 从 200 维中切出各 action 分量
  │  e.g. action.arm_joint_position = action_pred[:, 50:55]
  │       action.gripper_position   = action_pred[:, 18:19]
  │
  ▼
_modality_transform.unapply()  ← 反归一化
  │
  ▼
最终动作字典 (物理单位，list 格式)
```

### 2.2 LeRobot 机器人动作端

**KochFollower 动作空间**（来自 `src/lerobot/robots/koch_follower/koch_follower.py`）：

| 电机名 | ID | 型号 | 归一化模式 | 范围 |
|--------|-----|------|-----------|------|
| shoulder_pan | 1 | xl430-w250 | RANGE_M100_100 | [-100, 100] |
| shoulder_lift | 2 | xl430-w250 | RANGE_M100_100 | [-100, 100] |
| elbow_flex | 3 | xl330-m288 | RANGE_M100_100 | [-100, 100] |
| wrist_flex | 4 | xl330-m288 | RANGE_M100_100 | [-100, 100] |
| wrist_roll | 5 | xl330-m288 | RANGE_M100_100 | [-100, 100] |
| gripper | 6 | xl330-m288 | RANGE_0_100 | [0, 100] |

**LeRobot 期望的动作字典格式**：
```python
action = {
    "shoulder_pan.pos": 12.5,    # float, 绝对目标位置
    "shoulder_lift.pos": -45.3,
    "elbow_flex.pos": 30.1,
    "wrist_flex.pos": -60.2,
    "wrist_roll.pos": 5.0,
    "gripper.pos": 45.0,
}
```

### 2.3 SmolVLA 参考模式

SmolVLA 集成是 LeRobot 框架中原生的 VLA 策略实现，作为本方案的设计参考：

```
SmolVLA 文件结构：
src/lerobot/policies/smolvla/
├── configuration_smolvla.py   ← PreTrainedConfig 子类，@register_subclass("smolvla")
├── modeling_smolvla.py       ← PreTrainedPolicy 子类，实现 select_action / forward
├── processor_smolvla.py      ← make_smolvla_pre_post_processors()
├── smolvlm_with_expert.py    ← 底层模型实现
└── README.md

关键注册机制：
1. configuration_smolvla.py: @PreTrainedConfig.register_subclass("smolvla")
2. modeling_smolvla.py:       SmolVLAPolicy(PreTrainedPolicy), name = "smolvla"
3. factory.py:                get_policy_class("smolvla") → SmolVLAPolicy
4. processor_smolvla.py:      make_smolvla_pre_post_processors()
```

---

## 三、统一动作空间映射设计

### 3.1 Being-H05 200 维统一动作空间

Being-H05 使用 200 维统一动作空间实现跨具身泛化。与 Koch 相关的槽位如下：

| 语义槽位 | 维度范围 | 维度数 | 说明 |
|---------|---------|--------|------|
| eef_position | [0, 3) | 3 | 右臂末端位置（Koch 不使用） |
| eef_rotation | [3, 6) | 3 | 右臂末端旋转（Koch 不使用） |
| gripper_position | [18, 19) | 1 | 右夹爪位置 |
| arm_joint_position | [50, 57) | 7 | 右臂关节位置（Koch 只用前 5 维） |

### 3.2 Koch 数据配置的 UNIFIED_MAPPING

需要在 Being-H05 的 `DATA_CONFIG_MAP` 中新增 `koch_posttrain` 配置（或通过策略内部直接定义映射）：

```
KochPosttrainDataConfig.UNIFIED_MAPPING = {
    # 状态（观测）
    "state.arm_joint_position":  (50, 55),   # 5-dof 关节 → 统一空间 [50:55)
    "state.gripper_position":    (18, 19),   # 1-dof 夹爪 → 统一空间 [18:19)

    # 动作
    "action.arm_joint_position": (50, 55),   # 5-dof 关节命令
    "action.gripper_position":   (18, 19),   # 1-dof 夹爪命令
}
```

### 3.3 Koch 电机名 ↔ 5维关节向量的对应关系

5 维关节向量按顺序对应电机：

| 索引 | 电机名 | Being-H05 key |
|------|--------|---------------|
| 0 | shoulder_pan | action.arm_joint_position[0] |
| 1 | shoulder_lift | action.arm_joint_position[1] |
| 2 | elbow_flex | action.arm_joint_position[2] |
| 3 | wrist_flex | action.arm_joint_position[3] |
| 4 | wrist_roll | action.arm_joint_position[4] |
| — | gripper | action.gripper_position[0] |

---

## 四、整体架构设计

### 4.1 文件结构

```
src/lerobot/policies/beingh_koch/          ← 新增目录
├── __init__.py
├── configuration_beingh_koch.py           ← BeingHKochConfig (PreTrainedConfig 子类)
├── modeling_beingh_koch.py                ← BeingHKochPolicy (PreTrainedPolicy 子类)
├── processor_beingh_koch.py               ← make_beingh_koch_pre_post_processors()
├── beingh_wrapper.py                      ← BeingH05 模型的薄封装层
└── koch_data_config.py                    ← Koch 专有 DataConfig（UNIFIED_MAPPING 等）

lerobot_record_vla.py                      ← 项目根目录（推理启动脚本，可选）
```

### 4.2 模块职责

| 模块 | 职责 | 依赖 |
|------|------|------|
| `configuration_beingh_koch.py` | 定义策略配置类，注册 `beingh_koch` 类型 | `PreTrainedConfig` |
| `modeling_beingh_koch.py` | 实现 `select_action` / `predict_action_chunk`，对接 LeRobot 调用链 | `PreTrainedPolicy` |
| `processor_beingh_koch.py` | 定义预处理器（观测→模型输入）和后处理器（模型输出→LeRobot 动作） | `PolicyProcessorPipeline` |
| `beingh_wrapper.py` | 封装 `BeingHPolicy` 的初始化和调用，处理模型加载、tokenizer 等 | `BeingHPolicy` |
| `koch_data_config.py` | 定义 Koch 机器人的观测/动作到 200 维统一空间的映射 | Being-H05 `BaseDataConfig` |

### 4.3 与 LeRobot 框架的注册关系

```
PreTrainedConfig.register_subclass("beingh_koch")
         │
         ▼
BeingHKochConfig  ──extends──▶  PreTrainedConfig
         │
         ▼
BeingHKochPolicy  ──extends──▶  PreTrainedPolicy
  .config_class = BeingHKochConfig
  .name = "beingh_koch"
         │
         ▼
make_policy("beingh_koch")      ← 由 factory.py 动态发现
make_pre_post_processors()      ← 由 factory.py 根据 config.type 匹配
```

---

## 五、核心接口设计

### 5.1 BeingHKochConfig（策略配置）

继承 `PreTrainedConfig`，定义策略的输入/输出特征签名和模型参数。

```
类：BeingHKochConfig(PreTrainedConfig)
注册：@PreTrainedConfig.register_subclass("beingh_koch")

关键字段：
  # ── 模型路径 ──
  pretrained_path: str                 # Being-H05 模型检查点路径（如 D:/study/code/lerobot/model-h05）

  # ── 数据配置 ──
  data_config_name: str = "koch_posttrain"  # Being-H05 DataConfig 名称
  dataset_name: str = "koch_posttrain"       # 数据集名称（匹配 metadata.json 中的 key）
  embodiment_tag: str = "koch"                # 具身标签

  # ── 相机配置 ──
  video_keys: list[str]                  # 相机视图 key，如 ["video.above", "video.wrist"]
                                         # above=顶部相机, wrist=臂上相机

  # ── 观测/动作维度 ──
  n_obs_steps: int = 1                   # 观测历史步数
  chunk_size: int = 16                   # 动作块长度（与 action_chunk_length 一致）
  n_action_steps: int = 16               # 每次执行的动作步数

  # ── 归一化 ──
  normalization_mapping: dict            # 类似 SmolVLA，定义各模态的归一化方式
    = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    }

  # ── 推理参数 ──
  enable_rtc: bool = True                # 是否启用 Real-Time Chunking
  num_inference_timesteps: int | None = None  # Flow Matching 去噪步数（None=用模型默认值）
  use_mpg: bool | None = None            # 是否启用 MPG 增强（None=用模型默认值）
  device: str = "cuda"

  # ── 语言指令模板 ──
  instruction_template: str
    = "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"

  # ── 图像预处理 ──
  force_image_size: int = 224            # ViT 输入分辨率
  pad2square: bool = False               # 是否填充为正方形

  # ── 训练参数（保留用于后续微调扩展）──
  freeze_vision_encoder: bool = True
  train_expert_only: bool = True
```

### 5.2 BeingHKochPolicy（策略模型）

继承 `PreTrainedPolicy`，实现 `select_action` 接口供 `record_loop` 调用。

```
类：BeingHKochPolicy(PreTrainedPolicy)

核心属性：
  config_class = BeingHKochConfig
  name = "beingh_koch"

核心方法：
  ┌─────────────────────────────────────────────────────────────────┐
  │ def __init__(self, config: BeingHKochConfig)                      │
  │   1. 调用 super().__init__(config)                                │
  │   2. 初始化 BeingHWrapper（模型加载 + tokenizer）                  │
  │   3. 初始化 _queues（动作队列，实现 chunk-wise 调度）              │
  │   4. 初始化 metadata 和 statistics                                │
  │   5. 设置 self.reset()                                            │
  ├─────────────────────────────────────────────────────────────────┤
  │ def reset(self)                                                   │
  │   清空动作队列，重置推理状态                                       │
  ├─────────────────────────────────────────────────────────────────┤
  │ def select_action(self, batch: dict[str, Tensor]) -> Tensor       │
  │   1. 从 _queues 取出单步动作（如有缓存）                           │
  │   2. 若队列空，调用 _get_action_chunk(batch) 获取新动作块          │
  │   3. 将动作块 (1, chunk_size, 6) 转置后入队                        │
  │   4. 返回最左侧单步动作 (6,)                                      │
  ├─────────────────────────────────────────────────────────────────┤
  │ def _get_action_chunk(self, batch) -> Tensor                       │
  │   1. prepare_observations(batch) → BeingH 所需的观测字典           │
  │      - 提取图像: batch["observation.images.above"] → np array     │
  │      - 提取状态: batch["observation.state"] → 6-dim → 分配进200维  │
  │      - 提取语言: batch["task"] → string                           │
  │   2. beingh_policy.get_action(observations) → action_dict          │
  │   3. action_dict_to_tensor(action_dict) → (chunk_size, 6)         │
  │   4. 返回 Tensor (1, chunk_size, 6)                               │
  ├─────────────────────────────────────────────────────────────────┤
  │ def forward(self, batch) → dict[str, Tensor]                      │
  │   训练前向传播（后续微调时实现）                                    │
  └─────────────────────────────────────────────────────────────────┘

select_action 工作流程：
  ┌─────────────────────────────────────────────────────────────────┐
  │ batch (LeRobot 格式)                                             │
  │ {                                                                │
  │   "observation.images.above": Tensor (1, 480, 640, 3),          │
  │   "observation.images.wrist": Tensor (1, 480, 640, 3),          │
  │   "observation.state": Tensor (1, 6),  ← [sp, sl, ef, wf, wr, g]│
  │   "task": "pick the green toy and place into box"               │
  │ }                                                                │
  │         │                                                        │
  │         ▼ prepare_observations()                                 │
  │ {                                                                │
  │   "video.above": np.ndarray (480, 640, 3),  ← uint8 RGB         │
  │   "video.wrist": np.ndarray (480, 640, 3),  ← uint8 RGB         │
  │   "state.arm_joint_position": np.ndarray (5,),                   │
  │   "state.gripper_position": np.ndarray (1,),                     │
  │   "language.instruction": ["pick the green toy and place..."]    │
  │ }                                                                │
  │         │                                                        │
  │         ▼ BeingHPolicy.get_action()                              │
  │ {                                                                │
  │   "action.arm_joint_position": [[j1,...,j5], ...],  ← 16 steps  │
  │   "action.gripper_position": [[g1], ...]                         │
  │ }                                                                │
  │         │                                                        │
  │         ▼ action_dict_to_tensor()                                │
  │ Tensor (16, 6)  ← 拼接 arm_joint + gripper                      │
  │ 列顺序: [shoulder_pan, shoulder_lift, elbow_flex,                 │
  │          wrist_flex, wrist_roll, gripper]                        │
  └─────────────────────────────────────────────────────────────────┘
```

### 5.3 Processor Pipeline（预处理器与后处理器）

#### 预处理器（Preprocessor）：dict → dict

```
Pipeline 名称: "policy_preprocessor"

步骤序列：
  Step 1: RenameObservationsProcessorStep
    功能：将 LeRobot 的标准键名映射为 BeingH 需要的键名
    配置：
      rename_map = {
          "observation.images.above": "observation.images.above",
          "observation.images.wrist": "observation.images.wrist",
          "observation.state": "observation.state",
      }

  Step 2: BeingHStatePaddingProcessorStep
    功能：将 Koch 6 维状态向量填充/映射为 BeingH 200 维统一状态向量
    输入：observation.state: Tensor (1, 6)
    处理：
      - state[0] → shoulder_pan  → unified_state[50]
      - state[1] → shoulder_lift → unified_state[51]
      - state[2] → elbow_flex    → unified_state[52]
      - state[3] → wrist_flex    → unified_state[53]
      - state[4] → wrist_roll    → unified_state[54]
      - state[5] → gripper       → unified_state[18]
    输出：observation.state: Tensor (1, 200)

  Step 3: BeingHImagePreprocessorStep
    功能：图像格式转换（numpy → PIL Image）、resize 到 force_image_size
    输入：observation.images.{key}: Tensor (B, H, W, 3) uint8
    处理：转为 numpy → PIL Image → resize(224, 224) → 保持为 numpy 数组
    输出：video.{key}: np.ndarray (224, 224, 3) uint8

  Step 4: AddBatchDimensionProcessorStep (继承现有)
    功能：添加 batch 维度

  Step 5: DeviceProcessorStep (继承现有)
    功能：移动数据到指定设备
```

#### 后处理器（Postprocessor）：PolicyAction → PolicyAction

```
Pipeline 名称: "policy_postprocessor"

步骤序列：
  Step 1: BeingHActionUnsliceProcessorStep
    功能：从 BeingHPolicy 返回的 action_dict 中提取并拼接 Koch 6 维动作向量
    输入（PolicyAction 格式）：
      包含 "action.arm_joint_position" (5-d) 和 "action.gripper_position" (1-d)
    处理：
      koch_action = concat(action.arm_joint_position, action.gripper_position)
      即 Tensor (chunk_size, 6)，列序 = [sp, sl, ef, wf, wr, gripper]
    输出：PolicyAction，action 张量 shape 为 (chunk_size, 6)

  Step 2: BeingHActionClampProcessorStep（可选）
    功能：将动作值 clamp 到 Koch 电机的有效范围内
    范围：
      joints: [-100, 100]   (RANGE_M100_100 归一化模式的原始范围)
      gripper: [0, 100]     (RANGE_0_100)

  Step 3: UnnormalizerProcessorStep (继承现有)
    功能：反归一化（如需要，通常 BeingH 已在内部完成反归一化）

  Step 4: DeviceProcessorStep (继承现有)
    功能：数据移至 CPU
```

### 5.4 BeingHWrapper（模型封装）

```
类：BeingHWrapper

职责：封装 BeingHPolicy 的初始化、推理、元数据加载

属性：
  beingh_policy: BeingHPolicy           # Being-H05 推理实例
  model_path: str                       # 模型检查点路径
  data_config_name: str                 # 数据配置名称
  dataset_name: str                     # 数据集名称
  embodiment_tag: str                   # 具身标签
  metadata: DatasetMetadata             # 归一化统计信息
  unified_mapping: dict[str, tuple]     # 统一空间映射

方法：
  __init__(config: BeingHKochConfig)
    1. 验证 model_path 存在且包含必要文件
    2. 创建 BeingHPolicy(
         model_path=config.pretrained_path,
         data_config_name=config.data_config_name,      # "koch_posttrain"
         dataset_name=config.dataset_name,
         embodiment_tag=config.embodiment_tag,
         instruction_template=config.instruction_template,
         enable_rtc=config.enable_rtc,
         ...
       )
    3. 保存 unified_mapping 和 metadata 引用

  get_action(observations: dict) → dict[str, list]
    直接调用 self.beingh_policy.get_action(observations)

  get_unified_mapping() → dict[str, tuple[int, int]]
    返回 koch 的 UNIFIED_MAPPING

  close()
    清理资源
```

### 5.5 KochDataConfig（数据配置）

```
类：KochPosttrainDataConfig(BaseDataConfig)

职责：定义 Koch 机器人的观测/动作在 200 维统一空间中的映射关系

VIDEO_KEYS = ["video.above", "video.wrist"]
VIDEO_SOURCE_COLUMNS = {
    "video.above": "observation.images.above",
    "video.wrist": "observation.images.wrist",
}

STATE_KEYS = ["state.arm_joint_position", "state.gripper_position"]
ACTION_KEYS = ["action.arm_joint_position", "action.gripper_position"]

LANGUAGE_KEYS = ["language.instruction"]

UNIFIED_MAPPING = {
    # State
    "state.arm_joint_position":  (50, 55),   # 5-dof
    "state.gripper_position":    (18, 19),   # 1-dof

    # Action
    "action.arm_joint_position": (50, 55),   # 5-dof
    "action.gripper_position":   (18, 19),   # 1-dof
}

state_normalization_modes = {
    "state.arm_joint_position": "mean_std",
    "state.gripper_position": "mean_std",
}
action_normalization_modes = {
    "action.arm_joint_position": "mean_std",
    "action.gripper_position": "mean_std",
}
```

**需要注册到 `DATA_CONFIG_MAP`**：
```python
# 在 Being-H05/configs/data_config.py 中新增
DATA_CONFIG_MAP["koch_posttrain"] = KochPosttrainDataConfig
```

---

## 六、推理流程设计

### 6.1 整体时序图

```
lerobot-record                    BeingHKochPolicy              BeingHWrapper         BeingHPolicy
     │                                  │                            │                    │
     │  record_loop()                   │                            │                    │
     │──predict_action(obs)────────────▶│                            │                    │
     │                                  │                            │                    │
     │                                  │──preprocessor(obs)─────────│                    │
     │                                  │  (rename, state_pad,       │                    │
     │                                  │   image_preprocess)        │                    │
     │                                  │                            │                    │
     │                                  │──select_action(batch)──────│                    │
     │                                  │                            │                    │
     │                                  │  queue empty?              │                    │
     │                                  │──_get_action_chunk(batch)─▶│                    │
     │                                  │                            │                    │
     │                                  │                            │──prepare_obs()────▶│
     │                                  │                            │                    │
     │                                  │                            │◀──get_action(obs)──│
     │                                  │                            │  (200-dim unified) │
     │                                  │                            │                    │
     │                                  │◀──action_dict──────────────│                    │
     │                                  │  {action.arm_joint: [[],   │                    │
     │                                  │   action.gripper: [[]]}    │                    │
     │                                  │                            │                    │
     │                                  │──dict_to_tensor()──▶       │                    │
     │                                  │  shape: (chunk, 6)         │                    │
     │                                  │                            │                    │
     │                                  │──入队 _queues[ACTION]      │                    │
     │                                  │                            │                    │
     │                                  │◀──popleft() 单步动作────   │                    │
     │                                  │                            │                    │
     │◀──action_values (6,)─────────────│                            │                    │
     │                                  │                            │                    │
     │──postprocessor(action)──────────▶│                            │                    │
     │──make_robot_action()────────────▶│                            │                    │
     │  → {motor.pos: val, ...}         │                            │                    │
     │──robot.send_action()────────────▶│                            │                    │
```

### 6.2 关键数据转换节点

#### 节点 A：LeRobot 观测 → BeingH 观测

```
输入（来自 robot.get_observation() → robot_observation_processor）：
{
  "observation.images.above": np.ndarray (480, 640, 3) uint8,
  "observation.images.wrist": np.ndarray (480, 640, 3) uint8,
  "observation.state": np.ndarray (6,) float32,
      # [shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
      #  wrist_flex.pos, wrist_roll.pos, gripper.pos]
}

经过预处理后 → BeingH 格式：
{
  "video.above": np.ndarray (224, 224, 3) uint8,     ← resize 后
  "video.wrist": np.ndarray (224, 224, 3) uint8,     ← resize 后
  "state.arm_joint_position": np.ndarray (5,),        ← 切片
  "state.gripper_position": np.ndarray (1,),           ← 切片
  "language.instruction": ["pick the green toy..."],   ← list[str]
}
```

#### 节点 B：BeingH 模型推理

```
BeingH 内部：
  _modality_transform.apply(obs)
    → StateActionToTensor: numpy → torch
    → StateActionTransform: normalize (mean_std)
    → 构建 unified_state (200-dim zeros)
    → 填充: unified_state[50:55] = state.arm_joint_position (norm)
    → 填充: unified_state[18:19] = state.gripper_position (norm)

  _prepare_packed_inputs()
    → 构建 vision tokens (InternViT)
    → 构建 text tokens (Qwen3 tokenizer)
    → 构建 packed sequence

  model.get_action()
    → Flow Matching 扩散去噪 (4 steps)
    → 输出: action_pred (1 * 16, 200)  ← normalized unified actions

  _modality_transform.unapply(action_dict)
    → 切片: action.arm_joint = action_pred[50:55]
    → 切片: action.gripper = action_pred[18:19]
    → 反归一化: 从 mean_std 区间还原
    → tensor → numpy → list
```

#### 节点 C：BeingH 动作 → LeRobot 动作

```
BeingH 输出（物理值，numpy）：
{
  "action.arm_joint_position": ndarray (16, 5),
  "action.gripper_position": ndarray (16, 1),
}

在 BeingHKochPolicy._get_action_chunk() 中拼接：
  action_chunk = concat(..., axis=-1)  → (1, 16, 6)
  # [sp, sl, ef, wf, wr, gripper]

在 select_action() 中调度：
  self._queues[ACTION].extend(action_chunk.transpose(0, 1))
  # shape (16, 1, 6) → 逐个 (1, 6) 出队
  return self._queues[ACTION].popleft()  → (6,)

在 record_loop() 中：
  make_robot_action(action_values, dataset.features) → RobotAction:
  {
    "shoulder_pan.pos": 12.5,
    "shoulder_lift.pos": -45.3,
    "elbow_flex.pos": 30.1,
    "wrist_flex.pos": -60.2,
    "wrist_roll.pos": 5.0,
    "gripper.pos": 45.0,
  }
```

### 6.3 RTC（Real-Time Chunking）流程

当 `enable_rtc=True` 时：

```
T=0:  观测 → 模型推理 → action_chunk[0:16] → 执行 action[0]
                                              保留 action[1:16] 作为 prev_chunk
T=1:  观测 → 模型推理 (带 prev_chunk, inference_delay=1)
              → 模型锁定前 1 步（使用 prev_chunk 的前缀）
              → 去噪后 15 步 → action_chunk[1:16]
              → 执行 action[1]
              → prev_chunk = action[2:16]
T=2:  观测 → ... (依此类推)

当 prev_chunk 耗尽时（inference_delay >= chunk_size），触发新一轮完整推理。
```

---

## 七、与 LeRobot 框架的集成点

### 7.1 factory.py 修改

`src/lerobot/policies/factory.py` 中新增：

```python
# get_policy_class() 中新增
elif name == "beingh_koch":
    from lerobot.policies.beingh_koch.modeling_beingh_koch import BeingHKochPolicy
    return BeingHKochPolicy

# make_policy_config() 中新增
elif policy_type == "beingh_koch":
    return BeingHKochConfig(**kwargs)

# make_pre_post_processors() 中新增（如果使用动态发现则不需要）
elif isinstance(policy_cfg, BeingHKochConfig):
    from lerobot.policies.beingh_koch.processor_beingh_koch import (
        make_beingh_koch_pre_post_processors,
    )
    processors = make_beingh_koch_pre_post_processors(...)
```

实际上，由于 `factory.py` 中的 `_get_policy_cls_from_policy_name()` 和 `_make_processors_from_policy_config()` 已支持基于 `PreTrainedConfig.get_known_choices()` 的动态发现，只需要：
1. `BeingHKochConfig` 用 `@PreTrainedConfig.register_subclass("beingh_koch")` 注册
2. `BeingHKochPolicy` 放在 `modeling_beingh_koch.py` 中
3. `make_beingh_koch_pre_post_processors` 放在 `processor_beingh_koch.py` 中

框架会自动通过 importlib 加载。

### 7.2 lerobot_record.py 集成

不需要修改 `lerobot_record.py` 本身。通过 CLI 参数即可使用：

```bash
lerobot-record \
  --robot.type=koch_follower \
  --robot.port=COM6 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{
    above: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
  }" \
  --display_data=true \
  --dataset.repo_id=a/b \
  --dataset.root='d:/dataset/test' \
  --dataset.single_task="pick the green toy and place into box" \
  --policy.pretrained_path='D:/study/code/lerobot/model-h05' \
  --policy.type=beingh_koch \
  --policy.device='cuda'
```

### 7.3 可选：独立推理脚本

提供 `lerobot_record_vla.py`（项目根目录）作为独立推理脚本，支持不通过 lerobot-record 直接运行 Being-H05 推理：

```
功能：
  - 独立连接 Koch 机器人和相机
  - 加载 Being-H05 模型
  - 循环推理并控制机器人
  - 可选：记录数据集

与 lerobot-record 的区别：
  - 不依赖 lerobot-record 的完整框架
  - 更轻量、更灵活
  - 适合快速测试和调试
```

---

## 八、关键设计决策与依据

### 8.1 为什么选择在 Policy 层做适配而非在 Processor 层

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Policy 层适配**（本方案） | 复用 BeingHPolicy 的归一化/反归一化逻辑；图像预处理、tokenization 在 BeingH 内部完成，保持一致性 | Policy 类较重 |
| Processor 层适配 | Processor 逻辑简洁 | 需要重新实现 BeingH 的预处理/后处理逻辑，容易与训练时不一致 |

**结论**：采用 Policy 层适配，通过 `BeingHWrapper` 封装 `BeingHPolicy`，保留 Being-H05 原有的完整推理管线。

### 8.2 为什么使用 `select_action` 而非 `predict_action_chunk`

LeRobot 的 `record_loop()` 通过 `predict_action()` → `policy.select_action()` 调用链逐步获取动作。这与 Being-H05 的 chunk-wise 推理模式兼容：

```python
# predict_action() 中的核心调用
action_values = policy.select_action(batch)
```

`select_action` 内部维护动作队列，当队列为空时触发一次完整的 chunk 推理，然后逐步从队列中弹出单步动作。这与 SmolVLA 的设计完全一致。

### 8.3 状态向量映射方向

Koch 的 6 维状态（5 关节 + 1 夹爪）需要映射到 BeingH 的 200 维统一状态空间：
- 关节值 → `unified_state[50:55]`
- 夹爪值 → `unified_state[18]`
- 其余 194 维填 0

注意：这些值在传入 BeingHPolicy 后会由 `_modality_transform.apply()` 自动归一化，因此传入的应当是原始物理值（单位：Dynamixel 电机位置值 [-100,100] / [0,100]），而非提前归一化后的值。

### 8.4 相机 key 的命名约定

Being-H05 后训练数据使用 `video.above`（顶部相机）和 `video.wrist`（臂上相机）作为视频 key。LeRobot 的 Koch 配置通常使用 `cameras` 的子配置来命名相机。需要通过 `rename_map` 建立映射：

```
LeRobot 相机 key          →  BeingH 视频 key
observation.images.top   →  video.above   (顶部 Orbbec 相机)
observation.images.wrist →  video.wrist   (臂上 USB 相机)
```

`rename_map` 在 `RecordConfig.__post_init__` 中通过 CLI 参数 `--dataset.rename_map` 配置。

---

## 九、实现步骤

### Phase 1：基础设施层
1. ✅ 了解 Being-H05 模型结构和推理流程
2. ✅ 了解 LeRobot SmolVLA 集成模式
3. 创建 `KochPosttrainDataConfig` 并注册到 `DATA_CONFIG_MAP`
4. 创建 `BeingHKochConfig` 配置类
5. 更新 `model-h05/metadata.json` 中的 `koch_posttrain` 数据集

### Phase 2：模型封装层
6. 实现 `BeingHWrapper`，封装 BeingHPolicy 初始化
7. 实现 `BeingHKochPolicy`：
   - 实现 `select_action` 和 `_get_action_chunk`
   - 实现观测数据格式转换（LeRobot → BeingH）
   - 实现动作数据格式转换（BeingH → LeRobot）

### Phase 3：处理器层
8. 实现 `make_beingh_koch_pre_post_processors()`
9. 实现自定义 ProcessorStep：
   - `BeingHStatePaddingProcessorStep`
   - `BeingHImagePreprocessorStep`
   - `BeingHActionUnsliceProcessorStep`

### Phase 4：集成与测试
10. 在 `factory.py` 中（或通过动态发现）注册 policy
11. 编写并执行 CLI 推理命令
12. 端到端测试：模型加载 → 推理 → 动作执行 → 数据集保存

---

## 十、风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| BeingH 模型加载依赖复杂（vit_model, llm_model, connector 等） | 初始化失败 | 保持模型文件完整性，使用 safetensors 格式，确保所有依赖项在 conda 环境中 |
| 统一空间映射维度不匹配 | 推理输出维度错误 | 明确验证 Koch 的 arm_joint_position 使用 unified 空间的 [50:55) 范围 |
| 归一化统计数据不一致 | 动作值域异常 | 使用 `model-h05/koch_posttrain_metadata.json` 中的 statistics，确保 `gen_action_type=prop_hidden` 时的输出一致性 |
| LeRobot 的 `make_robot_action` 期望特定 key 名 | 动作无法发送给机器人 | 确保后处理器输出的 key 名与 `dataset.features` 一致，对应 Koch motor 名 |
| RTC 模式需要 `prev_chunk` 和 `inference_delay` | RTC 推理出错 | 在 `BeingHKochPolicy` 中正确管理 prev_chunk 状态 |
| 图像格式/尺寸差异 | ViT 推理失败 | 统一使用 `force_image_size=224`，pad2square=False（metadata 中视频为 640×480） |
| Qwen3 tokenizer 需要 trust_remote_code | 加载失败 | tokenizer 文件已在 model-h05 目录中，使用本地路径加载 |

---

## 附录 A：Being-H05 模型完整文件清单

```
model-h05/
├── config.json                    ← BeingH 模型配置
├── model.safetensors              ← 模型权重
├── koch_posttrain_metadata.json   ← Koch 后训练元数据（归一化统计 + 模态定义）
├── tokenizer.json                 ← Qwen3 tokenizer
├── tokenizer_config.json          ← tokenizer 配置
├── vocab.json                     ← 词表
├── merges.txt                     ← BPE merges
├── special_tokens_map.json        ← 特殊 token 映射
├── added_tokens.json              ← 新增 token
└── chat_template.jinja            ← 对话模板
```

## 附录 B：LeRobot 策略注册机制

```
1. PreTrainedConfig.register_subclass("beingh_koch")  ← 类装饰器
   → PreTrainedConfig._choice_classes["beingh_koch"] = BeingHKochConfig

2. BeingHKochPolicy.name = "beingh_koch"
   → 通过 PreTrainedPolicy 的 from_pretrained() 可以自动加载

3. factory._get_policy_cls_from_policy_name("beingh_koch")
   → 从 BeingHKochConfig.__module__ 推导 modeling 路径
   → importlib.import_module() 动态加载 BeingHKochPolicy

4. factory._make_processors_from_policy_config(config)
   → 从 BeingHKochConfig.__module__ 推导 processor 路径
   → importlib.import_module() 动态加载 make_beingh_koch_pre_post_processors
```

## 附录 C：BeHVLA 与 SmolVLA 架构对比

| 特性 | SmolVLA | Being-H05 (BeHVLA) |
|------|---------|---------------------|
| VLM 基座 | SmolVLM2-500M | Qwen3-1.7B + InternViT-6B |
| Action Expert | Cross-Attention | MoE (Mixture of Experts) |
| 动作生成 | Flow Matching | Flow Matching (4 steps) |
| 动作空间 | max_action_dim=32 | unified_action_dim=200 |
| 状态空间 | max_state_dim=32 | unified_state_dim=200 |
| RTC 支持 | 可选（RTCProcessor） | 可选（prefix locking） |
| Chunk 长度 | 50（可配） | 16（固定） |
| 图像编码 | SigLIP | InternViT + Pixel Shuffle |
| 系统设计 | 模型内置于策略 | 策略封装推理服务 |
