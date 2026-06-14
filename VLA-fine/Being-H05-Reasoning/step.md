# 实现进度记录 — Being-H05 VLA → LeRobot 适配

## 总体状态

- **MVP 代码**: ✅ 全部完成 (TODO-01 ~ TODO-09)
- **单元测试**: 56/56 通过 ✅
- **模型加载**: ✅ (CUDA, ~40s, 7.6GB VRAM)
- **推理流程**: ✅ (完整执行路径验证通过)
- **真实推理速度**: ⚠️ Windows eager SDPA 过慢 (需 Linux/Triton/FlashAttention)
- **真实机械臂**: 待验证 (需解决推理速度问题)

---

## 阶段一~四: MVP 核心功能 ✅

| TODO | 内容 | 测试 |
|------|------|------|
| 01 | 源码路径 + 依赖 | 9/9 |
| 02 | KochPosttrainDataConfig | 11/11 |
| 03 | BeingHKochConfig | 10/10 |
| 04 | BeingHWrapper | 10/10 |
| 05 | BeingHKochPolicy | 8/8 |
| 06 | ProcessorStep ×3 | 8/8 |
| 07 | Pipeline 组装 | - |
| 08 | Factory 注册 | - |
| 09 | CLI 脚本 | - |

---

## 真实推理验证

### 硬件环境
- GPU: NVIDIA RTX 5070 Laptop (8GB)
- CPU: 推理可用但极慢
- OS: Windows 11
- 机械臂: COM6 (Koch Follower, CH343)
- 顶部相机: index 1 (480×640)
- 臂上相机: index 4 (480×640)

### 适配问题及修复（详见 Debug.md）
1. Windows GBK 编码 → Unicode→ASCII
2. Triton 不可用 → torch.compile try/except
3. create_block_mask OOM (56 GiB) → 绕过
4. flex_attention API → SDPA + is_causal=True
5. GPU 显存不足 + eager SDPA 慢 → 平台限制

### 推理速度结论
- Being-H05 (Qwen3-1.7B + InternViT-6B) 需要 FlashAttention/Triton 优化推理
- Windows 环境只能使用 eager SDPA，packed sequence 推理速度不可用
- **推荐**: 使用 WSL2/Linux + FlashAttention2，或 GPU ≥12GB

---

## 待完成

| TODO | 名称 | 优先级 | 备注 |
|------|------|--------|------|
| 11 | 端到端集成测试 | 🟡 | 需解决推理速度问题后测试 |
| 12 | 训练支持 | 🟢 | |
| 13 | Variant 支持 | 🟢 | |
| 14 | RTC 高级功能 | 🟢 | |
