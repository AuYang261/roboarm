# Roboarm Grasp Skill

YOLO + LLM-based object detection and robotic arm grasping skill.

## Tools

- `detect_and_grasp(instruction: str)` — MCP tool (LLM-based)
  - Input: natural language instruction (e.g. "抓取红色积木")
  - Output: JSON with status, target object, and grasp result
  - Internal flow: capture frame → LLM detection → arm grasp → place
  - Corresponds to `llm/catch_by_llm.py`

- `classify_and_grasp(repeat: int)` — MCP tool (YOLO-based)
  - Input: repeat count (default 1)
  - Output: JSON with per-round detection and grasp results
  - Internal flow: capture frame → YOLO detect all objects → grasp each → place by class
  - Corresponds to `classification/catch_with_arm.py`

- `move_home()` — MCP tool
  - Return arm to home position with gripper open

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| ROBOARM_PATH | /home/xjy/roboarm | Path to the roboarm project root |
| ROBONIX_ATLAS | 127.0.0.1:50051 | Atlas control plane address |

## Dependencies

- roboarm project with config.yaml properly configured
- Camera (Orbbec or USB) connected
- Robotic arm (Lerobot or Piper) connected
