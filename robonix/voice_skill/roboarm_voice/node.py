"""
Voice Recognition Skill Node

Wraps the roboarm audio capture + whisper transcription pipeline
as a Robonix Skill with MCP tools.

Tools:
  - listen(duration) — record audio for N seconds and return transcribed text
"""

import json
import os
import sys
import threading
import time
import traceback

# -- Add roboarm project to Python path --
_ROBOARM_PATH = os.environ.get("ROBOARM_PATH")
if not _ROBOARM_PATH:
    raise ValueError("Set ROBOARM_PATH as root of roboarm project")
if _ROBOARM_PATH not in sys.path:
    sys.path.insert(0, _ROBOARM_PATH)

from mcp.server.fastmcp import FastMCP
from robonix_api import Skill

# ---------------------------------------------------------------------------
# Skill provider
# ---------------------------------------------------------------------------

_NAMESPACE = "robonix/skill/roboarm_voice"

skill = Skill(
    id="roboarm_voice",
    namespace=_NAMESPACE,
)

# ---------------------------------------------------------------------------
# FastMCP app + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("roboarm-voice")


@mcp.tool()
async def listen(duration: float = 5.0) -> str:
    """录制指定时长的音频，通过语音识别转写为文本。

    录音时长结束后自动停止并转写，返回识别结果。

    Args:
        duration: 录音时长，单位秒，默认5秒
    """
    import os
    import tempfile

    try:
        from llm.audio2text import MicPCMStream, audio_file2text

        mic = MicPCMStream(sample_rate=16000, channels=1, block_frames=640)

        print(f"[voice_skill] Recording for {duration}s...", flush=True)
        mic.start()

        # 录音期间分10次打印进度
        for i in range(10):
            time.sleep(duration / 10)
            print(f"[voice_skill] Recording... {(i + 1) * 10}%", flush=True)

        mic.stop()
        print("[voice_skill] Recording done, transcribing...", flush=True)

        text = audio_file2text(mic._save_path)
        print(f"[voice_skill] Transcription: {text}", flush=True)

        return json.dumps(
            {"status": "success", "text": text, "duration": duration},
            ensure_ascii=False,
        )

    except Exception as exc:
        traceback.print_exc()
        return json.dumps(
            {"status": "error", "error": str(exc)},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Tool metadata for capability declarations
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "listen",
        "description": "录制指定时长的音频并转写为文本。适用于通过语音获取用户指令。",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "录音时长，单位秒，默认5秒",
                }
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    skill.use_mcp_app(mcp)

    print("[voice_skill] bootstrapping...", flush=True)
    skill.bootstrap()
    print(
        f"[voice_skill] MCP server listening on port {skill._mcp_port}",
        flush=True,
    )

    endpoint = skill.mcp_endpoint
    for tool in _TOOLS:
        try:
            skill.declare_mcp(
                contract_id=f"{_NAMESPACE}/{tool['name']}",
                endpoint=endpoint,
                input_schema_json=json.dumps(tool["input_schema"]),
                description=tool["description"],
            )
            print(f"[voice_skill] declared MCP capability: {tool['name']}", flush=True)
        except Exception as exc:
            print(f"[voice_skill] declare {tool['name']} failed: {exc}", flush=True)

    print("[voice_skill] ready", flush=True)
    import signal

    stop = threading.Event()

    def _on_signal(signum, frame):
        print(f"[voice_skill] received signal {signum}, shutting down", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    stop.wait()


if __name__ == "__main__":
    main()
