# -*- coding:utf-8 -*-

import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import _thread as thread
import time
from unittest import result
from utils.config_getter import get_config_value
from time import mktime
import io
from pydub import AudioSegment
import websocket
import queue
import numpy as np
import base64
import datetime
import hashlib
import hmac
import json
import ssl
from datetime import datetime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

try:
    import sounddevice as sd
except Exception as e:
    print(e)
    print("Import sounddevice error.")
    print("Try 'sudo apt install portaudio19-dev'. Or no audio input device.")
import wave
import requests
from pathlib import Path

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

https_proxy = get_config_value("https_proxy", None, False)
if https_proxy is not None and https_proxy != "":
    os.environ["http_proxy"] = https_proxy
    os.environ["https_proxy"] = https_proxy


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} using xunfei API is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


class MicPCMStream:
    """
    采集麦克风音频，输出 16kHz/mono/int16 的 PCM bytes 到队列。
    """

    def __init__(
        self, sample_rate: int = 16000, channels: int = 1, block_frames: int = 640
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_frames = block_frames  # 每次回调多少帧（frames）
        self.q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
        self._stream: "sd.InputStream | None" = None
        self._closed = False

        self._save_path: str = os.path.join(os.path.dirname(__file__), "mic_record.wav")
        self._wav_fp: "wave.Wave_write | None" = None

    def _open_wav_if_needed(self) -> None:
        if self._wav_fp is not None:
            return
        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
        wf = wave.open(self._save_path, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)  # int16 => 2 bytes
        wf.setframerate(self.sample_rate)
        self._wav_fp = wf

    def start(self, device: int | None = None):
        self._open_wav_if_needed()

        def callback(indata: np.ndarray, frames: int, time_info, status):
            # indata dtype=int16, shape=(frames, channels)
            if self._closed:
                return

            raw = indata.tobytes()
            try:
                if self._wav_fp is not None:
                    self._wav_fp.writeframes(raw)
            except Exception:
                # 保存失败不应影响主流程
                pass

            try:
                self.q.put_nowait(raw)
            except queue.Full:
                # 丢帧（实时场景可接受）
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.block_frames,
            callback=callback,
            device=device,
        )
        self._stream.start()

    def stop(self):
        self._closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._wav_fp is not None:
            try:
                self._wav_fp.close()
            finally:
                self._wav_fp = None

    def read_bytes(self, n: int, timeout: float = 1.0) -> bytes:
        """
        阻塞式从队列拼接到 n 字节（用于发送线程）。
        """
        chunks: list[bytes] = []
        total = 0
        while total < n and not self._closed:
            try:
                b = self.q.get(timeout=timeout)
            except queue.Empty:
                continue
            chunks.append(b)
            total += len(b)

        data = b"".join(chunks)
        if len(data) > n:
            # 直接截断，多余部分丢弃
            data = data[:n]
        return data


@deprecated
class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioBytes, MicStream=None):
        """初始化websocket参数

        参数:
            APPID: 应用ID
            APIKey: API Key
            APISecret: API Secret
            AudioBytes: 音频数据，一次性识别全部音频字节
            MicStream: 麦克风音频流，流式识别，可选，不为空时优先使用麦克风，忽略 AudioBytes
        """
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioBytes: bytes = AudioBytes
        self.MicStream = MicStream
        self.iat_params = {
            "domain": "slm",
            "language": "zh_cn",
            "accent": "mandarin",
            "dwa": "wpgs",
            "result": {"encoding": "utf8", "compress": "raw", "format": "plain"},
        }

    # 生成url
    def create_url(self):
        url = "ws://iat.xf-yun.com/v1"
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "iat.xf-yun.com" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v1 " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = (
            'api_key="%s", algorithm="%s", headers="%s", signature="%s"'
            % (self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )
        # 将请求的鉴权参数组合为字典
        v = {"authorization": authorization, "date": date, "host": "iat.xf-yun.com"}
        # 拼接鉴权参数，生成url
        url = url + "?" + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


@deprecated
# 收到websocket消息的处理
def on_message(ws, message):
    # print("### on_message ###")
    message = json.loads(message)
    code = message["header"]["code"]
    status = message["header"]["status"]
    # print(message["header"]["sid"])
    if code != 0:
        print(f"请求错误：{code}")
        ws.close()
    else:
        payload = message.get("payload")
        if payload:
            text = payload["result"]["text"]
            text = json.loads(str(base64.b64decode(text), "utf8"))
            text_ws = text["ws"]
            result = ""
            for i in text_ws:
                for j in i["cw"]:
                    w = j["w"]
                    result += w
            print(result)
        if status == 2:
            ws.close()


@deprecated
# 收到websocket错误的处理
def on_error(ws, error):
    # print("### error:", error)
    pass


@deprecated
# 收到websocket关闭的处理
def on_close(ws, close_status_code, close_msg):
    # print("### closed :", close_status_code, close_msg)
    pass


@deprecated
def send(ws, wsParam: Ws_Param):
    frameSize = 1280  # 每一帧的音频大小
    intervel = 0.04  # 发送音频间隔(单位:s)
    status = (
        STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧
    )

    # 选择输入源：优先用麦克风流；否则退回用 wsParam.AudioBytes（文件）
    mic: MicPCMStream | None = getattr(wsParam, "MicStream", None)

    offset = 0
    total = 0
    mv = memoryview(wsParam.AudioBytes)
    total = len(mv)

    try:
        while True:
            # 取一帧 buf
            if mic is not None:
                buf = mic.read_bytes(frameSize)
                if not buf:
                    status = STATUS_LAST_FRAME
            else:
                if offset >= total:
                    buf = b""
                else:
                    buf = mv[offset : offset + frameSize].tobytes()
                    offset += frameSize
                if not buf:
                    status = STATUS_LAST_FRAME
            audio = str(base64.b64encode(buf), "utf-8")

            # 文件结束
            if not buf:
                status = STATUS_LAST_FRAME
            # 第一帧处理
            if status == STATUS_FIRST_FRAME:

                d = {
                    "header": {"status": 0, "app_id": wsParam.APPID},
                    "parameter": {"iat": wsParam.iat_params},
                    "payload": {
                        "audio": {
                            "audio": audio,
                            "sample_rate": 16000,
                            "encoding": "raw",
                        }
                    },
                }
                d = json.dumps(d)
                ws.send(d)
                status = STATUS_CONTINUE_FRAME
            # 中间帧处理
            elif status == STATUS_CONTINUE_FRAME:
                d = {
                    "header": {"status": 1, "app_id": wsParam.APPID},
                    "parameter": {"iat": wsParam.iat_params},
                    "payload": {
                        "audio": {
                            "audio": audio,
                            "sample_rate": 16000,
                            "encoding": "raw",
                        }
                    },
                }
                ws.send(json.dumps(d))
            # 最后一帧处理
            elif status == STATUS_LAST_FRAME:
                d = {
                    "header": {"status": 2, "app_id": wsParam.APPID},
                    "parameter": {"iat": wsParam.iat_params},
                    "payload": {
                        "audio": {
                            "audio": audio,
                            "sample_rate": 16000,
                            "encoding": "raw",
                        }
                    },
                }
                ws.send(json.dumps(d))
                break

            # 模拟音频采样间隔
            time.sleep(intervel)
    except Exception as e:
        print("send error:", e)


def mp3_bytes_to_pcm_16k_mono_s16le(mp3_bytes: bytes, format="mp3") -> bytes:
    """
    MP3(bytes) -> raw PCM bytes (s16le), 16kHz, mono
    参数:
        mp3_bytes: 输入的 MP3 格式音频数据
        format: 音频格式，mp3, wav 或 m4a（mp4）

    返回值:
        pcm_bytes: 不含 WAV 头的原始 PCM 数据，可直接用于你现有的
                   payload: {"encoding":"raw","sample_rate":16000}
    依赖:
        pydub（其内部可依赖 ffmpeg，但这里不直接调用 ffmpeg 命令行）
    """
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format=format)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio.raw_data


def get_audio_text(
    backend: str = "",
):
    """从麦克风获取一段音频并转写为文本

    参数:
        backend: "xunfei" 使用讯飞，其他值使用自部署语音转文字服务
    """

    mic = MicPCMStream(sample_rate=16000, channels=1, block_frames=640)
    input("按回车键开始录音...\n")
    mic.start()
    input("请开始说话...，按回车键结束录音。\n")
    mic.stop()
    print("录音结束，正在识别...")
    start_time = time.time()

    result = audio_file2text(mic._save_path, backend=backend)
    print("识别耗时: %.2f 秒" % (time.time() - start_time))
    return result


def audio_file2text(audio_path: str, backend: str = "") -> str:
    """从音频文件获取转写文本

    参数:
        audio_path: 音频文件路径
        backend: 使用的后端，"xunfei" 使用讯飞，其他值使用自部署服务
    """
    if backend == "xunfei":
        return _xfyun_transcribe(audio_path)
    return _whisper_transcribe(audio_path)


def _whisper_transcribe(audio_path: str) -> str:
    """调用自部署的 whisper 服务转写音频"""
    base_url = get_config_value("whisper_url")
    username = get_config_value("whisper_username")
    password = get_config_value("whisper_password")

    session = requests.Session()
    # session.trust_env = False  # 不使用系统代理，避免局域网请求走代理
    # 登录获取 session cookie
    if username:
        resp = session.post(
            f"{base_url}/api/login",
            json={"username": username, "password": password},
        )
        resp.raise_for_status()

    # 上传音频文件进行转写
    with open(audio_path, "rb") as f:
        resp = session.post(
            f"{base_url}/api/transcribe",
            files={"file": (os.path.basename(audio_path), f)},
            data={"language": "zh"},
        )
    resp.raise_for_status()
    return resp.json().get("text", "")


@deprecated
def _xfyun_transcribe(audio_path: str) -> str:
    """调用讯飞语音听写服务转写音频"""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_bytes = mp3_bytes_to_pcm_16k_mono_s16le(
        audio_bytes, format=os.path.splitext(audio_path)[1][1:]
    )

    wsParam = Ws_Param(
        APPID=get_config_value("APPID"),
        APISecret=get_config_value("APISecret"),
        APIKey=get_config_value("APIKey"),
        AudioBytes=audio_bytes,
    )
    result = ""

    # 闭包函数，处理收到的消息
    def on_message_local(ws, message):
        nonlocal result
        message = json.loads(message)
        code = message["header"]["code"]
        status = message["header"]["status"]
        if code != 0:
            # 出错就关闭
            ws.close()
            return
        payload = message.get("payload")
        if payload:
            text = payload["result"]["text"]
            text = json.loads(str(base64.b64decode(text), "utf8"))
            text_ws = text["ws"]
            chunk_text = ""
            for i in text_ws:
                for j in i["cw"]:
                    chunk_text += j.get("w", "")
            # 对于流式识别，每次只识别最新的，逐步累积结果
            # result += chunk_text
            # 读取文件识别，每次都会重复前面的结果，所以覆盖，但最后可能会输出一个句号，保留最长的结果
            if len(chunk_text) > len(result):
                result = chunk_text
        if status == 2:
            ws.close()

    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(
        wsUrl,
        on_message=on_message_local,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = lambda ws: thread.start_new_thread(send, (ws, wsParam))
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return result


if __name__ == "__main__":
    backend = get_config_value("audio2text_backend", default="", raise_if_missing=False)
    test_path = Path(__file__).parent / "dataset" / "audio" / "抓取蓝色积木.m4a"
    if test_path.exists():
        print(
            f"{test_path} 识别结果：",
            audio_file2text(test_path.as_posix(), backend=backend),
        )

    print("=== 麦克风实时识别 ===")
    result = get_audio_text(backend=backend)
    print("麦克风识别结果：", result)
