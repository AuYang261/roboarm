# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下，您可自行逐一或者复制到一个新的txt文件利用pip一次性安装：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#
#  语音听写流式 WebAPI 接口调用示例 接口文档（必看）：https://doc.xfyun.cn/rest_api/语音听写（流式版）.html
#  webapi 听写服务参考帖子（必看）：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=38947&extra=
#  语音听写流式WebAPI 服务，热词使用方式：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--个性化热词，
#  设置热词
#  注意：热词只能在识别的时候会增加热词的识别权重，需要注意的是增加相应词条的识别率，但并不是绝对的，具体效果以您测试为准。
#  语音听写流式WebAPI 服务，方言试用方法：登陆开放平台https://www.xfyun.cn/后，找到控制台--我的应用---语音听写（流式）---服务管理--识别语种列表
#  可添加语种或方言，添加后会显示该方言的参数值
#  错误码链接：https://www.xfyun.cn/document/error-code （code返回错误码时必看）
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import _thread as thread
import time
from unittest import result
from config_getter import get_config_value
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
import sounddevice as sd

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


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

    def start(self, device: int | None = None):
        def callback(indata: np.ndarray, frames: int, time_info, status):
            # indata dtype=int16, shape=(frames, channels)
            if status:
                # 可以打印，但不要太频繁
                pass
            if self._closed:
                return
            try:
                self.q.put_nowait(indata.tobytes())
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
            # 多出来的塞回去（简化：不塞回也行，但会丢少量数据）
            extra = data[n:]
            data = data[:n]
            try:
                self.q.put_nowait(extra)
            except queue.Full:
                pass
        return data


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioBytes, MicStream=None):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioBytes = AudioBytes
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


# 收到websocket错误的处理
def on_error(ws, error):
    # print("### error:", error)
    pass


# 收到websocket关闭的处理
def on_close(ws, close_status_code, close_msg):
    print("### closed :", close_status_code, close_msg)


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


def mp3_bytes_to_pcm_16k_mono_s16le(mp3_bytes: bytes) -> bytes:
    """
    MP3(bytes) -> raw PCM bytes (s16le), 16kHz, mono

    返回值:
        pcm_bytes: 不含 WAV 头的原始 PCM 数据，可直接用于你现有的
                   payload: {"encoding":"raw","sample_rate":16000}
    依赖:
        pydub（其内部可依赖 ffmpeg，但这里不直接调用 ffmpeg 命令行）
    """
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    return audio.raw_data


def get_audio_text(
    appid,
    api_key,
    api_secret,
):
    """从麦克风获取一段音频并转写为文本"""
    mic = MicPCMStream(sample_rate=16000, channels=1, block_frames=640)
    mic.start()
    input("请开始说话...，按回车键结束录音。\n")
    mic.stop()
    print("录音结束，正在识别...")

    audio_chunks: list[bytes] = []
    while True:
        try:
            chunk = mic.q.get_nowait()
        except queue.Empty:
            break
        audio_chunks.append(chunk)
    audio_bytes = b"".join(audio_chunks)

    wsParam = Ws_Param(
        APPID=appid,
        APISecret=api_secret,
        APIKey=api_key,
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
            # 不打印，直接累计到 result
            result += chunk_text
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
    with open(os.path.join(os.path.dirname(__file__), "iat_mp3_16k.mp3"), "rb") as f:
        audio_bytes = f.read()

    audio_bytes = mp3_bytes_to_pcm_16k_mono_s16le(audio_bytes)

    wsParam = Ws_Param(
        APPID=get_config_value("APPID"),
        APISecret=get_config_value("APISecret"),
        APIKey=get_config_value("APIKey"),
        AudioBytes=audio_bytes,
    )
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(
        wsUrl,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = lambda ws: thread.start_new_thread(send, (ws, wsParam))
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    print("=== 麦克风实时识别 ===")

    result = get_audio_text(
        appid=get_config_value("APPID"),
        api_key=get_config_value("APIKey"),
        api_secret=get_config_value("APISecret"),
    )
    print("麦克风识别结果：", result)
