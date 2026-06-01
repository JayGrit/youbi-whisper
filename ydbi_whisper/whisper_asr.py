from __future__ import annotations

import logging
from pathlib import Path

from .config import DEVICE, WHISPER_DOWNLOAD_ROOT, WHISPER_MODEL, device

# 全局模型缓存变量，用于避免每次识别时都重新加载 Whisper 模型
_MODEL = None

# 当前模块的 logger，用于输出运行时日志
log = logging.getLogger(__name__)


def _word_timestamps_for(runtime_device: str) -> bool:
    return "mps" not in runtime_device.lower()


def current_asr_config(language: str | None = None, *, load_model: bool = False) -> dict:
    model = _load_model() if load_model else None
    runtime_device = str(getattr(model, "device", device())).lower() if model is not None else device()
    return {
        "model": WHISPER_MODEL,
        "configured_device": DEVICE,
        "runtime_device": runtime_device,
        "download_root": WHISPER_DOWNLOAD_ROOT or None,
        "transcribe_options": {
            "language": language,
            "word_timestamps": _word_timestamps_for(runtime_device),
            "verbose": False,
        },
    }


def _load_model():
    # 声明使用全局变量 _MODEL
    global _MODEL

    # 如果模型还没有加载过，则进行首次加载
    if _MODEL is None:
        # 延迟导入 whisper，避免模块加载时就引入较重的依赖
        import whisper

        # 加载 Whisper 模型
        # WHISPER_MODEL：模型名称，例如 tiny/base/small/medium/large 等
        # device()：运行设备，例如 cpu、cuda、mps
        # download_root：模型下载或缓存目录；如果为空，则交给 whisper 使用默认目录
        _MODEL = whisper.load_model(
            WHISPER_MODEL,
            device=device(),
            download_root=WHISPER_DOWNLOAD_ROOT or None,
        )

    # 返回已加载好的模型实例
    return _MODEL


def _to_ms(seconds: float) -> int:
    # 将秒转换成毫秒
    # Whisper 返回的时间通常是秒，这里统一转成毫秒，方便后续字幕/片段处理
    return int(round(float(seconds) * 1000))


def _convert_words(words: list) -> list:
    # 将 Whisper 返回的词级时间戳转换成统一格式
    return [
        {
            # 单词文本，Whisper 中字段名通常是 word
            "text": w.get("word", ""),

            # 单词开始时间，单位从秒转换成毫秒
            "start_time": _to_ms(w.get("start", 0.0)),

            # 单词结束时间，单位从秒转换成毫秒
            "end_time": _to_ms(w.get("end", 0.0)),
        }

        # 遍历每一个词级结果
        # 如果 words 为 None 或空，则使用空列表，避免报错
        for w in words or []
    ]


def _convert_segments(segments: list) -> list:
    # 将 Whisper 返回的句子/片段级 segments 转换成统一的 utterances 格式
    return [
        {
            # 当前语音片段的文本内容，去掉首尾空白
            "text": seg.get("text", "").strip(),

            # 当前片段开始时间，单位从秒转换成毫秒
            "start_time": _to_ms(seg.get("start", 0.0)),

            # 当前片段结束时间，单位从秒转换成毫秒
            "end_time": _to_ms(seg.get("end", 0.0)),

            # 当前片段中的词级时间戳信息
            "words": _convert_words(seg.get("words", [])),
        }

        # 遍历 Whisper 返回的每一个 segment
        for seg in segments
    ]


def recognize_speech(vocals_file: Path, session: Path, language: str) -> dict:
    from pydub import AudioSegment

    # 当前任务的 metadata 目录，用于存放识别过程或后续阶段生成的元数据
    metadata_dir = session / "metadata"

    # 确保 metadata 目录存在
    # parents=True 表示父目录不存在时也一并创建
    # exist_ok=True 表示目录已存在时不报错
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 加载或复用 Whisper 模型
    model = _load_model()

    # 获取模型实际运行设备
    # 如果 model 上没有 device 属性，则回退使用配置里的 device()
    # 转成小写字符串，方便后续判断
    runtime_device = str(getattr(model, "device", device())).lower()

    # 判断是否启用词级时间戳
    # 在 MPS 设备上，Whisper 的 word_timestamps 可能触发 float64 DTW 相关问题
    # 因此如果运行在 mps 上，就关闭词级时间戳
    word_timestamps = _word_timestamps_for(runtime_device)

    # 如果关闭了词级时间戳，打印 warning，方便排查为什么 words 为空
    if not word_timestamps:
        log.warning("Whisper is running on MPS; word timestamps are disabled to avoid MPS float64 DTW failure.")
    log.info(
        "whisper runtime audio=%s model=%s configured_device=%s runtime_device=%s language=%s word_timestamps=%s",
        vocals_file,
        WHISPER_MODEL,
        DEVICE,
        runtime_device,
        language,
        word_timestamps,
    )

    # 调用 Whisper 进行语音识别
    # vocals_file：待识别的人声文件
    # language：指定识别语言
    # word_timestamps：是否返回词级时间戳
    # verbose=False：关闭详细输出
    result = model.transcribe(
        str(vocals_file),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
    )

    # 将 Whisper 返回的 segments 转成项目内部统一的 utterances 格式
    utterances = _convert_segments(result.get("segments", []))

    # 如果 Whisper 没有返回任何语音片段，说明识别结果不可用
    if not utterances:
        raise RuntimeError("Whisper did not return any segments.")

    # 使用 pydub 读取音频文件，并计算音频总时长
    # len(AudioSegment) 返回毫秒数
    duration_ms = len(AudioSegment.from_file(vocals_file))

    # 组装最终返回结果
    # 结构尽量对齐常见 ASR 服务返回格式：
    # audio_info 存放音频信息
    # result 存放识别文本和分段结果
    payload = {
        # 音频基础信息
        "audio_info": {"duration": duration_ms},

        # ASR 识别结果
        "result": {
            # 完整识别文本
            "text": (result.get("text") or "").strip(),

            # 分段识别结果
            "utterances": utterances,
        },
    }

    # 返回标准化后的 ASR payload
    return payload
