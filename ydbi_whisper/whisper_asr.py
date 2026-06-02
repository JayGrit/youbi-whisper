from __future__ import annotations

import json
import logging
import re
import wave
from pathlib import Path

from . import db
from .config import (
    DEVICE,
    MODEL_ROOT,
    NLTK_DATA,
    TORCH_HOME,
    WHISPER_DOWNLOAD_ROOT,
    WHISPER_ENGINE,
    WHISPER_MODEL,
    WHISPERX_ALIGN,
    WHISPERX_ALIGN_INTERPOLATE_METHOD,
    WHISPERX_ALIGN_MODEL,
    WHISPERX_ALIGN_MODEL_DIR,
    WHISPERX_MODEL_PATH,
    WHISPERX_BATCH_SIZE,
    WHISPERX_CHUNK_SIZE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_REGROUP_MAX_CHARS,
    WHISPERX_REGROUP_MAX_DURATION_MS,
    WHISPER_SEMANTIC_SEGMENT_ENABLED,
    WHISPER_SEMANTIC_SEGMENT_MAX_WORDS,
    WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS,
    WHISPERX_VAD_METHOD,
    WHISPERX_VAD_OFFSET,
    WHISPERX_VAD_ONSET,
    device,
)

# 全局模型缓存变量，用于避免每次识别时都重新加载 Whisper 模型
_MODEL = None

# 当前模块的 logger，用于输出运行时日志
log = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"')\]}]*$")
MINOR_BREAK_RE = re.compile(r"[,;:，；：]+[\"')\]}]*$")


def _word_timestamps_for(runtime_device: str) -> bool:
    return "mps" not in runtime_device.lower()


def current_asr_config(language: str | None = None, *, load_model: bool = False) -> dict:
    model = _load_model() if load_model else None
    configured_runtime_device = _runtime_device_for_engine()
    runtime_device = str(getattr(model, "device", configured_runtime_device)).lower() if model is not None else configured_runtime_device
    return {
        "engine": WHISPER_ENGINE,
        "model": WHISPER_MODEL,
        "model_path": _model_name_or_path(),
        "configured_device": DEVICE,
        "runtime_device": runtime_device,
        "download_root": WHISPER_DOWNLOAD_ROOT or None,
        "torch_home": str(TORCH_HOME),
        "nltk_data": str(NLTK_DATA),
        "transcribe_options": {
            "language": language,
            "word_timestamps": WHISPER_ENGINE == "openai" and _word_timestamps_for(runtime_device),
            "verbose": False,
        },
        "whisperx": {
            "batch_size": WHISPERX_BATCH_SIZE,
            "compute_type": WHISPERX_COMPUTE_TYPE,
            "vad_method": WHISPERX_VAD_METHOD,
            "vad_options": _whisperx_vad_options(),
            "align": WHISPERX_ALIGN,
            "align_model": WHISPERX_ALIGN_MODEL or None,
            "align_model_dir": WHISPERX_ALIGN_MODEL_DIR,
            "align_interpolate_method": WHISPERX_ALIGN_INTERPOLATE_METHOD,
            "regroup": {
                "max_chars": WHISPERX_REGROUP_MAX_CHARS,
                "max_duration_ms": WHISPERX_REGROUP_MAX_DURATION_MS,
            },
            "semantic_caption": {
                "enabled": WHISPER_SEMANTIC_SEGMENT_ENABLED,
                "target_words": WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS,
                "max_words": WHISPER_SEMANTIC_SEGMENT_MAX_WORDS,
            },
        },
    }


def _runtime_device_for_engine() -> str:
    runtime_device = device()
    if WHISPER_ENGINE == "whisperx" and "mps" in runtime_device.lower():
        return "cpu"
    return runtime_device


def _whisperx_vad_options() -> dict:
    return {
        "chunk_size": WHISPERX_CHUNK_SIZE,
        "vad_onset": WHISPERX_VAD_ONSET,
        "vad_offset": WHISPERX_VAD_OFFSET,
    }


def _model_name_or_path() -> str:
    if WHISPER_ENGINE != "whisperx":
        return WHISPER_MODEL

    configured_path = Path(WHISPERX_MODEL_PATH).expanduser() if WHISPERX_MODEL_PATH else None
    if configured_path is not None:
        return str(configured_path)

    local_model = MODEL_ROOT / "faster-whisper-large-v3-turbo"
    if local_model.exists():
        return str(local_model)

    return WHISPER_MODEL


def _load_model():
    # 声明使用全局变量 _MODEL
    global _MODEL

    # 如果模型还没有加载过，则进行首次加载
    if _MODEL is None:
        runtime_device = _runtime_device_for_engine()
        log.info(
            "whisper loading model engine=%s model=%s model_path=%s configured_device=%s runtime_device=%s download_root=%s torch_home=%s",
            WHISPER_ENGINE,
            WHISPER_MODEL,
            _model_name_or_path(),
            DEVICE,
            runtime_device,
            WHISPER_DOWNLOAD_ROOT or None,
            TORCH_HOME,
        )
        if WHISPER_ENGINE == "whisperx":
            import whisperx

            _MODEL = whisperx.load_model(
                _model_name_or_path(),
                runtime_device,
                compute_type=WHISPERX_COMPUTE_TYPE,
                language=None,
                vad_method=WHISPERX_VAD_METHOD,
                vad_options=_whisperx_vad_options(),
                download_root=WHISPER_DOWNLOAD_ROOT or None,
            )
            log.info(
                "whisper model loaded engine=%s model=%s model_path=%s runtime_device=%s",
                WHISPER_ENGINE,
                WHISPER_MODEL,
                _model_name_or_path(),
                runtime_device,
            )
            return _MODEL

        if WHISPER_ENGINE != "openai":
            raise ValueError(f"Unsupported whisper engine: {WHISPER_ENGINE}")

        # 延迟导入 whisper，避免模块加载时就引入较重的依赖
        import whisper

        # 加载 Whisper 模型
        # WHISPER_MODEL：模型名称，例如 tiny/base/small/medium/large 等
        # device()：运行设备，例如 cpu、cuda、mps
        # download_root：模型下载或缓存目录；如果为空，则交给 whisper 使用默认目录
        _MODEL = whisper.load_model(
            WHISPER_MODEL,
            device=runtime_device,
            download_root=WHISPER_DOWNLOAD_ROOT or None,
        )
        log.info("whisper model loaded engine=%s model=%s runtime_device=%s", WHISPER_ENGINE, WHISPER_MODEL, runtime_device)

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


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _segment_from_words(words: list[dict]) -> dict | None:
    timed_words = [word for word in words if "start" in word and "end" in word]
    text = " ".join(_word_text(word) for word in words if _word_text(word)).strip()
    if not text or not timed_words:
        return None
    return {
        "text": text,
        "start": float(timed_words[0]["start"]),
        "end": float(timed_words[-1]["end"]),
        "words": words,
    }


def _segment_too_long(segment: dict) -> bool:
    text = str(segment.get("text") or "")
    duration_ms = int(round((float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)) * 1000))
    return len(text) >= WHISPERX_REGROUP_MAX_CHARS or duration_ms >= WHISPERX_REGROUP_MAX_DURATION_MS


def _split_long_segment_on_minor_punctuation(segment: dict) -> list[dict]:
    if not _segment_too_long(segment):
        return [segment]

    words = segment.get("words") or []
    if not words:
        return [segment]

    split_segments = []
    current_words = []

    def append_group(group_words: list[dict]) -> None:
        grouped = _segment_from_words(group_words)
        if grouped is not None:
            split_segments.append(grouped)

    def flush_current() -> None:
        nonlocal current_words
        append_group(current_words)
        current_words = []

    for word in words:
        text = _word_text(word)
        if not text:
            continue
        current_words.append(word)
        current_segment = _segment_from_words(current_words)
        if current_segment is None or not _segment_too_long(current_segment):
            continue

        split_at = None
        for idx in range(len(current_words) - 2, -1, -1):
            if MINOR_BREAK_RE.search(_word_text(current_words[idx])):
                split_at = idx + 1
                break

        if split_at is None and len(current_words) > 1:
            split_at = len(current_words) - 1

        if split_at:
            append_group(current_words[:split_at])
            current_words = current_words[split_at:]

    flush_current()
    return split_segments or [segment]


def _sentence_segmenter_language(language: str) -> str:
    normalized = str(language or "en").strip().lower().replace("_", "-")
    if not normalized:
        return "en"
    primary = normalized.split("-", 1)[0]
    return {
        "zh": "zh",
        "cn": "zh",
        "ja": "ja",
        "jp": "ja",
        "ko": "ko",
    }.get(primary, primary)


def _segment_words_with_pysbd(words: list[dict], language: str) -> list[dict]:
    try:
        import pysbd
    except Exception as exc:
        log.warning("pysbd import failed, fallback to punctuation regroup: %s", exc)
        return []

    word_spans = []
    text_parts = []
    cursor = 0
    for word in words:
        text = _word_text(word)
        if not text:
            continue
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        word_spans.append((start, cursor, word))

    full_text = "".join(text_parts).strip()
    if not full_text or not word_spans:
        return []

    segmenter_language = _sentence_segmenter_language(language)
    try:
        segmenter = pysbd.Segmenter(language=segmenter_language, clean=False)
        sentence_texts = [sentence.strip() for sentence in segmenter.segment(full_text) if sentence.strip()]
    except Exception as exc:
        log.warning("pysbd segment failed language=%s, fallback to punctuation regroup: %s", segmenter_language, exc)
        return []

    if len(sentence_texts) <= 1:
        return []

    segments = []
    search_from = 0
    word_index = 0
    for sentence in sentence_texts:
        sentence_start = full_text.find(sentence, search_from)
        if sentence_start < 0:
            return []
        sentence_end = sentence_start + len(sentence)
        sentence_words = []
        while word_index < len(word_spans) and word_spans[word_index][1] <= sentence_start:
            word_index += 1
        scan_index = word_index
        while scan_index < len(word_spans) and word_spans[scan_index][0] < sentence_end:
            sentence_words.append(word_spans[scan_index][2])
            scan_index += 1
        word_index = scan_index
        segment = _segment_from_words(sentence_words)
        if segment is not None:
            segments.extend(_split_long_segment_on_minor_punctuation(segment))
        search_from = sentence_end

    return segments


def _regroup_words_by_punctuation(words: list[dict]) -> list[dict]:
    regrouped = []
    current_words = []

    def flush() -> None:
        nonlocal current_words
        segment = _segment_from_words(current_words)
        if segment is None:
            current_words = []
            return
        regrouped.extend(_split_long_segment_on_minor_punctuation(segment))
        current_words = []

    for word in words:
        text = _word_text(word)
        if not text:
            continue
        if "start" not in word or "end" not in word:
            flush()
            continue

        current_words.append(word)
        if SENTENCE_END_RE.search(text):
            flush()

    flush()
    return regrouped


def _regroup_aligned_segments(segments: list[dict], language: str, diagnostics: dict | None = None) -> list[dict]:
    regrouped = []
    current_words = []

    def flush() -> None:
        nonlocal current_words
        if not current_words:
            return
        sentence_segments = []
        try:
            from .semantic_caption import segment_words_with_llm

            if diagnostics is not None:
                sentence_segments = segment_words_with_llm(current_words, language, diagnostics) or []
        except Exception as exc:
            log.warning("semantic caption segmentation failed before fallback language=%s: %s", language, exc)
            if diagnostics is not None:
                diagnostics.setdefault("errors", []).append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if not sentence_segments:
            sentence_segments = _segment_words_with_pysbd(current_words, language)
            if sentence_segments and diagnostics is not None:
                diagnostics.setdefault("rule_fallbacks", []).append(
                    {
                        "method": "pysbd",
                        "word_count": len(current_words),
                        "segments": len(sentence_segments),
                    }
                )
        if not sentence_segments:
            sentence_segments = _regroup_words_by_punctuation(current_words)
            if diagnostics is not None:
                diagnostics.setdefault("rule_fallbacks", []).append(
                    {
                        "method": "punctuation",
                        "word_count": len(current_words),
                        "segments": len(sentence_segments),
                    }
                )
        regrouped.extend(sentence_segments)
        current_words = []

    for segment in segments:
        words = segment.get("words") or []
        if not words:
            flush()
            regrouped.append(segment)
            continue

        for word in words:
            text = _word_text(word)
            if not text:
                continue
            if "start" not in word or "end" not in word:
                flush()
                continue
            current_words.append(word)

    flush()
    return regrouped or segments


def _align_whisperx_result(
    whisperx,
    result: dict,
    audio: object,
    language: str,
    runtime_device: str,
    *,
    task_id: str | None = None,
) -> dict:
    if not WHISPERX_ALIGN or not result.get("segments"):
        return result

    align_language = str(result.get("language") or language or "en").lower()
    log.info(
        "whisperx align model loading language=%s model=%s model_dir=%s runtime_device=%s",
        align_language,
        WHISPERX_ALIGN_MODEL or None,
        WHISPERX_ALIGN_MODEL_DIR,
        runtime_device,
    )
    Path(WHISPERX_ALIGN_MODEL_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    align_model, align_metadata = whisperx.load_align_model(
        align_language,
        runtime_device,
        model_name=WHISPERX_ALIGN_MODEL or None,
        model_dir=WHISPERX_ALIGN_MODEL_DIR,
    )
    log.info(
        "whisperx align start language=%s segments=%s interpolate_method=%s",
        align_language,
        len(result.get("segments") or []),
        WHISPERX_ALIGN_INTERPOLATE_METHOD,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        runtime_device,
        interpolate_method=WHISPERX_ALIGN_INTERPOLATE_METHOD,
        return_char_alignments=False,
        print_progress=False,
    )
    aligned_segments = aligned.get("segments") or []
    semantic_diagnostics: dict = {
        "task_id": task_id,
        "language": align_language,
        "input_aligned_segments": len(aligned_segments),
        "input_word_segments": len(aligned.get("word_segments") or []),
        "chunks": [],
    }
    regrouped_segments = _regroup_aligned_segments(aligned_segments, align_language, semantic_diagnostics)
    aligned["segments"] = regrouped_segments
    aligned["text"] = " ".join(str(seg.get("text") or "").strip() for seg in regrouped_segments).strip()
    aligned["language"] = align_language
    semantic_diagnostics["output_segments"] = len(regrouped_segments)
    aligned["semantic_caption_segmentation"] = semantic_diagnostics
    log.info(
        "whisperx align done language=%s aligned_segments=%s regrouped_segments=%s word_segments=%s",
        align_language,
        len(aligned_segments),
        len(regrouped_segments),
        len(aligned.get("word_segments") or []),
    )
    return aligned


def _transcribe_whisperx_raw(model: object, vocals_file: Path, language: str, runtime_device: str) -> tuple[dict, object]:
    import whisperx

    log.info("whisperx loading audio file=%s", vocals_file)
    audio = whisperx.load_audio(str(vocals_file))
    log.info(
        "whisperx transcribe start file=%s language=%s batch_size=%s chunk_size=%s runtime_device=%s",
        vocals_file,
        language,
        WHISPERX_BATCH_SIZE,
        WHISPERX_CHUNK_SIZE,
        runtime_device,
    )
    result = model.transcribe(
        audio,
        batch_size=WHISPERX_BATCH_SIZE,
        language=language,
        chunk_size=WHISPERX_CHUNK_SIZE,
        verbose=False,
    )
    segments = result.get("segments", [])
    result["text"] = " ".join(str(seg.get("text") or "").strip() for seg in segments).strip()
    log.info(
        "whisperx transcribe done file=%s language=%s segments=%s text_chars=%s",
        vocals_file,
        result.get("language") or language,
        len(segments),
        len(result["text"]),
    )
    return result, audio


def _transcribe(model: object, vocals_file: Path, language: str, runtime_device: str, word_timestamps: bool) -> dict:
    if WHISPER_ENGINE == "whisperx":
        import whisperx

        result, audio = _transcribe_whisperx_raw(model, vocals_file, language, runtime_device)
        result = _align_whisperx_result(whisperx, result, audio, language, runtime_device)
        return result

    log.info(
        "openai whisper transcribe start file=%s language=%s runtime_device=%s word_timestamps=%s",
        vocals_file,
        language,
        runtime_device,
        word_timestamps,
    )
    return model.transcribe(
        str(vocals_file),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
    )


def _audio_duration_ms(audio_file: Path) -> int:
    if audio_file.suffix.lower() == ".wav":
        with wave.open(str(audio_file), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate > 0:
                return int(round(wav_file.getnframes() * 1000 / frame_rate))

    from pydub import AudioSegment

    return len(AudioSegment.from_file(audio_file))


def _transcribe_resumable(model: object, vocals_file: Path, language: str, runtime_device: str, word_timestamps: bool, task_id: str | None) -> dict:
    if not task_id or WHISPER_ENGINE != "whisperx":
        return _transcribe(model, vocals_file, language, runtime_device, word_timestamps)

    import whisperx

    cached_align = db.get_whisper_step_result(task_id, "align_regrouped")
    if cached_align:
        log.info("whisper resume task=%s step=align_regrouped", task_id)
        return cached_align["payload_json"]

    cached_raw = db.get_whisper_step_result(task_id, "transcribe_raw")
    audio = None
    if cached_raw:
        log.info("whisper resume task=%s step=transcribe_raw", task_id)
        result = cached_raw["payload_json"]
        audio = whisperx.load_audio(str(vocals_file))
    else:
        result, audio = _transcribe_whisperx_raw(model, vocals_file, language, runtime_device)
        db.upsert_whisper_step_result(
            task_id,
            "transcribe_raw",
            status="success",
            language=str(result.get("language") or language),
            model=WHISPER_MODEL,
            payload=result,
            diagnostics={
                "segments": len(result.get("segments") or []),
                "text_chars": len(str(result.get("text") or "")),
            },
        )

    aligned = _align_whisperx_result(whisperx, result, audio, language, runtime_device, task_id=task_id)
    db.upsert_whisper_step_result(
        task_id,
        "align_regrouped",
        status="success",
        language=str(aligned.get("language") or language),
        model=WHISPER_MODEL,
        payload=aligned,
        diagnostics=aligned.get("semantic_caption_segmentation") or {},
    )
    return aligned


def recognize_speech(vocals_file: Path, session: Path, language: str, task_id: str | None = None) -> dict:
    # 当前任务的 metadata 目录，用于存放识别过程或后续阶段生成的元数据
    metadata_dir = session / "metadata"

    # 确保 metadata 目录存在
    # parents=True 表示父目录不存在时也一并创建
    # exist_ok=True 表示目录已存在时不报错
    metadata_dir.mkdir(parents=True, exist_ok=True)
    semantic_debug_path = metadata_dir / "semantic_caption_segmentation.json"

    if task_id:
        cached_final = db.get_whisper_step_result(task_id, "final_payload")
        if cached_final:
            log.info("whisper resume task=%s step=final_payload", task_id)
            payload = cached_final["payload_json"]
            cached_align = db.get_whisper_step_result(task_id, "align_regrouped")
            semantic_debug = (cached_align or {}).get("diagnostics_json") or {
                "enabled": WHISPER_SEMANTIC_SEGMENT_ENABLED,
                "status": "not_available",
                "reason": "cached_final_payload_has_no_align_diagnostics",
            }
            semantic_debug_path.write_text(json.dumps(semantic_debug, ensure_ascii=False, indent=2), encoding="utf-8")
            result = payload.setdefault("result", {})
            result["semantic_caption_debug_path"] = str(semantic_debug_path)
            return payload

    # 加载或复用 Whisper 模型
    model = _load_model()

    # 获取模型实际运行设备
    # 如果 model 上没有 device 属性，则回退使用配置里的 device()
    # 转成小写字符串，方便后续判断
    runtime_device = str(getattr(model, "device", _runtime_device_for_engine())).lower()

    # 判断是否启用词级时间戳
    # 在 MPS 设备上，Whisper 的 word_timestamps 可能触发 float64 DTW 相关问题
    # 因此如果运行在 mps 上，就关闭词级时间戳
    word_timestamps = WHISPER_ENGINE == "openai" and _word_timestamps_for(runtime_device)

    # 如果 OpenAI Whisper 在 MPS 上关闭了词级时间戳，打印 warning，方便排查为什么 words 为空
    if WHISPER_ENGINE == "openai" and not word_timestamps:
        log.warning("Whisper is running on MPS; word timestamps are disabled to avoid MPS float64 DTW failure.")
    log.info(
        "whisper runtime audio=%s engine=%s model=%s configured_device=%s runtime_device=%s language=%s word_timestamps=%s vad_method=%s vad_options=%s align=%s",
        vocals_file,
        WHISPER_ENGINE,
        WHISPER_MODEL,
        DEVICE,
        runtime_device,
        language,
        word_timestamps,
        WHISPERX_VAD_METHOD if WHISPER_ENGINE == "whisperx" else None,
        _whisperx_vad_options() if WHISPER_ENGINE == "whisperx" else None,
        WHISPERX_ALIGN if WHISPER_ENGINE == "whisperx" else None,
    )

    # 调用 Whisper 进行语音识别
    # vocals_file：待识别的人声文件
    # language：指定识别语言
    # word_timestamps：是否返回词级时间戳
    # verbose=False：关闭详细输出
    result = _transcribe_resumable(model, vocals_file, language, runtime_device, word_timestamps, task_id)
    semantic_debug = result.get("semantic_caption_segmentation") or {
        "enabled": WHISPER_SEMANTIC_SEGMENT_ENABLED,
        "status": "not_available",
        "reason": "transcribe_result_has_no_semantic_caption_diagnostics",
    }
    semantic_debug_path.write_text(json.dumps(semantic_debug, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("semantic caption diagnostics written path=%s", semantic_debug_path)
    if task_id:
        db.save_whisper_semantic_chunks(task_id, semantic_debug)

    # 将 Whisper 返回的 segments 转成项目内部统一的 utterances 格式
    utterances = _convert_segments(result.get("segments", []))
    log.info("whisper result converted audio=%s utterances=%s", vocals_file, len(utterances))

    # 如果 Whisper 没有返回任何语音片段，说明识别结果不可用
    if not utterances:
        raise RuntimeError("Whisper did not return any segments.")

    # 使用 pydub 读取音频文件，并计算音频总时长
    # len(AudioSegment) 返回毫秒数
    duration_ms = _audio_duration_ms(vocals_file)

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

            # 本轮语义字幕切分的完整诊断输出文件
            "semantic_caption_debug_path": str(semantic_debug_path),
        },
    }
    if task_id:
        db.upsert_whisper_step_result(
            task_id,
            "final_payload",
            status="success",
            language=language,
            model=WHISPER_MODEL,
            payload=payload,
            diagnostics={
                "segments": len(utterances),
                "text_chars": len(str((result.get("text") or "").strip())),
                "semantic_caption_debug_path": str(semantic_debug_path),
            },
        )

    # 返回标准化后的 ASR payload
    return payload
