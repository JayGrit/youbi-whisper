from __future__ import annotations

import logging
import importlib
import inspect
import re
import unicodedata
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
    WHISPER_RUNTIME_DEVICE,
    WHISPERX_ALIGN,
    WHISPERX_ALIGN_INTERPOLATE_METHOD,
    WHISPERX_ALIGN_LOCAL_FILES_ONLY,
    WHISPERX_ALIGN_MODEL,
    WHISPERX_ALIGN_MODEL_DIR,
    WHISPERX_MODEL_PATH,
    WHISPERX_BATCH_SIZE,
    WHISPERX_CHUNK_SIZE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_DIARIZE_HF_TOKEN,
    WHISPERX_DIARIZE_MAX_SPEAKERS,
    WHISPERX_DIARIZE_MIN_SPEAKERS,
    _huggingface_token,
    WHISPERX_REGROUP_MAX_CHARS,
    WHISPERX_REGROUP_MAX_DURATION_MS,
    WHISPERX_VAD_METHOD,
    WHISPERX_VAD_OFFSET,
    WHISPERX_VAD_ONSET,
    device,
)

# 全局模型缓存变量，用于避免每次识别时都重新加载 Whisper 模型
_MODEL = None
_ALIGN_MODELS: dict[tuple[str, str, str, str], tuple[object, dict]] = {}
_DIARIZATION_MODELS: dict[tuple[str, str], object] = {}

# 当前模块的 logger，用于输出运行时日志
log = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"')\]}]*$")
MINOR_BREAK_RE = re.compile(r"[,;:，；：]+[\"')\]}]*$")


class NoSpeechDetected(RuntimeError):
    """Raised when VAD finds no speech that can be transcribed."""


def _is_empty_whisperx_batch_error(exc: IndexError) -> bool:
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            frame.f_code.co_name == "__call__"
            and frame.f_code.co_filename.endswith("transformers/pipelines/base.py")
        ):
            return str(exc) == "list index out of range"
        traceback = traceback.tb_next
    return False


WEAK_CONJUNCTIONS = {
    "and",
    "but",
    "or",
    "so",
    "yet",
    "because",
    "although",
    "though",
    "while",
    "whereas",
}


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
        "operator": DEVICE,
        "configured_device": WHISPER_RUNTIME_DEVICE,
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
            "align_local_files_only": WHISPERX_ALIGN_LOCAL_FILES_ONLY,
            "align_interpolate_method": WHISPERX_ALIGN_INTERPOLATE_METHOD,
            "regroup": {
                "max_chars": WHISPERX_REGROUP_MAX_CHARS,
                "max_duration_ms": WHISPERX_REGROUP_MAX_DURATION_MS,
            },
            "diarize": {
                "available": bool(WHISPERX_DIARIZE_HF_TOKEN),
                "min_speakers": WHISPERX_DIARIZE_MIN_SPEAKERS,
                "max_speakers": WHISPERX_DIARIZE_MAX_SPEAKERS,
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
        log.debug(
            "whisper loading model engine=%s model=%s model_path=%s configured_device=%s runtime_device=%s download_root=%s torch_home=%s",
            WHISPER_ENGINE,
            WHISPER_MODEL,
            _model_name_or_path(),
            WHISPER_RUNTIME_DEVICE,
            runtime_device,
            WHISPER_DOWNLOAD_ROOT or None,
            TORCH_HOME,
        )
        if WHISPER_ENGINE == "whisperx":
            import whisperx

            model_name_or_path = _model_name_or_path()
            if not Path(model_name_or_path).expanduser().exists():
                log.info("正在下载语音识别模型 %s（首次使用）", WHISPER_MODEL)
            _MODEL = whisperx.load_model(
                model_name_or_path,
                runtime_device,
                compute_type=WHISPERX_COMPUTE_TYPE,
                language=None,
                vad_method=WHISPERX_VAD_METHOD,
                vad_options=_whisperx_vad_options(),
                download_root=WHISPER_DOWNLOAD_ROOT or None,
            )
            log.debug(
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

        model_file = (
            Path(WHISPER_DOWNLOAD_ROOT).expanduser() / f"{WHISPER_MODEL}.pt"
            if WHISPER_DOWNLOAD_ROOT
            else None
        )
        if model_file is not None and not model_file.is_file():
            log.info("正在下载语音识别模型 %s（首次使用）", WHISPER_MODEL)

        # 加载 Whisper 模型
        # WHISPER_MODEL：模型名称，例如 tiny/base/small/medium/large 等
        # device()：运行设备，例如 cpu、cuda、mps
        # download_root：模型下载或缓存目录；如果为空，则交给 whisper 使用默认目录
        _MODEL = whisper.load_model(
            WHISPER_MODEL,
            device=runtime_device,
            download_root=WHISPER_DOWNLOAD_ROOT or None,
        )
        log.debug("whisper model loaded engine=%s model=%s runtime_device=%s", WHISPER_ENGINE, WHISPER_MODEL, runtime_device)

    # 返回已加载好的模型实例
    return _MODEL


def _to_ms(seconds: float) -> int:
    # 将秒转换成毫秒
    # Whisper 返回的时间通常是秒，这里统一转成毫秒，方便后续字幕/片段处理
    return int(round(float(seconds) * 1000))


def _convert_words(words: list) -> list:
    # 将 Whisper 返回的词级时间戳转换成统一格式
    converted = []
    for w in words or []:
        item = {
            # 单词文本，Whisper 中字段名通常是 word
            "text": w.get("word") or w.get("text") or "",

            # 单词开始时间，单位从秒转换成毫秒
            "start_time": int(w.get("start_time")) if w.get("start_time") is not None else _to_ms(w.get("start", 0.0)),

            # 单词结束时间，单位从秒转换成毫秒
            "end_time": int(w.get("end_time")) if w.get("end_time") is not None else _to_ms(w.get("end", 0.0)),
        }
        for key in (
            "_whisper_word_global_index",
            "_whisper_aligned_segment_index",
            "_whisper_aligned_segment_id",
            "_whisper_aligned_word_id",
        ):
            if key in w:
                item[key] = w[key]
        converted.append(item)
    return converted


def _convert_segments(segments: list) -> list:
    # 将 Whisper 返回的句子/片段级 segments 转换成统一的 utterances 格式
    converted = []
    for seg in segments:
        item = {
            # 当前语音片段的文本内容，去掉首尾空白
            "text": seg.get("text", "").strip(),

            # 当前片段开始时间，单位从秒转换成毫秒
            "start_time": _to_ms(seg.get("start", 0.0)),

            # 当前片段结束时间，单位从秒转换成毫秒
            "end_time": _to_ms(seg.get("end", 0.0)),

            # 当前片段中的词级时间戳信息
            "words": _convert_words(seg.get("words", [])),
        }
        for key in (
            "speaker",
            "_whisper_aligned_segment_index",
            "_whisper_aligned_segment_id",
            "_whisper_pysbd_index",
            "_whisper_split_id",
        ):
            if key in seg:
                item[key] = seg[key]
        converted.append(item)
    return converted


def _word_text(word: dict) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _is_punctuation_only(text: str) -> bool:
    value = str(text or "").strip()
    return bool(value) and all(
        unicodedata.category(char).startswith("P") for char in value
    )


def _narration_segments_without_punctuation_timing(
    segments: list[dict],
    language: str,
) -> tuple[list[dict], list[dict]]:
    """Remove punctuation timestamps and expose their duration as subtitle gaps."""
    cleaned_segments = []
    caption_segments = []

    for aligned_index, segment in enumerate(segments):
        source_words = segment.get("words") or []
        cleaned_words = [
            word for word in source_words if not _is_punctuation_only(_word_text(word))
        ]
        cleaned_segments.append({**segment, "words": cleaned_words})

        current_words: list[dict] = []
        trailing_punctuation = ""

        def flush() -> None:
            nonlocal current_words, trailing_punctuation
            if not current_words:
                trailing_punctuation = ""
                return
            caption = _segment_from_words(current_words, language)
            if caption is not None:
                caption["text"] = _join_word_texts(current_words, language)
                caption["_whisper_aligned_segment_index"] = aligned_index
                caption["_whisper_segmentation_method"] = (
                    "narration_punctuation_gap"
                )
                caption_segments.append(caption)
            current_words = []
            trailing_punctuation = ""

        for word in source_words:
            text = _word_text(word)
            if not text:
                continue
            if _is_punctuation_only(text):
                if current_words:
                    trailing_punctuation += text
                continue
            if trailing_punctuation:
                flush()
            if "start" in word and "end" in word:
                current_words.append(word)

        flush()

    return cleaned_segments, caption_segments


def _word_separator(language: str | None) -> str:
    return "" if _sentence_segmenter_language(language or "en") in {"zh", "ja"} else " "


def _join_word_texts(words: list[dict], language: str | None = None) -> str:
    return _word_separator(language).join(
        _word_text(word) for word in words if _word_text(word)
    ).strip()


def _assign_word_indexes(segments: list[dict]) -> None:
    global_word_index = 0
    for segment_index, segment in enumerate(segments):
        for word in segment.get("words") or []:
            if not _word_text(word):
                continue
            word["_whisper_word_global_index"] = global_word_index
            word["_whisper_aligned_segment_index"] = segment_index
            global_word_index += 1


def _speaker_label(item: dict | None) -> str | None:
    if not item:
        return None
    speaker = str(item.get("speaker") or "").strip()
    return speaker or None


def _speaker_from_words(words: list[dict]) -> str | None:
    counts: dict[str, int] = {}
    first_label: str | None = None
    for word in words:
        speaker = _speaker_label(word)
        if not speaker:
            continue
        first_label = first_label or speaker
        counts[speaker] = counts.get(speaker, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda label: (counts[label], label == first_label))


def _patch_hf_hub_download_auth_alias() -> None:
    try:
        import huggingface_hub
    except Exception:
        return

    original = getattr(huggingface_hub, "hf_hub_download", None)
    if original is None or "use_auth_token" in inspect.signature(original).parameters:
        return
    if getattr(original, "_ydbi_accepts_use_auth_token", False):
        return

    def compatible_hf_hub_download(*args, **kwargs):
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        return original(*args, **kwargs)

    compatible_hf_hub_download._ydbi_accepts_use_auth_token = True
    huggingface_hub.hf_hub_download = compatible_hf_hub_download
    for module_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
        "pyannote.audio.pipelines.utils.getter",
    ):
        try:
            module = importlib.import_module(module_name)
            module.hf_hub_download = compatible_hf_hub_download
        except Exception:
            pass


def _patch_torch_load_for_pyannote_checkpoint() -> None:
    try:
        import torch
    except Exception:
        return

    original = torch.load
    if getattr(original, "_ydbi_pyannote_weights_only_default", False):
        return

    def compatible_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    compatible_torch_load._ydbi_pyannote_weights_only_default = True
    torch.load = compatible_torch_load


def _segment_from_words(words: list[dict], language: str | None = None) -> dict | None:
    timed_words = [word for word in words if "start" in word and "end" in word]
    text = _join_word_texts(words, language)
    if not text or not timed_words:
        return None
    first_word = words[0] if words else {}
    segment = {
        "text": text,
        "start": float(timed_words[0]["start"]),
        "end": float(timed_words[-1]["end"]),
        "words": words,
        "_whisper_aligned_segment_index": first_word.get("_whisper_aligned_segment_index"),
    }
    speaker = _speaker_from_words(words)
    if speaker:
        segment["speaker"] = speaker
    return segment


def _segment_too_long(segment: dict) -> bool:
    text = str(segment.get("text") or "")
    duration_ms = int(round((float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)) * 1000))
    return len(text) >= WHISPERX_REGROUP_MAX_CHARS or duration_ms >= WHISPERX_REGROUP_MAX_DURATION_MS


def _long_split_reason(segment: dict) -> str:
    text_too_long = len(str(segment.get("text") or "")) >= WHISPERX_REGROUP_MAX_CHARS
    duration_ms = int(round((float(segment.get("end") or 0.0) - float(segment.get("start") or 0.0)) * 1000))
    duration_too_long = duration_ms >= WHISPERX_REGROUP_MAX_DURATION_MS
    if text_too_long and duration_too_long:
        return "text_and_duration_too_long"
    if text_too_long:
        return "text_too_long"
    if duration_too_long:
        return "duration_too_long"
    return "none"


_WTPSPLIT_MODEL = None
_WTPSPLIT_UNAVAILABLE = False


def _word_text_length(words: list[dict], language: str | None = None) -> int:
    return len(_join_word_texts(words, language))


def _split_punctuation_value(text: str) -> str | None:
    match = re.search(r"[,;:，；：]+", text)
    return match.group(0) if match else None


def _normalized_word_text(text: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", text.lower())


def _minor_punctuation_split_at(words: list[dict]) -> int | None:
    candidates = [idx + 1 for idx, word in enumerate(words[:-1]) if MINOR_BREAK_RE.search(_word_text(word))]
    if not candidates:
        return None

    total_length = _word_text_length(words)
    midpoint = total_length / 2
    return min(
        candidates,
        key=lambda split_at: (
            abs(_word_text_length(words[:split_at]) - midpoint),
            abs(_word_text_length(words[split_at:]) - midpoint),
        ),
    )


def _weak_conjunction_split_at(words: list[dict]) -> int | None:
    candidates = [
        idx
        for idx, word in enumerate(words[1:-1], start=1)
        if _normalized_word_text(_word_text(word)) in WEAK_CONJUNCTIONS
    ]
    if not candidates:
        return None

    total_length = _word_text_length(words)
    midpoint = total_length / 2
    return min(
        candidates,
        key=lambda split_at: (
            abs(_word_text_length(words[:split_at]) - midpoint),
            abs(_word_text_length(words[split_at:]) - midpoint),
        ),
    )


def _wtpsplit_model():
    global _WTPSPLIT_MODEL, _WTPSPLIT_UNAVAILABLE
    if _WTPSPLIT_MODEL is not None:
        return _WTPSPLIT_MODEL
    if _WTPSPLIT_UNAVAILABLE:
        return None
    try:
        from wtpsplit import SaT
        try:
            from transformers.utils.logging import disable_progress_bar

            disable_progress_bar()
        except Exception:
            pass
    except Exception as exc:
        try:
            from wtpsplit import WtP
        except Exception as fallback_exc:
            _WTPSPLIT_UNAVAILABLE = True
            log.warning("语义分句组件不可用，长句将保持原样")
            log.debug("wtpsplit import failed", exc_info=fallback_exc)
            return None

        try:
            _WTPSPLIT_MODEL = WtP("wtp-bert-mini", ignore_legacy_warning=True)
        except Exception as fallback_exc:
            _WTPSPLIT_UNAVAILABLE = True
            log.warning("语义分句模型加载失败，长句将保持原样")
            log.debug("wtpsplit WtP model load failed", exc_info=fallback_exc)
            return None
        return _WTPSPLIT_MODEL

    try:
        model_name: str | Path = "sat-3l-sm"
        tokenizer_name: str | Path = "facebookAI/xlm-roberta-base"
        from_pretrained_kwargs = None

        try:
            from huggingface_hub import snapshot_download

            model_snapshot = Path(
                snapshot_download(
                    "segment-any-text/sat-3l-sm",
                    local_files_only=True,
                )
            )
            tokenizer_snapshot = Path(
                snapshot_download(
                    "facebookAI/xlm-roberta-base",
                    local_files_only=True,
                )
            )
            model_is_complete = (
                (model_snapshot / "config.json").is_file()
                and any(
                    (model_snapshot / filename).is_file()
                    for filename in ("model.safetensors", "pytorch_model.bin")
                )
            )
            tokenizer_is_complete = (
                (tokenizer_snapshot / "tokenizer_config.json").is_file()
                and any(
                    (tokenizer_snapshot / filename).is_file()
                    for filename in ("tokenizer.json", "sentencepiece.bpe.model")
                )
            )
            if not model_is_complete or not tokenizer_is_complete:
                raise FileNotFoundError("incomplete wtpsplit cache")
            model_name = model_snapshot
            tokenizer_name = tokenizer_snapshot
            from_pretrained_kwargs = {"local_files_only": True}
        except Exception:
            log.info("正在下载语义分句模型 sat-3l-sm（首次使用）")

        _WTPSPLIT_MODEL = SaT(
            model_name,
            tokenizer_name_or_path=tokenizer_name,
            from_pretrained_kwargs=from_pretrained_kwargs,
        )
    except Exception as exc:
        _WTPSPLIT_UNAVAILABLE = True
        log.warning("语义分句模型加载失败，长句将保持原样")
        log.debug("wtpsplit SaT model load failed", exc_info=exc)
        return None
    return _WTPSPLIT_MODEL


def _semantic_parts(model: object, full_text: str, lang: str) -> list[str]:
    try:
        return [
            part.strip()
            for part in model.split(
                full_text,
                max_length=WHISPERX_REGROUP_MAX_CHARS,
                prior_type="gaussian",
                prior_kwargs={"lang_code": lang},
            )
            if part.strip()
        ]
    except TypeError:
        try:
            return [
                part.strip()
                for part in model.split(
                    full_text,
                    lang_code=lang,
                    max_length=WHISPERX_REGROUP_MAX_CHARS,
                )
                if part.strip()
            ]
        except TypeError:
            return [part.strip() for part in model.split(full_text, lang_code=lang) if part.strip()]


def _semantic_split_at(words: list[dict], language: str | None = None) -> int | None:
    if len(words) <= 1:
        return None

    model = _wtpsplit_model()
    if model is None:
        return None

    text_parts = []
    word_spans = []
    cursor = 0
    for word in words:
        text = _word_text(word)
        if not text:
            continue
        separator = _word_separator(language)
        if text_parts and separator:
            text_parts.append(separator)
            cursor += len(separator)
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        word_spans.append((start, cursor))

    full_text = "".join(text_parts).strip()
    if not full_text or len(word_spans) <= 1:
        return None

    lang = _sentence_segmenter_language(language or "en")
    try:
        semantic_parts = _semantic_parts(model, full_text, lang)
    except Exception as exc:
        log.warning("语义分句失败，长句将保持原样")
        log.debug("wtpsplit split failed", exc_info=exc)
        return None

    if len(semantic_parts) <= 1:
        return None

    boundaries = []
    search_from = 0
    for part in semantic_parts[:-1]:
        start = full_text.find(part, search_from)
        if start < 0:
            return None
        end = start + len(part)
        boundaries.append(end)
        search_from = end

    split_candidates = []
    for boundary in boundaries:
        split_at = None
        for index, (_, word_end) in enumerate(word_spans):
            if word_end >= boundary:
                split_at = index + 1
                break
        if split_at is not None and 0 < split_at < len(words):
            split_candidates.append(split_at)

    if not split_candidates:
        return None

    midpoint = len(full_text) / 2
    return min(split_candidates, key=lambda split_at: abs(word_spans[split_at - 1][1] - midpoint))


def _split_long_segment_on_minor_punctuation(segment: dict, pysbd_index: int | None = None) -> list[dict]:
    if not _segment_too_long(segment):
        return [{
            **segment,
            "_whisper_pysbd_index": pysbd_index,
            "_whisper_split_applied": False,
            "_whisper_split_reason": "none",
        }]

    words = segment.get("words") or []
    if not words:
        return [{
            **segment,
            "_whisper_pysbd_index": pysbd_index,
            "_whisper_split_applied": False,
            "_whisper_split_reason": _long_split_reason(segment),
        }]

    split_segments = []
    current_words = []
    split_reason = _long_split_reason(segment)
    original_text = str(segment.get("text") or "").strip()
    language = segment.get("_whisper_pysbd_language")

    def append_group(
        group_words: list[dict],
        *,
        split_applied: bool,
        split_at_word_index: int | None = None,
        split_punctuation: str | None = None,
        split_conjunction: str | None = None,
        split_method: str | None = None,
    ) -> None:
        grouped = _segment_from_words(group_words, language)
        if grouped is not None:
            grouped["_whisper_pysbd_index"] = pysbd_index
            grouped["_whisper_split_applied"] = split_applied
            grouped["_whisper_split_reason"] = split_reason
            grouped["_whisper_split_trigger"] = split_reason
            grouped["_whisper_split_method"] = split_method or ("none" if not split_applied else "unknown")
            grouped["_whisper_split_at_word_index"] = split_at_word_index
            grouped["_whisper_split_punctuation"] = split_punctuation
            grouped["_whisper_split_conjunction"] = split_conjunction
            grouped["_whisper_original_text"] = original_text
            split_segments.append(grouped)

    def flush_current() -> None:
        nonlocal current_words
        append_group(current_words, split_applied=bool(split_segments))
        current_words = []

    for word in words:
        text = _word_text(word)
        if not text:
            continue
        current_words.append(word)
        current_segment = _segment_from_words(current_words, language)
        if current_segment is None or not _segment_too_long(current_segment):
            continue

        split_at = _minor_punctuation_split_at(current_words)
        split_method = "minor_punctuation" if split_at is not None else None
        split_conjunction = None

        if split_at is None:
            split_at = _weak_conjunction_split_at(current_words)
            split_method = "weak_conjunction" if split_at is not None else None
            split_conjunction = _normalized_word_text(_word_text(current_words[split_at])) if split_at is not None else None

        if split_at is None:
            split_at = _semantic_split_at(current_words, language)
            split_method = "semantic_wtpsplit" if split_at is not None else None

        if split_at:
            split_word = current_words[split_at - 1]
            append_group(
                current_words[:split_at],
                split_applied=True,
                split_at_word_index=split_at - 1,
                split_punctuation=_split_punctuation_value(_word_text(split_word)),
                split_conjunction=split_conjunction,
                split_method=split_method,
            )
            current_words = current_words[split_at:]

    flush_current()
    if split_segments:
        last_method = None
        last_punctuation = None
        last_conjunction = None
        for grouped in split_segments:
            if grouped.get("_whisper_split_method") not in {None, "unknown", "none"}:
                last_method = grouped.get("_whisper_split_method")
            elif last_method is not None:
                grouped["_whisper_split_method"] = last_method
            if grouped.get("_whisper_split_punctuation"):
                last_punctuation = grouped.get("_whisper_split_punctuation")
            elif last_punctuation is not None and grouped.get("_whisper_split_method") == "minor_punctuation":
                grouped["_whisper_split_punctuation"] = last_punctuation
            if grouped.get("_whisper_split_conjunction"):
                last_conjunction = grouped.get("_whisper_split_conjunction")
            elif last_conjunction is not None and grouped.get("_whisper_split_method") == "weak_conjunction":
                grouped["_whisper_split_conjunction"] = last_conjunction

        source_part_index = 1
        source_part_count = len(split_segments)
        for grouped in split_segments:
            grouped["_whisper_original_part_index"] = source_part_index
            grouped["_whisper_original_part_count"] = source_part_count
            source_part_index += 1
    return split_segments or [{
        **segment,
        "_whisper_pysbd_index": pysbd_index,
        "_whisper_split_applied": False,
        "_whisper_split_reason": split_reason,
        "_whisper_split_trigger": split_reason,
        "_whisper_split_method": "none",
        "_whisper_original_text": original_text,
    }]


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
        log.warning("句子切分组件不可用，已改用标点切分")
        log.debug("pysbd import failed", exc_info=exc)
        return []

    word_spans = []
    text_parts = []
    cursor = 0
    for word in words:
        text = _word_text(word)
        if not text:
            continue
        separator = _word_separator(language)
        if text_parts and separator:
            text_parts.append(separator)
            cursor += len(separator)
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
        log.warning("句子切分失败，已改用标点切分")
        log.debug("pysbd segment failed language=%s", segmenter_language, exc_info=exc)
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
        segment = _segment_from_words(sentence_words, language)
        if segment is not None:
            segment["_whisper_pysbd_language"] = segmenter_language
            segment["_whisper_segmentation_method"] = "pysbd"
            segments.append(segment)
        search_from = sentence_end

    return segments


def _regroup_words_by_punctuation(words: list[dict], language: str) -> list[dict]:
    regrouped = []
    current_words = []

    def flush() -> None:
        nonlocal current_words
        segment = _segment_from_words(current_words, language)
        if segment is None:
            current_words = []
            return
        segment["_whisper_segmentation_method"] = "punctuation"
        regrouped.append(segment)
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


def _regroup_aligned_segments(segments: list[dict], language: str) -> tuple[list[dict], list[dict]]:
    pysbd_segments = []
    current_words = []
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current_words, current_speaker
        if not current_words:
            return
        sentence_segments = _segment_words_with_pysbd(current_words, language)
        if not sentence_segments:
            sentence_segments = _regroup_words_by_punctuation(current_words, language)
        pysbd_segments.extend(sentence_segments)
        current_words = []
        current_speaker = None

    for segment in segments:
        words = segment.get("words") or []
        if not words:
            flush()
            pysbd_segments.append({
                **segment,
                "_whisper_segmentation_method": "aligned_no_words",
            })
            continue

        for word in words:
            text = _word_text(word)
            if not text:
                continue
            if "start" not in word or "end" not in word:
                flush()
                continue
            speaker = _speaker_label(word)
            if current_words and speaker and current_speaker and speaker != current_speaker:
                flush()
            current_words.append(word)
            if speaker:
                current_speaker = speaker

    flush()
    if not pysbd_segments:
        pysbd_segments = [
            {
                **segment,
                "_whisper_segmentation_method": "aligned_passthrough",
            }
            for segment in segments
        ]

    for index, segment in enumerate(pysbd_segments):
        segment["_whisper_pysbd_index"] = index

    long_split_segments = []
    for index, segment in enumerate(pysbd_segments):
        long_split_segments.extend(_split_long_segment_on_minor_punctuation(segment, pysbd_index=index))

    return pysbd_segments, long_split_segments


def _local_align_model_name(whisperx, language: str) -> str | None:
    configured_model = WHISPERX_ALIGN_MODEL or None
    if configured_model and Path(configured_model).expanduser().exists():
        return str(Path(configured_model).expanduser())

    alignment_module = getattr(whisperx, "alignment", None)
    if alignment_module is None:
        alignment_module = importlib.import_module("whisperx.alignment")
    hf_models = getattr(alignment_module, "DEFAULT_ALIGN_MODELS_HF", {})
    model_name = configured_model or hf_models.get(language)
    if not model_name or "/" not in model_name:
        return model_name

    cache_root = Path(WHISPERX_ALIGN_MODEL_DIR).expanduser()
    repo_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    main_ref = repo_cache / "refs" / "main"
    if not main_ref.is_file():
        raise RuntimeError(
            "WhisperX align model is not available in the local cache: "
            f"model={model_name}, dir={cache_root}"
        )
    revision = main_ref.read_text(encoding="utf-8").strip()
    snapshot = repo_cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError(
            "WhisperX align model snapshot is missing from the local cache: "
            f"model={model_name}, revision={revision}, dir={cache_root}"
        )
    return str(snapshot)


def _load_align_model(whisperx, language: str, runtime_device: str):
    kwargs = {
        "model_name": WHISPERX_ALIGN_MODEL or None,
        "model_dir": WHISPERX_ALIGN_MODEL_DIR,
    }
    try:
        return whisperx.load_align_model(
            language,
            runtime_device,
            model_cache_only=WHISPERX_ALIGN_LOCAL_FILES_ONLY,
            **kwargs,
        )
    except TypeError as exc:
        if "model_cache_only" not in str(exc):
            raise
        if WHISPERX_ALIGN_LOCAL_FILES_ONLY:
            kwargs["model_name"] = _local_align_model_name(whisperx, language)
        return whisperx.load_align_model(language, runtime_device, **kwargs)


def _align_whisperx_result(
    whisperx,
    result: dict,
    audio: object,
    language: str,
    runtime_device: str,
    task_id: str | None = None,
) -> dict:
    if not WHISPERX_ALIGN or not result.get("segments"):
        return result

    align_language = str(result.get("language") or language or "en").lower()
    task_label = task_id or "本地任务"
    log.info("任务 %s：正在加载时间轴对齐模型", task_label)
    log.debug(
        "whisperx align model loading language=%s model=%s model_dir=%s runtime_device=%s",
        align_language,
        WHISPERX_ALIGN_MODEL or None,
        WHISPERX_ALIGN_MODEL_DIR,
        runtime_device,
    )
    Path(WHISPERX_ALIGN_MODEL_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    align_key = (
        align_language,
        runtime_device,
        WHISPERX_ALIGN_MODEL or "",
        str(Path(WHISPERX_ALIGN_MODEL_DIR).expanduser()),
    )
    cached_align = _ALIGN_MODELS.get(align_key)
    if cached_align is None:
        try:
            cached_align = _load_align_model(
                whisperx, align_language, runtime_device
            )
        except (OSError, ValueError) as exc:
            if WHISPERX_ALIGN_LOCAL_FILES_ONLY:
                raise RuntimeError(
                    "WhisperX align model is not available in the local cache: "
                    f"language={align_language}, dir={WHISPERX_ALIGN_MODEL_DIR}"
                ) from exc
            raise
        _ALIGN_MODELS[align_key] = cached_align
    align_model, align_metadata = cached_align
    log.info(
        "任务 %s：正在进行时间轴对齐，原始片段 %d 段",
        task_label,
        len(result.get("segments") or []),
    )
    log.debug(
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
    aligned["text"] = " ".join(str(seg.get("text") or "").strip() for seg in aligned_segments).strip()
    aligned["language"] = align_language
    log.info(
        "任务 %s：时间轴对齐完成，共 %d 段、%d 个词",
        task_label,
        len(aligned_segments),
        len(aligned.get("word_segments") or []),
    )
    log.debug(
        "whisperx align done language=%s aligned_segments=%s word_segments=%s",
        align_language,
        len(aligned_segments),
        len(aligned.get("word_segments") or []),
    )
    return aligned


def _diarize_whisperx_result(
    whisperx,
    result: dict,
    audio: object,
    runtime_device: str,
    *,
    task_id: str | None = None,
) -> dict:
    hf_token = _huggingface_token() or WHISPERX_DIARIZE_HF_TOKEN
    if not hf_token:
        raise RuntimeError(
            "WhisperX speaker diarization requires YDBI_WHISPERX_DIARIZE_HF_TOKEN "
            "or HF_TOKEN."
        )

    task_label = task_id or "本地任务"
    _patch_hf_hub_download_auth_alias()
    _patch_torch_load_for_pyannote_checkpoint()
    from whisperx.diarize import DiarizationPipeline

    model_key = (runtime_device, hf_token)
    diarize_model = _DIARIZATION_MODELS.get(model_key)
    if diarize_model is None:
        log.info("任务 %s：正在加载说话人分离模型", task_label)
        init_params = inspect.signature(DiarizationPipeline.__init__).parameters
        token_kwargs = (
            {"use_auth_token": hf_token}
            if "use_auth_token" in init_params
            else {"token": hf_token}
        )
        diarize_model = DiarizationPipeline(device=runtime_device, **token_kwargs)
        _DIARIZATION_MODELS[model_key] = diarize_model

    kwargs = {}
    if WHISPERX_DIARIZE_MIN_SPEAKERS is not None:
        kwargs["min_speakers"] = WHISPERX_DIARIZE_MIN_SPEAKERS
    if WHISPERX_DIARIZE_MAX_SPEAKERS is not None:
        kwargs["max_speakers"] = WHISPERX_DIARIZE_MAX_SPEAKERS

    log.info("任务 %s：正在区分说话人", task_label)
    diarize_segments = diarize_model(audio, **kwargs)
    assigned = whisperx.assign_word_speakers(diarize_segments, result)
    speaker_count = len(
        {
            speaker
            for segment in assigned.get("segments") or []
            for speaker in [str(segment.get("speaker") or "").strip()]
            if speaker
        }
    )
    log.info("任务 %s：说话人区分完成，共 %d 个说话人", task_label, speaker_count)
    return assigned


def _load_diarization_pipeline(runtime_device: str, task_label: str):
    hf_token = _huggingface_token() or WHISPERX_DIARIZE_HF_TOKEN
    if not hf_token:
        raise RuntimeError(
            "WhisperX speaker diarization requires YDBI_WHISPERX_DIARIZE_HF_TOKEN "
            "or HF_TOKEN."
        )

    _patch_hf_hub_download_auth_alias()
    _patch_torch_load_for_pyannote_checkpoint()
    from whisperx.diarize import DiarizationPipeline

    model_key = (runtime_device, hf_token)
    diarize_model = _DIARIZATION_MODELS.get(model_key)
    if diarize_model is not None:
        return diarize_model

    log.info("任务 %s：正在加载说话人分离模型", task_label)
    init_params = inspect.signature(DiarizationPipeline.__init__).parameters
    token_kwargs = (
        {"use_auth_token": hf_token}
        if "use_auth_token" in init_params
        else {"token": hf_token}
    )
    diarize_model = DiarizationPipeline(device=runtime_device, **token_kwargs)
    _DIARIZATION_MODELS[model_key] = diarize_model
    return diarize_model


def _diarization_rows(diarize_segments: object) -> list[dict]:
    if hasattr(diarize_segments, "to_dict"):
        return list(diarize_segments.to_dict("records"))
    return []


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _assign_segment_speakers_from_diarization(
    segments: list[dict],
    diarize_segments: object,
) -> None:
    rows = _diarization_rows(diarize_segments)
    if not rows:
        return

    for segment in segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        scores: dict[str, float] = {}
        for row in rows:
            speaker = str(row.get("speaker") or "").strip()
            if not speaker:
                continue
            row_start = float(row.get("start") or 0.0)
            row_end = float(row.get("end") or row_start)
            overlap = _overlap_seconds(start, end, row_start, row_end)
            if overlap > 0:
                scores[speaker] = scores.get(speaker, 0.0) + overlap
        if scores:
            segment["speaker"] = max(scores, key=scores.get)


def _diarize_raw_segments(
    result: dict,
    audio: object,
    runtime_device: str,
    *,
    task_id: str | None = None,
) -> dict:
    task_label = task_id or "本地任务"
    diarize_model = _load_diarization_pipeline(runtime_device, task_label)

    kwargs = {}
    if WHISPERX_DIARIZE_MIN_SPEAKERS is not None:
        kwargs["min_speakers"] = WHISPERX_DIARIZE_MIN_SPEAKERS
    if WHISPERX_DIARIZE_MAX_SPEAKERS is not None:
        kwargs["max_speakers"] = WHISPERX_DIARIZE_MAX_SPEAKERS

    log.info("任务 %s：正在区分说话人", task_label)
    diarize_segments = diarize_model(audio, **kwargs)
    segments = result.get("segments") or []
    _assign_segment_speakers_from_diarization(segments, diarize_segments)
    speaker_count = len(
        {
            str(segment.get("speaker") or "").strip()
            for segment in segments
            if str(segment.get("speaker") or "").strip()
        }
    )
    log.info("任务 %s：说话人区分完成，共 %d 个说话人", task_label, speaker_count)
    return {
        **result,
        "segments": segments,
        "text": " ".join(str(seg.get("text") or "").strip() for seg in segments).strip(),
    }


def _transcribe(model: object, vocals_file: Path, language: str, runtime_device: str, word_timestamps: bool) -> tuple[dict, object | None]:
    if WHISPER_ENGINE == "whisperx":
        import whisperx

        log.debug("whisperx loading audio file=%s", vocals_file)
        audio = whisperx.load_audio(str(vocals_file))
        log.debug(
            "whisperx transcribe start file=%s language=%s batch_size=%s chunk_size=%s runtime_device=%s",
            vocals_file,
            language,
            WHISPERX_BATCH_SIZE,
            WHISPERX_CHUNK_SIZE,
            runtime_device,
        )
        try:
            result = model.transcribe(
                audio,
                batch_size=WHISPERX_BATCH_SIZE,
                language=language,
                chunk_size=WHISPERX_CHUNK_SIZE,
                verbose=False,
            )
        except IndexError as exc:
            if not _is_empty_whisperx_batch_error(exc):
                raise
            raise NoSpeechDetected(
                "WhisperX VAD found no active speech in the input audio."
            ) from exc
        segments = result.get("segments", [])
        result["text"] = " ".join(str(seg.get("text") or "").strip() for seg in segments).strip()
        log.debug(
            "whisperx transcribe done file=%s language=%s segments=%s text_chars=%s",
            vocals_file,
            result.get("language") or language,
            len(segments),
            len(result["text"]),
        )
        return result, audio

    log.debug(
        "openai whisper transcribe start file=%s language=%s runtime_device=%s word_timestamps=%s",
        vocals_file,
        language,
        runtime_device,
        word_timestamps,
    )
    return (
        model.transcribe(
            str(vocals_file),
            language=language,
            word_timestamps=word_timestamps,
            verbose=False,
        ),
        None,
    )


def align_known_text(
    vocals_file: Path,
    known_segments: list[dict],
    language: str,
    *,
    task_id: str | None = None,
    run_id: int | None = None,
) -> dict:
    from pydub import AudioSegment
    import whisperx

    if WHISPER_ENGINE != "whisperx":
        raise RuntimeError("Known-text alignment requires the whisperx engine.")
    if not WHISPERX_ALIGN:
        raise RuntimeError("Known-text alignment requires WhisperX alignment to be enabled.")

    duration_ms = len(AudioSegment.from_file(vocals_file))
    transcript = []
    previous_end = 0
    for expected_index, segment in enumerate(known_segments):
        item_index = int(segment.get("item_index"))
        status = str(segment.get("status") or "").strip().lower()
        text = str(segment.get("text") or "").strip()
        start_time = int(segment.get("start_time") or 0)
        end_time = int(segment.get("end_time") or 0)
        if item_index != expected_index:
            raise ValueError(
                f"Narration segment indexes are not contiguous at {item_index}."
            )
        if status and status != "success":
            raise ValueError(
                f"Narration segment {item_index} is not successful: {status}."
            )
        if not text:
            raise ValueError(f"Narration segment {item_index} has empty text.")
        if (
            start_time < previous_end
            or end_time <= start_time
            or end_time > duration_ms + 20
        ):
            raise ValueError(
                f"Narration segment {item_index} has invalid timing "
                f"{start_time}-{end_time} for audio duration {duration_ms}."
            )
        end_time = min(end_time, duration_ms)
        transcript.append(
            {
                "text": text,
                "start": start_time / 1000.0,
                "end": end_time / 1000.0,
            }
        )
        previous_end = end_time

    runtime_device = _runtime_device_for_engine()
    audio = whisperx.load_audio(str(vocals_file))
    raw_result = {
        "segments": transcript,
        "text": _word_separator(language).join(
            segment["text"] for segment in transcript
        ),
        "language": language,
    }
    if run_id is not None:
        db.update_whisper_run_runtime(
            run_id,
            runtime_device=runtime_device,
            model_path=WHISPERX_ALIGN_MODEL or "default-align-model",
            input_duration_ms=duration_ms,
        )

    raw_segment_ids: dict[int, int] = {}
    if task_id is not None and run_id is not None:
        raw_segment_ids = db.save_whisper_raw_segments(
            run_id, task_id, raw_result["segments"]
        )

    aligned_result = _align_whisperx_result(
        whisperx,
        raw_result,
        audio,
        language,
        runtime_device,
        task_id=task_id,
    )
    aligned_segments, narration_segments = (
        _narration_segments_without_punctuation_timing(
            aligned_result.get("segments") or [],
            language,
        )
    )
    _assign_word_indexes(aligned_segments)
    word_ids: dict[int, int] = {}
    aligned_segment_ids: dict[int, int] = {}
    if task_id is not None and run_id is not None:
        aligned_segment_ids = db.save_whisper_aligned_segments(
            run_id,
            task_id,
            aligned_segments,
            raw_segment_ids,
        )
        for segment_index, segment in enumerate(aligned_segments):
            segment["_whisper_aligned_segment_id"] = aligned_segment_ids.get(
                segment_index
            )
            for word in segment.get("words") or []:
                word["_whisper_aligned_segment_id"] = aligned_segment_ids.get(
                    segment_index
                )
        word_ids = db.save_whisper_aligned_words(
            run_id, task_id, aligned_segments, aligned_segment_ids
        )
        for segment in aligned_segments:
            for word in segment.get("words") or []:
                global_index = word.get("_whisper_word_global_index")
                if global_index in word_ids:
                    word["_whisper_aligned_word_id"] = word_ids[global_index]

    pysbd_segments = narration_segments
    final_segments = narration_segments
    if task_id is not None and run_id is not None:
        for segment in pysbd_segments:
            aligned_index = segment.get("_whisper_aligned_segment_index")
            if aligned_index is not None:
                segment["_whisper_aligned_segment_id"] = aligned_segment_ids.get(
                    int(aligned_index)
                )
        pysbd_segment_ids = db.save_whisper_pysbd_segments(
            run_id, task_id, pysbd_segments, word_ids
        )
        split_ids = db.save_whisper_splits(
            run_id,
            task_id,
            final_segments,
            pysbd_segment_ids,
            word_ids,
        )
        for segment_index, segment in enumerate(final_segments):
            segment["_whisper_split_id"] = split_ids.get(segment_index)
            for word in segment.get("words") or []:
                global_index = word.get("_whisper_word_global_index")
                if global_index in word_ids:
                    word["_whisper_aligned_word_id"] = word_ids[global_index]

    utterances = _convert_segments(final_segments)
    if not utterances:
        raise RuntimeError("Known-text alignment did not return any segments.")
    return {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": _word_separator(language).join(
                str(segment.get("text") or "").strip()
                for segment in final_segments
            ),
            "utterances": utterances,
        },
    }


def recognize_speech(
    vocals_file: Path,
    session: Path,
    language: str,
    *,
    task_id: str | None = None,
    run_id: int | None = None,
    diarize: bool = False,
) -> dict:
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
    runtime_device = str(getattr(model, "device", _runtime_device_for_engine())).lower()
    if run_id is not None:
        db.update_whisper_run_runtime(run_id, runtime_device=runtime_device, model_path=_model_name_or_path())

    # 判断是否启用词级时间戳
    # 在 MPS 设备上，Whisper 的 word_timestamps 可能触发 float64 DTW 相关问题
    # 因此如果运行在 mps 上，就关闭词级时间戳
    word_timestamps = WHISPER_ENGINE == "openai" and _word_timestamps_for(runtime_device)

    # 如果 OpenAI Whisper 在 MPS 上关闭了词级时间戳，打印 warning，方便排查为什么 words 为空
    if WHISPER_ENGINE == "openai" and not word_timestamps:
        log.warning("当前设备不支持词级时间戳，已自动关闭")
    log.debug(
        "whisper runtime audio=%s engine=%s model=%s configured_device=%s runtime_device=%s language=%s word_timestamps=%s vad_method=%s vad_options=%s align=%s diarize=%s",
        vocals_file,
        WHISPER_ENGINE,
        WHISPER_MODEL,
        WHISPER_RUNTIME_DEVICE,
        runtime_device,
        language,
        word_timestamps,
        WHISPERX_VAD_METHOD if WHISPER_ENGINE == "whisperx" else None,
        _whisperx_vad_options() if WHISPER_ENGINE == "whisperx" else None,
        WHISPERX_ALIGN if WHISPER_ENGINE == "whisperx" else None,
        diarize,
    )

    if diarize and WHISPER_ENGINE != "whisperx":
        raise RuntimeError("Speaker diarization requires the whisperx engine.")

    # 调用 Whisper 进行语音识别
    # vocals_file：待识别的人声文件
    # language：指定识别语言
    # word_timestamps：是否返回词级时间戳
    # verbose=False：关闭详细输出
    raw_result, audio = _transcribe(model, vocals_file, language, runtime_device, word_timestamps)
    result = raw_result
    raw_segment_ids: dict[int, int] = {}
    if task_id is not None and run_id is not None:
        raw_segment_ids = db.save_whisper_raw_segments(run_id, task_id, raw_result.get("segments", []))

    if diarize:
        if audio is None:
            raise RuntimeError("Speaker diarization requires loaded WhisperX audio.")
        result = _diarize_raw_segments(
            raw_result,
            audio,
            runtime_device,
            task_id=task_id,
        )
        utterances = _convert_segments(result.get("segments", []))
        log.debug("whisper dialogue result converted audio=%s utterances=%s", vocals_file, len(utterances))
        if not utterances:
            raise RuntimeError("Whisper dialogue did not return any segments.")
        duration_ms = len(AudioSegment.from_file(vocals_file))
        if run_id is not None:
            db.update_whisper_run_runtime(
                run_id,
                runtime_device=runtime_device,
                model_path=_model_name_or_path(),
                input_duration_ms=duration_ms,
            )
        return {
            "audio_info": {"duration": duration_ms},
            "result": {
                "text": (result.get("text") or "").strip(),
                "utterances": utterances,
            },
        }

    if WHISPER_ENGINE == "whisperx" and audio is not None:
        import whisperx

        aligned_result = _align_whisperx_result(
            whisperx,
            raw_result,
            audio,
            language,
            runtime_device,
            task_id=task_id,
        )
        if diarize:
            aligned_result = _diarize_whisperx_result(
                whisperx,
                aligned_result,
                audio,
                runtime_device,
                task_id=task_id,
            )
        aligned_segments = aligned_result.get("segments") or []
        _assign_word_indexes(aligned_segments)
        if task_id is not None and run_id is not None:
            aligned_segment_ids = db.save_whisper_aligned_segments(
                run_id,
                task_id,
                aligned_segments,
                raw_segment_ids,
            )
            for segment_index, segment in enumerate(aligned_segments):
                segment["_whisper_aligned_segment_id"] = aligned_segment_ids.get(segment_index)
                for word in segment.get("words") or []:
                    word["_whisper_aligned_segment_id"] = aligned_segment_ids.get(segment_index)
            word_ids = db.save_whisper_aligned_words(run_id, task_id, aligned_segments, aligned_segment_ids)
            for segment in aligned_segments:
                for word in segment.get("words") or []:
                    global_index = word.get("_whisper_word_global_index")
                    if global_index in word_ids:
                        word["_whisper_aligned_word_id"] = word_ids[global_index]
        else:
            word_ids = {}

        align_language = str(aligned_result.get("language") or language or "en").lower()
        pysbd_segments, split_checked_segments = _regroup_aligned_segments(aligned_segments, align_language)
        if task_id is not None and run_id is not None:
            for segment in pysbd_segments:
                aligned_index = segment.get("_whisper_aligned_segment_index")
                if aligned_index is not None:
                    segment["_whisper_aligned_segment_id"] = aligned_segment_ids.get(int(aligned_index))
            pysbd_segment_ids = db.save_whisper_pysbd_segments(run_id, task_id, pysbd_segments, word_ids)
            split_ids = db.save_whisper_splits(
                run_id,
                task_id,
                split_checked_segments,
                pysbd_segment_ids,
                word_ids,
            )
            for segment_index, segment in enumerate(split_checked_segments):
                segment["_whisper_split_id"] = split_ids.get(segment_index)
                for word in segment.get("words") or []:
                    global_index = word.get("_whisper_word_global_index")
                    if global_index in word_ids:
                        word["_whisper_aligned_word_id"] = word_ids[global_index]

        result = {
            **aligned_result,
            "segments": split_checked_segments,
            "text": " ".join(str(seg.get("text") or "").strip() for seg in split_checked_segments).strip(),
        }
    elif task_id is not None and run_id is not None:
        aligned_segment_ids = db.save_whisper_aligned_segments(run_id, task_id, raw_result.get("segments", []), raw_segment_ids)
        _assign_word_indexes(raw_result.get("segments", []))
        word_ids = db.save_whisper_aligned_words(run_id, task_id, raw_result.get("segments", []), aligned_segment_ids)
        pysbd_segment_ids = db.save_whisper_pysbd_segments(run_id, task_id, raw_result.get("segments", []), word_ids)
        split_ids = db.save_whisper_splits(
            run_id,
            task_id,
            raw_result.get("segments", []),
            pysbd_segment_ids,
            word_ids,
        )
        for segment_index, segment in enumerate(raw_result.get("segments", [])):
            segment["_whisper_split_id"] = split_ids.get(segment_index)

    # 将 Whisper 返回的 segments 转成项目内部统一的 utterances 格式
    utterances = _convert_segments(result.get("segments", []))
    log.debug("whisper result converted audio=%s utterances=%s", vocals_file, len(utterances))

    # 如果 Whisper 没有返回任何语音片段，说明识别结果不可用
    if not utterances:
        raise RuntimeError("Whisper did not return any segments.")

    # 使用 pydub 读取音频文件，并计算音频总时长
    # len(AudioSegment) 返回毫秒数
    duration_ms = len(AudioSegment.from_file(vocals_file))
    if run_id is not None:
        db.update_whisper_run_runtime(run_id, runtime_device=runtime_device, model_path=_model_name_or_path(), input_duration_ms=duration_ms)

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
