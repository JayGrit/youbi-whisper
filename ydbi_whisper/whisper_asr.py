from __future__ import annotations
import logging
import importlib
import inspect
import re
import unicodedata
from pathlib import Path
from . import db
from .config import DEVICE, MODEL_ROOT, NLTK_DATA, TORCH_HOME, WHISPER_DOWNLOAD_ROOT, WHISPER_ENGINE, WHISPER_MODEL, WHISPER_RUNTIME_DEVICE, WHISPERX_ALIGN, WHISPERX_ALIGN_INTERPOLATE_METHOD, WHISPERX_ALIGN_LOCAL_FILES_ONLY, WHISPERX_ALIGN_MODEL, WHISPERX_ALIGN_MODEL_DIR, WHISPERX_MODEL_PATH, WHISPERX_BATCH_SIZE, WHISPERX_CHUNK_SIZE, WHISPERX_COMPUTE_TYPE, WHISPERX_DIARIZE_HF_TOKEN, WHISPERX_DIARIZE_MAX_SPEAKERS, WHISPERX_DIARIZE_MIN_SPEAKERS, _huggingface_token, WHISPERX_REGROUP_MAX_CHARS, WHISPERX_REGROUP_MAX_DURATION_MS, WHISPERX_VAD_METHOD, WHISPERX_VAD_OFFSET, WHISPERX_VAD_ONSET, device
_MODEL = None
_ALIGN_MODELS: dict[tuple[str, str, str, str], tuple[object, dict]] = {}
_DIARIZATION_MODELS: dict[tuple[str, str], object] = {}
log = logging.getLogger(__name__)
SENTENCE_END_RE = re.compile('[.!?。！？]+[\\"\')\\]}]*$')
MINOR_BREAK_RE = re.compile('[,;:，；：]+[\\"\')\\]}]*$')

class NoSpeechDetected(RuntimeError):
    """Raised when VAD finds no speech that can be transcribed."""

def _is_empty_whisperx_batch_error(exc: IndexError) -> bool:
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == '__call__' and frame.f_code.co_filename.endswith('transformers/pipelines/base.py'):
            return str(exc) == 'list index out of range'
        traceback = traceback.tb_next
    return False
WEAK_CONJUNCTIONS = {'and', 'but', 'or', 'so', 'yet', 'because', 'although', 'though', 'while', 'whereas'}

def _word_timestamps_for(runtime_device: str) -> bool:
    return 'mps' not in runtime_device.lower()

def current_asr_config(language: str | None=None, *, load_model: bool=False) -> dict:
    model = _load_model() if load_model else None
    configured_runtime_device = _runtime_device_for_engine()
    runtime_device = str(getattr(model, 'device', configured_runtime_device)).lower() if model is not None else configured_runtime_device
    return {'engine': WHISPER_ENGINE, 'model': WHISPER_MODEL, 'model_path': _model_name_or_path(), 'operator': DEVICE, 'configured_device': WHISPER_RUNTIME_DEVICE, 'runtime_device': runtime_device, 'download_root': WHISPER_DOWNLOAD_ROOT or None, 'torch_home': str(TORCH_HOME), 'nltk_data': str(NLTK_DATA), 'transcribe_options': {'language': language, 'word_timestamps': WHISPER_ENGINE == 'openai' and _word_timestamps_for(runtime_device), 'verbose': False}, 'whisperx': {'batch_size': WHISPERX_BATCH_SIZE, 'compute_type': WHISPERX_COMPUTE_TYPE, 'vad_method': WHISPERX_VAD_METHOD, 'vad_options': _whisperx_vad_options(), 'align': WHISPERX_ALIGN, 'align_model': WHISPERX_ALIGN_MODEL or None, 'align_model_dir': WHISPERX_ALIGN_MODEL_DIR, 'align_local_files_only': WHISPERX_ALIGN_LOCAL_FILES_ONLY, 'align_interpolate_method': WHISPERX_ALIGN_INTERPOLATE_METHOD, 'regroup': {'max_chars': WHISPERX_REGROUP_MAX_CHARS, 'max_duration_ms': WHISPERX_REGROUP_MAX_DURATION_MS}, 'diarize': {'available': bool(WHISPERX_DIARIZE_HF_TOKEN), 'min_speakers': WHISPERX_DIARIZE_MIN_SPEAKERS, 'max_speakers': WHISPERX_DIARIZE_MAX_SPEAKERS}}}

def _runtime_device_for_engine() -> str:
    runtime_device = device()
    if WHISPER_ENGINE == 'whisperx' and 'mps' in runtime_device.lower():
        return 'cpu'
    return runtime_device

def _whisperx_vad_options() -> dict:
    return {'chunk_size': WHISPERX_CHUNK_SIZE, 'vad_onset': WHISPERX_VAD_ONSET, 'vad_offset': WHISPERX_VAD_OFFSET}

def _model_name_or_path() -> str:
    if WHISPER_ENGINE != 'whisperx':
        return WHISPER_MODEL
    configured_path = Path(WHISPERX_MODEL_PATH).expanduser() if WHISPERX_MODEL_PATH else None
    if configured_path is not None:
        return str(configured_path)
    local_model = MODEL_ROOT / 'faster-whisper-large-v3-turbo'
    if local_model.exists():
        return str(local_model)
    return WHISPER_MODEL

def _load_model():
    global _MODEL
    if _MODEL is None:
        runtime_device = _runtime_device_for_engine()
        log.debug('whisper loading model engine=%s model=%s model_path=%s configured_device=%s runtime_device=%s download_root=%s torch_home=%s', WHISPER_ENGINE, WHISPER_MODEL, _model_name_or_path(), WHISPER_RUNTIME_DEVICE, runtime_device, WHISPER_DOWNLOAD_ROOT or None, TORCH_HOME)
        if WHISPER_ENGINE == 'whisperx':
            import whisperx
            model_name_or_path = _model_name_or_path()
            if not Path(model_name_or_path).expanduser().exists():
                log.info('正在下载语音识别模型 %s（首次使用）', WHISPER_MODEL)
            _MODEL = whisperx.load_model(model_name_or_path, runtime_device, compute_type=WHISPERX_COMPUTE_TYPE, language=None, vad_method=WHISPERX_VAD_METHOD, vad_options=_whisperx_vad_options(), download_root=WHISPER_DOWNLOAD_ROOT or None)
            log.debug('whisper model loaded engine=%s model=%s model_path=%s runtime_device=%s', WHISPER_ENGINE, WHISPER_MODEL, _model_name_or_path(), runtime_device)
            return _MODEL
        if WHISPER_ENGINE != 'openai':
            raise ValueError(f'Unsupported whisper engine: {WHISPER_ENGINE}')
        import whisper
        model_file = Path(WHISPER_DOWNLOAD_ROOT).expanduser() / f'{WHISPER_MODEL}.pt' if WHISPER_DOWNLOAD_ROOT else None
        if model_file is not None and (not model_file.is_file()):
            log.info('正在下载语音识别模型 %s（首次使用）', WHISPER_MODEL)
        _MODEL = whisper.load_model(WHISPER_MODEL, device=runtime_device, download_root=WHISPER_DOWNLOAD_ROOT or None)
        log.debug('whisper model loaded engine=%s model=%s runtime_device=%s', WHISPER_ENGINE, WHISPER_MODEL, runtime_device)
    return _MODEL
_WTPSPLIT_MODEL = None
_WTPSPLIT_UNAVAILABLE = False
from .asr_orchestration import align_known_text, recognize_speech

from .semantic_segmentation import (
    _assign_word_indexes,
    _convert_segments,
    _convert_words,
    _is_punctuation_only,
    _join_word_texts,
    _long_split_reason,
    _minor_punctuation_split_at,
    _narration_segments_without_punctuation_timing,
    _normalized_word_text,
    _regroup_aligned_segments,
    _regroup_words_by_punctuation,
    _segment_from_words,
    _segment_too_long,
    _segment_words_with_pysbd,
    _semantic_parts,
    _semantic_split_at,
    _sentence_segmenter_language,
    _speaker_from_words,
    _speaker_label,
    _split_long_segment_on_minor_punctuation,
    _split_punctuation_value,
    _to_ms,
    _weak_conjunction_split_at,
    _word_separator,
    _word_text,
    _word_text_length,
    _wtpsplit_model,
)
from .whisperx_alignment import (
    _align_whisperx_result,
    _assign_segment_speakers_from_diarization,
    _diarization_rows,
    _diarize_raw_segments,
    _diarize_whisperx_result,
    _load_align_model,
    _load_diarization_pipeline,
    _local_align_model_name,
    _overlap_seconds,
    _patch_hf_hub_download_auth_alias,
    _patch_torch_load_for_pyannote_checkpoint,
)
