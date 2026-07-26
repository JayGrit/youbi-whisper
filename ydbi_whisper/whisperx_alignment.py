from __future__ import annotations

from . import whisper_asr as _core

def _patch_hf_hub_download_auth_alias() -> None:
    try:
        import huggingface_hub
    except Exception:
        return
    original = getattr(huggingface_hub, 'hf_hub_download', None)
    if original is None or 'use_auth_token' in _core.inspect.signature(original).parameters:
        return
    if getattr(original, '_ydbi_accepts_use_auth_token', False):
        return

    def compatible_hf_hub_download(*args, **kwargs):
        use_auth_token = kwargs.pop('use_auth_token', None)
        if use_auth_token is not None and 'token' not in kwargs:
            kwargs['token'] = use_auth_token
        return original(*args, **kwargs)
    compatible_hf_hub_download._ydbi_accepts_use_auth_token = True
    huggingface_hub.hf_hub_download = compatible_hf_hub_download
    for module_name in ('pyannote.audio.core.pipeline', 'pyannote.audio.core.model', 'pyannote.audio.pipelines.utils.getter'):
        try:
            module = _core.importlib.import_module(module_name)
            module.hf_hub_download = compatible_hf_hub_download
        except Exception:
            pass

def _patch_torch_load_for_pyannote_checkpoint() -> None:
    try:
        import torch
    except Exception:
        return
    original = torch.load
    if getattr(original, '_ydbi_pyannote_weights_only_default', False):
        return

    def compatible_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original(*args, **kwargs)
    compatible_torch_load._ydbi_pyannote_weights_only_default = True
    torch.load = compatible_torch_load

def _local_align_model_name(whisperx, language: str) -> str | None:
    configured_model = _core.WHISPERX_ALIGN_MODEL or None
    if configured_model and _core.Path(configured_model).expanduser().exists():
        return str(_core.Path(configured_model).expanduser())
    alignment_module = getattr(whisperx, 'alignment', None)
    if alignment_module is None:
        alignment_module = _core.importlib.import_module('whisperx.alignment')
    hf_models = getattr(alignment_module, 'DEFAULT_ALIGN_MODELS_HF', {})
    model_name = configured_model or hf_models.get(language)
    if not model_name or '/' not in model_name:
        return model_name
    cache_root = _core.Path(_core.WHISPERX_ALIGN_MODEL_DIR).expanduser()
    repo_cache = cache_root / f"models--{model_name.replace('/', '--')}"
    main_ref = repo_cache / 'refs' / 'main'
    if not main_ref.is_file():
        raise RuntimeError(f'WhisperX align model is not available in the local cache: model={model_name}, dir={cache_root}')
    revision = main_ref.read_text(encoding='utf-8').strip()
    snapshot = repo_cache / 'snapshots' / revision
    if not snapshot.is_dir():
        raise RuntimeError(f'WhisperX align model snapshot is missing from the local cache: model={model_name}, revision={revision}, dir={cache_root}')
    return str(snapshot)

def _load_align_model(whisperx, language: str, runtime_device: str):
    kwargs = {'model_name': _core.WHISPERX_ALIGN_MODEL or None, 'model_dir': _core.WHISPERX_ALIGN_MODEL_DIR}
    try:
        return whisperx.load_align_model(language, runtime_device, model_cache_only=_core.WHISPERX_ALIGN_LOCAL_FILES_ONLY, **kwargs)
    except TypeError as exc:
        if 'model_cache_only' not in str(exc):
            raise
        if _core.WHISPERX_ALIGN_LOCAL_FILES_ONLY:
            kwargs['model_name'] = _core._local_align_model_name(whisperx, language)
        return whisperx.load_align_model(language, runtime_device, **kwargs)

def _align_whisperx_result(whisperx, result: dict, audio: object, language: str, runtime_device: str, task_id: str | None=None) -> dict:
    if not _core.WHISPERX_ALIGN or not result.get('segments'):
        return result
    align_language = str(result.get('language') or language or 'en').lower()
    task_label = task_id or '本地任务'
    _core.log.info('任务 %s：正在加载时间轴对齐模型', task_label)
    _core.log.debug('whisperx align model loading language=%s model=%s model_dir=%s runtime_device=%s', align_language, _core.WHISPERX_ALIGN_MODEL or None, _core.WHISPERX_ALIGN_MODEL_DIR, runtime_device)
    _core.Path(_core.WHISPERX_ALIGN_MODEL_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    align_key = (align_language, runtime_device, _core.WHISPERX_ALIGN_MODEL or '', str(_core.Path(_core.WHISPERX_ALIGN_MODEL_DIR).expanduser()))
    cached_align = _core._ALIGN_MODELS.get(align_key)
    if cached_align is None:
        try:
            cached_align = _core._load_align_model(whisperx, align_language, runtime_device)
        except (OSError, ValueError) as exc:
            if _core.WHISPERX_ALIGN_LOCAL_FILES_ONLY:
                raise RuntimeError(f'WhisperX align model is not available in the local cache: language={align_language}, dir={_core.WHISPERX_ALIGN_MODEL_DIR}') from exc
            raise
        _core._ALIGN_MODELS[align_key] = cached_align
    align_model, align_metadata = cached_align
    _core.log.info('任务 %s：正在进行时间轴对齐，原始片段 %d 段', task_label, len(result.get('segments') or []))
    _core.log.debug('whisperx align start language=%s segments=%s interpolate_method=%s', align_language, len(result.get('segments') or []), _core.WHISPERX_ALIGN_INTERPOLATE_METHOD)
    aligned = whisperx.align(result['segments'], align_model, align_metadata, audio, runtime_device, interpolate_method=_core.WHISPERX_ALIGN_INTERPOLATE_METHOD, return_char_alignments=False, print_progress=False)
    aligned_segments = aligned.get('segments') or []
    aligned['text'] = ' '.join((str(seg.get('text') or '').strip() for seg in aligned_segments)).strip()
    aligned['language'] = align_language
    _core.log.info('任务 %s：时间轴对齐完成，共 %d 段、%d 个词', task_label, len(aligned_segments), len(aligned.get('word_segments') or []))
    _core.log.debug('whisperx align done language=%s aligned_segments=%s word_segments=%s', align_language, len(aligned_segments), len(aligned.get('word_segments') or []))
    return aligned

def _diarize_whisperx_result(whisperx, result: dict, audio: object, runtime_device: str, *, task_id: str | None=None) -> dict:
    hf_token = _core._huggingface_token() or _core.WHISPERX_DIARIZE_HF_TOKEN
    if not hf_token:
        raise RuntimeError('WhisperX speaker diarization requires YDBI_WHISPERX_DIARIZE_HF_TOKEN or HF_TOKEN.')
    task_label = task_id or '本地任务'
    _core._patch_hf_hub_download_auth_alias()
    _core._patch_torch_load_for_pyannote_checkpoint()
    from whisperx.diarize import DiarizationPipeline
    model_key = (runtime_device, hf_token)
    diarize_model = _core._DIARIZATION_MODELS.get(model_key)
    if diarize_model is None:
        _core.log.info('任务 %s：正在加载说话人分离模型', task_label)
        init_params = _core.inspect.signature(DiarizationPipeline.__init__).parameters
        token_kwargs = {'use_auth_token': hf_token} if 'use_auth_token' in init_params else {'token': hf_token}
        diarize_model = DiarizationPipeline(device=runtime_device, **token_kwargs)
        _core._DIARIZATION_MODELS[model_key] = diarize_model
    kwargs = {}
    if _core.WHISPERX_DIARIZE_MIN_SPEAKERS is not None:
        kwargs['min_speakers'] = _core.WHISPERX_DIARIZE_MIN_SPEAKERS
    if _core.WHISPERX_DIARIZE_MAX_SPEAKERS is not None:
        kwargs['max_speakers'] = _core.WHISPERX_DIARIZE_MAX_SPEAKERS
    _core.log.info('任务 %s：正在区分说话人', task_label)
    diarize_segments = diarize_model(audio, **kwargs)
    assigned = whisperx.assign_word_speakers(diarize_segments, result)
    speaker_count = len({speaker for segment in assigned.get('segments') or [] for speaker in [str(segment.get('speaker') or '').strip()] if speaker})
    _core.log.info('任务 %s：说话人区分完成，共 %d 个说话人', task_label, speaker_count)
    return assigned

def _load_diarization_pipeline(runtime_device: str, task_label: str):
    hf_token = _core._huggingface_token() or _core.WHISPERX_DIARIZE_HF_TOKEN
    if not hf_token:
        raise RuntimeError('WhisperX speaker diarization requires YDBI_WHISPERX_DIARIZE_HF_TOKEN or HF_TOKEN.')
    _core._patch_hf_hub_download_auth_alias()
    _core._patch_torch_load_for_pyannote_checkpoint()
    from whisperx.diarize import DiarizationPipeline
    model_key = (runtime_device, hf_token)
    diarize_model = _core._DIARIZATION_MODELS.get(model_key)
    if diarize_model is not None:
        return diarize_model
    _core.log.info('任务 %s：正在加载说话人分离模型', task_label)
    init_params = _core.inspect.signature(DiarizationPipeline.__init__).parameters
    token_kwargs = {'use_auth_token': hf_token} if 'use_auth_token' in init_params else {'token': hf_token}
    diarize_model = DiarizationPipeline(device=runtime_device, **token_kwargs)
    _core._DIARIZATION_MODELS[model_key] = diarize_model
    return diarize_model

def _diarization_rows(diarize_segments: object) -> list[dict]:
    if hasattr(diarize_segments, 'to_dict'):
        return list(diarize_segments.to_dict('records'))
    return []

def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))

def _assign_segment_speakers_from_diarization(segments: list[dict], diarize_segments: object) -> None:
    rows = _core._diarization_rows(diarize_segments)
    if not rows:
        return
    for segment in segments:
        start = float(segment.get('start') or 0.0)
        end = float(segment.get('end') or start)
        scores: dict[str, float] = {}
        for row in rows:
            speaker = str(row.get('speaker') or '').strip()
            if not speaker:
                continue
            row_start = float(row.get('start') or 0.0)
            row_end = float(row.get('end') or row_start)
            overlap = _core._overlap_seconds(start, end, row_start, row_end)
            if overlap > 0:
                scores[speaker] = scores.get(speaker, 0.0) + overlap
        if scores:
            segment['speaker'] = max(scores, key=scores.get)

def _diarize_raw_segments(result: dict, audio: object, runtime_device: str, *, task_id: str | None=None) -> dict:
    task_label = task_id or '本地任务'
    diarize_model = _core._load_diarization_pipeline(runtime_device, task_label)
    kwargs = {}
    if _core.WHISPERX_DIARIZE_MIN_SPEAKERS is not None:
        kwargs['min_speakers'] = _core.WHISPERX_DIARIZE_MIN_SPEAKERS
    if _core.WHISPERX_DIARIZE_MAX_SPEAKERS is not None:
        kwargs['max_speakers'] = _core.WHISPERX_DIARIZE_MAX_SPEAKERS
    _core.log.info('任务 %s：正在区分说话人', task_label)
    diarize_segments = diarize_model(audio, **kwargs)
    segments = result.get('segments') or []
    _core._assign_segment_speakers_from_diarization(segments, diarize_segments)
    speaker_count = len({str(segment.get('speaker') or '').strip() for segment in segments if str(segment.get('speaker') or '').strip()})
    _core.log.info('任务 %s：说话人区分完成，共 %d 个说话人', task_label, speaker_count)
    return {**result, 'segments': segments, 'text': ' '.join((str(seg.get('text') or '').strip() for seg in segments)).strip()}
