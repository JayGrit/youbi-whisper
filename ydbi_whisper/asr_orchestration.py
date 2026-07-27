from __future__ import annotations

from . import whisper_asr as _core

def _transcribe(model: object, vocals_file: _core.Path, language: str, runtime_device: str, word_timestamps: bool) -> tuple[dict, object | None]:
    if _core.WHISPER_ENGINE == 'whisperx':
        import whisperx
        _core.log.debug('whisperx loading audio file=%s', vocals_file)
        audio = whisperx.load_audio(str(vocals_file))
        _core.log.debug('whisperx transcribe start file=%s language=%s batch_size=%s chunk_size=%s runtime_device=%s', vocals_file, language, _core.WHISPERX_BATCH_SIZE, _core.WHISPERX_CHUNK_SIZE, runtime_device)
        try:
            result = model.transcribe(audio, batch_size=_core.WHISPERX_BATCH_SIZE, language=language, chunk_size=_core.WHISPERX_CHUNK_SIZE, verbose=False)
        except IndexError as exc:
            if not _core._is_empty_whisperx_batch_error(exc):
                raise
            raise _core.NoSpeechDetected('无人声') from exc
        segments = result.get('segments', [])
        result['text'] = ' '.join((str(seg.get('text') or '').strip() for seg in segments)).strip()
        _core.log.debug('whisperx transcribe done file=%s language=%s segments=%s text_chars=%s', vocals_file, result.get('language') or language, len(segments), len(result['text']))
        return (result, audio)
    _core.log.debug('openai whisper transcribe start file=%s language=%s runtime_device=%s word_timestamps=%s', vocals_file, language, runtime_device, word_timestamps)
    return (model.transcribe(str(vocals_file), language=language, word_timestamps=word_timestamps, verbose=False), None)

def align_known_text(vocals_file: _core.Path, known_segments: list[dict], language: str, *, task_id: str | None=None, run_id: int | None=None) -> dict:
    from pydub import AudioSegment
    import whisperx
    if _core.WHISPER_ENGINE != 'whisperx':
        raise RuntimeError('Known-text alignment requires the whisperx engine.')
    if not _core.WHISPERX_ALIGN:
        raise RuntimeError('Known-text alignment requires WhisperX alignment to be enabled.')
    duration_ms = len(AudioSegment.from_file(vocals_file))
    transcript = []
    previous_end = 0
    for expected_index, segment in enumerate(known_segments):
        item_index = int(segment.get('item_index'))
        status = str(segment.get('status') or '').strip().lower()
        text = str(segment.get('text') or '').strip()
        start_time = int(segment.get('start_time') or 0)
        end_time = int(segment.get('end_time') or 0)
        if item_index != expected_index:
            raise ValueError(f'Narration segment indexes are not contiguous at {item_index}.')
        if status and status != 'success':
            raise ValueError(f'Narration segment {item_index} is not successful: {status}.')
        if not text:
            raise ValueError(f'Narration segment {item_index} has empty text.')
        if start_time < previous_end or end_time <= start_time or end_time > duration_ms + 20:
            raise ValueError(f'Narration segment {item_index} has invalid timing {start_time}-{end_time} for audio duration {duration_ms}.')
        end_time = min(end_time, duration_ms)
        transcript.append({'text': text, 'start': start_time / 1000.0, 'end': end_time / 1000.0})
        previous_end = end_time
    runtime_device = _core._runtime_device_for_engine()
    audio = whisperx.load_audio(str(vocals_file))
    raw_result = {'segments': transcript, 'text': _core._word_separator(language).join((segment['text'] for segment in transcript)), 'language': language}
    if run_id is not None:
        _core.db.update_whisper_run_runtime(run_id, runtime_device=runtime_device, model_path=_core.WHISPERX_ALIGN_MODEL or 'default-align-model', input_duration_ms=duration_ms)
    raw_segment_ids: dict[int, int] = {}
    if task_id is not None and run_id is not None:
        raw_segment_ids = _core.db.save_whisper_raw_segments(run_id, task_id, raw_result['segments'])
    aligned_result = _core._align_whisperx_result(whisperx, raw_result, audio, language, runtime_device, task_id=task_id)
    aligned_segments, narration_segments = _core._narration_segments_without_punctuation_timing(aligned_result.get('segments') or [], language)
    _core._assign_word_indexes(aligned_segments)
    word_ids: dict[int, int] = {}
    aligned_segment_ids: dict[int, int] = {}
    if task_id is not None and run_id is not None:
        aligned_segment_ids = _core.db.save_whisper_aligned_segments(run_id, task_id, aligned_segments, raw_segment_ids)
        for segment_index, segment in enumerate(aligned_segments):
            segment['_whisper_aligned_segment_id'] = aligned_segment_ids.get(segment_index)
            for word in segment.get('words') or []:
                word['_whisper_aligned_segment_id'] = aligned_segment_ids.get(segment_index)
        word_ids = _core.db.save_whisper_aligned_words(run_id, task_id, aligned_segments, aligned_segment_ids)
        for segment in aligned_segments:
            for word in segment.get('words') or []:
                global_index = word.get('_whisper_word_global_index')
                if global_index in word_ids:
                    word['_whisper_aligned_word_id'] = word_ids[global_index]
    pysbd_segments = narration_segments
    final_segments = narration_segments
    if task_id is not None and run_id is not None:
        for segment in pysbd_segments:
            aligned_index = segment.get('_whisper_aligned_segment_index')
            if aligned_index is not None:
                segment['_whisper_aligned_segment_id'] = aligned_segment_ids.get(int(aligned_index))
        pysbd_segment_ids = _core.db.save_whisper_pysbd_segments(run_id, task_id, pysbd_segments, word_ids)
        split_ids = _core.db.save_whisper_splits(run_id, task_id, final_segments, pysbd_segment_ids, word_ids)
        for segment_index, segment in enumerate(final_segments):
            segment['_whisper_split_id'] = split_ids.get(segment_index)
            for word in segment.get('words') or []:
                global_index = word.get('_whisper_word_global_index')
                if global_index in word_ids:
                    word['_whisper_aligned_word_id'] = word_ids[global_index]
    utterances = _core._convert_segments(final_segments)
    if not utterances:
        raise RuntimeError('Known-text alignment did not return any segments.')
    return {'audio_info': {'duration': duration_ms}, 'result': {'text': _core._word_separator(language).join((str(segment.get('text') or '').strip() for segment in final_segments)), 'utterances': utterances}}

def recognize_speech(vocals_file: _core.Path, session: _core.Path, language: str, *, task_id: str | None=None, run_id: int | None=None, diarize: bool=False) -> dict:
    from pydub import AudioSegment
    metadata_dir = session / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    model = _core._load_model()
    runtime_device = str(getattr(model, 'device', _core._runtime_device_for_engine())).lower()
    if run_id is not None:
        _core.db.update_whisper_run_runtime(run_id, runtime_device=runtime_device, model_path=_core._model_name_or_path())
    word_timestamps = _core.WHISPER_ENGINE == 'openai' and _core._word_timestamps_for(runtime_device)
    if _core.WHISPER_ENGINE == 'openai' and (not word_timestamps):
        _core.log.warning('当前设备不支持词级时间戳，已自动关闭')
    _core.log.debug('whisper runtime audio=%s engine=%s model=%s configured_device=%s runtime_device=%s language=%s word_timestamps=%s vad_method=%s vad_options=%s align=%s diarize=%s', vocals_file, _core.WHISPER_ENGINE, _core.WHISPER_MODEL, _core.WHISPER_RUNTIME_DEVICE, runtime_device, language, word_timestamps, _core.WHISPERX_VAD_METHOD if _core.WHISPER_ENGINE == 'whisperx' else None, _core._whisperx_vad_options() if _core.WHISPER_ENGINE == 'whisperx' else None, _core.WHISPERX_ALIGN if _core.WHISPER_ENGINE == 'whisperx' else None, diarize)
    if diarize and _core.WHISPER_ENGINE != 'whisperx':
        raise RuntimeError('Speaker diarization requires the whisperx engine.')
    raw_result, audio = _transcribe(model, vocals_file, language, runtime_device, word_timestamps)
    result = raw_result
    raw_segment_ids: dict[int, int] = {}
    if task_id is not None and run_id is not None:
        raw_segment_ids = _core.db.save_whisper_raw_segments(run_id, task_id, raw_result.get('segments', []))
    if _core.WHISPER_ENGINE == 'whisperx' and audio is not None:
        import whisperx
        aligned_result = _core._align_whisperx_result(whisperx, raw_result, audio, language, runtime_device, task_id=task_id)
        if diarize:
            aligned_result = _core._diarize_whisperx_result(whisperx, aligned_result, audio, runtime_device, task_id=task_id)
        aligned_segments = aligned_result.get('segments') or []
        _core._assign_word_indexes(aligned_segments)
        if task_id is not None and run_id is not None:
            aligned_segment_ids = _core.db.save_whisper_aligned_segments(run_id, task_id, aligned_segments, raw_segment_ids)
            for segment_index, segment in enumerate(aligned_segments):
                segment['_whisper_aligned_segment_id'] = aligned_segment_ids.get(segment_index)
                for word in segment.get('words') or []:
                    word['_whisper_aligned_segment_id'] = aligned_segment_ids.get(segment_index)
            word_ids = _core.db.save_whisper_aligned_words(run_id, task_id, aligned_segments, aligned_segment_ids)
            for segment in aligned_segments:
                for word in segment.get('words') or []:
                    global_index = word.get('_whisper_word_global_index')
                    if global_index in word_ids:
                        word['_whisper_aligned_word_id'] = word_ids[global_index]
        else:
            word_ids = {}
        align_language = str(aligned_result.get('language') or language or 'en').lower()
        pysbd_segments, split_checked_segments = _core._regroup_aligned_segments(aligned_segments, align_language)
        if task_id is not None and run_id is not None:
            for segment in pysbd_segments:
                aligned_index = segment.get('_whisper_aligned_segment_index')
                if aligned_index is not None:
                    segment['_whisper_aligned_segment_id'] = aligned_segment_ids.get(int(aligned_index))
            pysbd_segment_ids = _core.db.save_whisper_pysbd_segments(run_id, task_id, pysbd_segments, word_ids)
            split_ids = _core.db.save_whisper_splits(run_id, task_id, split_checked_segments, pysbd_segment_ids, word_ids)
            for segment_index, segment in enumerate(split_checked_segments):
                segment['_whisper_split_id'] = split_ids.get(segment_index)
                for word in segment.get('words') or []:
                    global_index = word.get('_whisper_word_global_index')
                    if global_index in word_ids:
                        word['_whisper_aligned_word_id'] = word_ids[global_index]
        result = {**aligned_result, 'segments': split_checked_segments, 'text': ' '.join((str(seg.get('text') or '').strip() for seg in split_checked_segments)).strip()}
    elif task_id is not None and run_id is not None:
        aligned_segment_ids = _core.db.save_whisper_aligned_segments(run_id, task_id, raw_result.get('segments', []), raw_segment_ids)
        _core._assign_word_indexes(raw_result.get('segments', []))
        word_ids = _core.db.save_whisper_aligned_words(run_id, task_id, raw_result.get('segments', []), aligned_segment_ids)
        pysbd_segment_ids = _core.db.save_whisper_pysbd_segments(run_id, task_id, raw_result.get('segments', []), word_ids)
        split_ids = _core.db.save_whisper_splits(run_id, task_id, raw_result.get('segments', []), pysbd_segment_ids, word_ids)
        for segment_index, segment in enumerate(raw_result.get('segments', [])):
            segment['_whisper_split_id'] = split_ids.get(segment_index)
    utterances = _core._convert_segments(result.get('segments', []))
    _core.log.debug('whisper result converted audio=%s utterances=%s', vocals_file, len(utterances))
    if not utterances:
        raise _core.NoSpeechDetected('无人声')
    duration_ms = len(AudioSegment.from_file(vocals_file))
    if run_id is not None:
        _core.db.update_whisper_run_runtime(run_id, runtime_device=runtime_device, model_path=_core._model_name_or_path(), input_duration_ms=duration_ms)
    payload = {'audio_info': {'duration': duration_ms}, 'result': {'text': (result.get('text') or '').strip(), 'utterances': utterances}}
    return payload
