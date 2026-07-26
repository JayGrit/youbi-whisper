from __future__ import annotations

from . import whisper_asr as _core

def _to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))

def _convert_words(words: list) -> list:
    converted = []
    for w in words or []:
        item = {'text': w.get('word') or w.get('text') or '', 'start_time': int(w.get('start_time')) if w.get('start_time') is not None else _core._to_ms(w.get('start', 0.0)), 'end_time': int(w.get('end_time')) if w.get('end_time') is not None else _core._to_ms(w.get('end', 0.0))}
        for key in ('_whisper_word_global_index', '_whisper_aligned_segment_index', '_whisper_aligned_segment_id', '_whisper_aligned_word_id'):
            if key in w:
                item[key] = w[key]
        converted.append(item)
    return converted

def _convert_segments(segments: list) -> list:
    converted = []
    for seg in segments:
        item = {'text': seg.get('text', '').strip(), 'start_time': _core._to_ms(seg.get('start', 0.0)), 'end_time': _core._to_ms(seg.get('end', 0.0)), 'words': _core._convert_words(seg.get('words', []))}
        for key in ('speaker', '_whisper_aligned_segment_index', '_whisper_aligned_segment_id', '_whisper_pysbd_index', '_whisper_split_id'):
            if key in seg:
                item[key] = seg[key]
        converted.append(item)
    return converted

def _word_text(word: dict) -> str:
    return str(word.get('word') or word.get('text') or '').strip()

def _is_punctuation_only(text: str) -> bool:
    value = str(text or '').strip()
    return bool(value) and all((_core.unicodedata.category(char).startswith('P') for char in value))

def _narration_segments_without_punctuation_timing(segments: list[dict], language: str) -> tuple[list[dict], list[dict]]:
    """Remove punctuation timestamps and expose their duration as subtitle gaps."""
    cleaned_segments = []
    caption_segments = []
    for aligned_index, segment in enumerate(segments):
        source_words = segment.get('words') or []
        cleaned_words = [word for word in source_words if not _core._is_punctuation_only(_core._word_text(word))]
        cleaned_segments.append({**segment, 'words': cleaned_words})
        current_words: list[dict] = []
        trailing_punctuation = ''

        def flush() -> None:
            nonlocal current_words, trailing_punctuation
            if not current_words:
                trailing_punctuation = ''
                return
            caption = _core._segment_from_words(current_words, language)
            if caption is not None:
                caption['text'] = _core._join_word_texts(current_words, language)
                caption['_whisper_aligned_segment_index'] = aligned_index
                caption['_whisper_segmentation_method'] = 'narration_punctuation_gap'
                caption_segments.append(caption)
            current_words = []
            trailing_punctuation = ''
        for word in source_words:
            text = _core._word_text(word)
            if not text:
                continue
            if _core._is_punctuation_only(text):
                if current_words:
                    trailing_punctuation += text
                continue
            if trailing_punctuation:
                flush()
            if 'start' in word and 'end' in word:
                current_words.append(word)
        flush()
    return (cleaned_segments, caption_segments)

def _word_separator(language: str | None) -> str:
    return '' if _core._sentence_segmenter_language(language or 'en') in {'zh', 'ja'} else ' '

def _join_word_texts(words: list[dict], language: str | None=None) -> str:
    return _core._word_separator(language).join((_core._word_text(word) for word in words if _core._word_text(word))).strip()

def _assign_word_indexes(segments: list[dict]) -> None:
    global_word_index = 0
    for segment_index, segment in enumerate(segments):
        for word in segment.get('words') or []:
            if not _core._word_text(word):
                continue
            word['_whisper_word_global_index'] = global_word_index
            word['_whisper_aligned_segment_index'] = segment_index
            global_word_index += 1

def _speaker_label(item: dict | None) -> str | None:
    if not item:
        return None
    speaker = str(item.get('speaker') or '').strip()
    return speaker or None

def _speaker_from_words(words: list[dict]) -> str | None:
    counts: dict[str, int] = {}
    first_label: str | None = None
    for word in words:
        speaker = _core._speaker_label(word)
        if not speaker:
            continue
        first_label = first_label or speaker
        counts[speaker] = counts.get(speaker, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda label: (counts[label], label == first_label))

def _segment_from_words(words: list[dict], language: str | None=None) -> dict | None:
    timed_words = [word for word in words if 'start' in word and 'end' in word]
    text = _core._join_word_texts(words, language)
    if not text or not timed_words:
        return None
    first_word = words[0] if words else {}
    segment = {'text': text, 'start': float(timed_words[0]['start']), 'end': float(timed_words[-1]['end']), 'words': words, '_whisper_aligned_segment_index': first_word.get('_whisper_aligned_segment_index')}
    speaker = _core._speaker_from_words(words)
    if speaker:
        segment['speaker'] = speaker
    return segment

def _segment_too_long(segment: dict) -> bool:
    text = str(segment.get('text') or '')
    duration_ms = int(round((float(segment.get('end') or 0.0) - float(segment.get('start') or 0.0)) * 1000))
    return len(text) >= _core.WHISPERX_REGROUP_MAX_CHARS or duration_ms >= _core.WHISPERX_REGROUP_MAX_DURATION_MS

def _long_split_reason(segment: dict) -> str:
    text_too_long = len(str(segment.get('text') or '')) >= _core.WHISPERX_REGROUP_MAX_CHARS
    duration_ms = int(round((float(segment.get('end') or 0.0) - float(segment.get('start') or 0.0)) * 1000))
    duration_too_long = duration_ms >= _core.WHISPERX_REGROUP_MAX_DURATION_MS
    if text_too_long and duration_too_long:
        return 'text_and_duration_too_long'
    if text_too_long:
        return 'text_too_long'
    if duration_too_long:
        return 'duration_too_long'
    return 'none'

def _word_text_length(words: list[dict], language: str | None=None) -> int:
    return len(_core._join_word_texts(words, language))

def _split_punctuation_value(text: str) -> str | None:
    match = _core.re.search('[,;:，；：]+', text)
    return match.group(0) if match else None

def _normalized_word_text(text: str) -> str:
    return _core.re.sub('^[^\\w]+|[^\\w]+$', '', text.lower())

def _minor_punctuation_split_at(words: list[dict]) -> int | None:
    candidates = [idx + 1 for idx, word in enumerate(words[:-1]) if _core.MINOR_BREAK_RE.search(_core._word_text(word))]
    if not candidates:
        return None
    total_length = _core._word_text_length(words)
    midpoint = total_length / 2
    return min(candidates, key=lambda split_at: (abs(_core._word_text_length(words[:split_at]) - midpoint), abs(_core._word_text_length(words[split_at:]) - midpoint)))

def _weak_conjunction_split_at(words: list[dict]) -> int | None:
    candidates = [idx for idx, word in enumerate(words[1:-1], start=1) if _core._normalized_word_text(_core._word_text(word)) in _core.WEAK_CONJUNCTIONS]
    if not candidates:
        return None
    total_length = _core._word_text_length(words)
    midpoint = total_length / 2
    return min(candidates, key=lambda split_at: (abs(_core._word_text_length(words[:split_at]) - midpoint), abs(_core._word_text_length(words[split_at:]) - midpoint)))

def _wtpsplit_model():
    if _core._WTPSPLIT_MODEL is not None:
        return _core._WTPSPLIT_MODEL
    if _core._WTPSPLIT_UNAVAILABLE:
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
            _core._WTPSPLIT_UNAVAILABLE = True
            _core.log.warning('语义分句组件不可用，长句将保持原样')
            _core.log.debug('wtpsplit import failed', exc_info=fallback_exc)
            return None
        try:
            _core._WTPSPLIT_MODEL = WtP('wtp-bert-mini', ignore_legacy_warning=True)
        except Exception as fallback_exc:
            _core._WTPSPLIT_UNAVAILABLE = True
            _core.log.warning('语义分句模型加载失败，长句将保持原样')
            _core.log.debug('wtpsplit WtP model load failed', exc_info=fallback_exc)
            return None
        return _core._WTPSPLIT_MODEL
    try:
        model_name: str | _core.Path = 'sat-3l-sm'
        tokenizer_name: str | _core.Path = 'facebookAI/xlm-roberta-base'
        from_pretrained_kwargs = None
        try:
            from huggingface_hub import snapshot_download
            model_snapshot = _core.Path(snapshot_download('segment-any-text/sat-3l-sm', local_files_only=True))
            tokenizer_snapshot = _core.Path(snapshot_download('facebookAI/xlm-roberta-base', local_files_only=True))
            model_is_complete = (model_snapshot / 'config.json').is_file() and any(((model_snapshot / filename).is_file() for filename in ('model.safetensors', 'pytorch_model.bin')))
            tokenizer_is_complete = (tokenizer_snapshot / 'tokenizer_config.json').is_file() and any(((tokenizer_snapshot / filename).is_file() for filename in ('tokenizer.json', 'sentencepiece.bpe.model')))
            if not model_is_complete or not tokenizer_is_complete:
                raise FileNotFoundError('incomplete wtpsplit cache')
            model_name = model_snapshot
            tokenizer_name = tokenizer_snapshot
            from_pretrained_kwargs = {'local_files_only': True}
        except Exception:
            _core.log.info('正在下载语义分句模型 sat-3l-sm（首次使用）')
        _core._WTPSPLIT_MODEL = SaT(model_name, tokenizer_name_or_path=tokenizer_name, from_pretrained_kwargs=from_pretrained_kwargs)
    except Exception as exc:
        _core._WTPSPLIT_UNAVAILABLE = True
        _core.log.warning('语义分句模型加载失败，长句将保持原样')
        _core.log.debug('wtpsplit SaT model load failed', exc_info=exc)
        return None
    return _core._WTPSPLIT_MODEL

def _semantic_parts(model: object, full_text: str, lang: str) -> list[str]:
    try:
        return [part.strip() for part in model.split(full_text, max_length=_core.WHISPERX_REGROUP_MAX_CHARS, prior_type='gaussian', prior_kwargs={'lang_code': lang}) if part.strip()]
    except TypeError:
        try:
            return [part.strip() for part in model.split(full_text, lang_code=lang, max_length=_core.WHISPERX_REGROUP_MAX_CHARS) if part.strip()]
        except TypeError:
            return [part.strip() for part in model.split(full_text, lang_code=lang) if part.strip()]

def _semantic_split_at(words: list[dict], language: str | None=None) -> int | None:
    if len(words) <= 1:
        return None
    model = _core._wtpsplit_model()
    if model is None:
        return None
    text_parts = []
    word_spans = []
    cursor = 0
    for word in words:
        text = _core._word_text(word)
        if not text:
            continue
        separator = _core._word_separator(language)
        if text_parts and separator:
            text_parts.append(separator)
            cursor += len(separator)
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        word_spans.append((start, cursor))
    full_text = ''.join(text_parts).strip()
    if not full_text or len(word_spans) <= 1:
        return None
    lang = _core._sentence_segmenter_language(language or 'en')
    try:
        semantic_parts = _core._semantic_parts(model, full_text, lang)
    except Exception as exc:
        _core.log.warning('语义分句失败，长句将保持原样')
        _core.log.debug('wtpsplit split failed', exc_info=exc)
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

def _split_long_segment_on_minor_punctuation(segment: dict, pysbd_index: int | None=None) -> list[dict]:
    if not _core._segment_too_long(segment):
        return [{**segment, '_whisper_pysbd_index': pysbd_index, '_whisper_split_applied': False, '_whisper_split_reason': 'none'}]
    words = segment.get('words') or []
    if not words:
        return [{**segment, '_whisper_pysbd_index': pysbd_index, '_whisper_split_applied': False, '_whisper_split_reason': _core._long_split_reason(segment)}]
    split_segments = []
    current_words = []
    split_reason = _core._long_split_reason(segment)
    original_text = str(segment.get('text') or '').strip()
    language = segment.get('_whisper_pysbd_language')

    def append_group(group_words: list[dict], *, split_applied: bool, split_at_word_index: int | None=None, split_punctuation: str | None=None, split_conjunction: str | None=None, split_method: str | None=None) -> None:
        grouped = _core._segment_from_words(group_words, language)
        if grouped is not None:
            grouped['_whisper_pysbd_index'] = pysbd_index
            grouped['_whisper_split_applied'] = split_applied
            grouped['_whisper_split_reason'] = split_reason
            grouped['_whisper_split_trigger'] = split_reason
            grouped['_whisper_split_method'] = split_method or ('none' if not split_applied else 'unknown')
            grouped['_whisper_split_at_word_index'] = split_at_word_index
            grouped['_whisper_split_punctuation'] = split_punctuation
            grouped['_whisper_split_conjunction'] = split_conjunction
            grouped['_whisper_original_text'] = original_text
            split_segments.append(grouped)

    def flush_current() -> None:
        nonlocal current_words
        append_group(current_words, split_applied=bool(split_segments))
        current_words = []
    for word in words:
        text = _core._word_text(word)
        if not text:
            continue
        current_words.append(word)
        current_segment = _core._segment_from_words(current_words, language)
        if current_segment is None or not _core._segment_too_long(current_segment):
            continue
        split_at = _core._minor_punctuation_split_at(current_words)
        split_method = 'minor_punctuation' if split_at is not None else None
        split_conjunction = None
        if split_at is None:
            split_at = _core._weak_conjunction_split_at(current_words)
            split_method = 'weak_conjunction' if split_at is not None else None
            split_conjunction = _core._normalized_word_text(_core._word_text(current_words[split_at])) if split_at is not None else None
        if split_at is None:
            split_at = _core._semantic_split_at(current_words, language)
            split_method = 'semantic_wtpsplit' if split_at is not None else None
        if split_at:
            split_word = current_words[split_at - 1]
            append_group(current_words[:split_at], split_applied=True, split_at_word_index=split_at - 1, split_punctuation=_core._split_punctuation_value(_core._word_text(split_word)), split_conjunction=split_conjunction, split_method=split_method)
            current_words = current_words[split_at:]
    flush_current()
    if split_segments:
        last_method = None
        last_punctuation = None
        last_conjunction = None
        for grouped in split_segments:
            if grouped.get('_whisper_split_method') not in {None, 'unknown', 'none'}:
                last_method = grouped.get('_whisper_split_method')
            elif last_method is not None:
                grouped['_whisper_split_method'] = last_method
            if grouped.get('_whisper_split_punctuation'):
                last_punctuation = grouped.get('_whisper_split_punctuation')
            elif last_punctuation is not None and grouped.get('_whisper_split_method') == 'minor_punctuation':
                grouped['_whisper_split_punctuation'] = last_punctuation
            if grouped.get('_whisper_split_conjunction'):
                last_conjunction = grouped.get('_whisper_split_conjunction')
            elif last_conjunction is not None and grouped.get('_whisper_split_method') == 'weak_conjunction':
                grouped['_whisper_split_conjunction'] = last_conjunction
        source_part_index = 1
        source_part_count = len(split_segments)
        for grouped in split_segments:
            grouped['_whisper_original_part_index'] = source_part_index
            grouped['_whisper_original_part_count'] = source_part_count
            source_part_index += 1
    return split_segments or [{**segment, '_whisper_pysbd_index': pysbd_index, '_whisper_split_applied': False, '_whisper_split_reason': split_reason, '_whisper_split_trigger': split_reason, '_whisper_split_method': 'none', '_whisper_original_text': original_text}]

def _sentence_segmenter_language(language: str) -> str:
    normalized = str(language or 'en').strip().lower().replace('_', '-')
    if not normalized:
        return 'en'
    primary = normalized.split('-', 1)[0]
    return {'zh': 'zh', 'cn': 'zh', 'ja': 'ja', 'jp': 'ja', 'ko': 'ko'}.get(primary, primary)

def _segment_words_with_pysbd(words: list[dict], language: str) -> list[dict]:
    try:
        import pysbd
    except Exception as exc:
        _core.log.warning('句子切分组件不可用，已改用标点切分')
        _core.log.debug('pysbd import failed', exc_info=exc)
        return []
    word_spans = []
    text_parts = []
    cursor = 0
    for word in words:
        text = _core._word_text(word)
        if not text:
            continue
        separator = _core._word_separator(language)
        if text_parts and separator:
            text_parts.append(separator)
            cursor += len(separator)
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        word_spans.append((start, cursor, word))
    full_text = ''.join(text_parts).strip()
    if not full_text or not word_spans:
        return []
    segmenter_language = _core._sentence_segmenter_language(language)
    try:
        segmenter = pysbd.Segmenter(language=segmenter_language, clean=False)
        sentence_texts = [sentence.strip() for sentence in segmenter.segment(full_text) if sentence.strip()]
    except Exception as exc:
        _core.log.warning('句子切分失败，已改用标点切分')
        _core.log.debug('pysbd segment failed language=%s', segmenter_language, exc_info=exc)
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
        segment = _core._segment_from_words(sentence_words, language)
        if segment is not None:
            segment['_whisper_pysbd_language'] = segmenter_language
            segment['_whisper_segmentation_method'] = 'pysbd'
            segments.append(segment)
        search_from = sentence_end
    return segments

def _regroup_words_by_punctuation(words: list[dict], language: str) -> list[dict]:
    regrouped = []
    current_words = []

    def flush() -> None:
        nonlocal current_words
        segment = _core._segment_from_words(current_words, language)
        if segment is None:
            current_words = []
            return
        segment['_whisper_segmentation_method'] = 'punctuation'
        regrouped.append(segment)
        current_words = []
    for word in words:
        text = _core._word_text(word)
        if not text:
            continue
        if 'start' not in word or 'end' not in word:
            flush()
            continue
        current_words.append(word)
        if _core.SENTENCE_END_RE.search(text):
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
        sentence_segments = _core._segment_words_with_pysbd(current_words, language)
        if not sentence_segments:
            sentence_segments = _core._regroup_words_by_punctuation(current_words, language)
        pysbd_segments.extend(sentence_segments)
        current_words = []
        current_speaker = None
    for segment in segments:
        words = segment.get('words') or []
        if not words:
            flush()
            pysbd_segments.append({**segment, '_whisper_segmentation_method': 'aligned_no_words'})
            continue
        for word in words:
            text = _core._word_text(word)
            if not text:
                continue
            if 'start' not in word or 'end' not in word:
                flush()
                continue
            speaker = _core._speaker_label(word)
            if current_words and speaker and current_speaker and (speaker != current_speaker):
                flush()
            current_words.append(word)
            if speaker:
                current_speaker = speaker
    flush()
    if not pysbd_segments:
        pysbd_segments = [{**segment, '_whisper_segmentation_method': 'aligned_passthrough'} for segment in segments]
    for index, segment in enumerate(pysbd_segments):
        segment['_whisper_pysbd_index'] = index
    long_split_segments = []
    for index, segment in enumerate(pysbd_segments):
        long_split_segments.extend(_core._split_long_segment_on_minor_punctuation(segment, pysbd_index=index))
    return (pysbd_segments, long_split_segments)
