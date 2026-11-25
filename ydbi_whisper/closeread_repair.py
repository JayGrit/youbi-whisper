"""Recover known CloseRead speech missing from the first ASR pass."""

from difflib import SequenceMatcher
import logging

log = logging.getLogger(__name__)
RATE = 16000
PADDING = 0.5


def _normalize(text):
    return "".join(c for c in text.casefold() if c.isalnum())


def _script_gaps(words, script, duration):
    """Yield gaps bracketed by words matched to the ordered script."""
    original = [i for i, c in enumerate(script) for _ in _normalize(c)]
    normalized = _normalize(script)
    observed = [_normalize(word["text"]) for word in words]
    owners = [i for i, token in enumerate(observed) for _ in token]
    bounds = {}
    for a, b, count in SequenceMatcher(None, normalized, "".join(observed), autojunk=False).get_matching_blocks():
        for shift in range(count):
            index = owners[b + shift]
            position = original[a + shift]
            first, _ = bounds.get(index, (position, position))
            bounds[index] = first, position
    for index in range(len(words) + 1):
        start = words[index - 1]["end"] if index else 0
        end = words[index]["start"] if index < len(words) else duration
        if end - start + 1e-9 < 3:
            continue
        left = bounds.get(index - 1) if index else (-1, -1)
        right = bounds.get(index) if index < len(words) else (len(script), len(script))
        if left and right:
            missing = script[left[1] + 1:right[0]].strip()
            if _normalize(missing):
                yield start, end, missing


def _matched_words(expected, words):
    tokens = [_normalize(word["text"]) for word in words]
    observed = "".join(tokens)
    matches = SequenceMatcher(None, _normalize(expected), observed, autojunk=False).get_matching_blocks()
    matched = {block.b + i: block.a + i for block in matches for i in range(block.size)}
    offset = 0
    for word, token in zip(words, tokens):
        end = offset + len(token)
        if token and all(i in matched for i in range(offset, end)):
            yield word, matched[offset], matched[end - 1] + 1
        offset = end


def repair(words, script, duration, audio, model, aligner, *, forced_words=None):
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    import numpy as np

    gaps = list(_script_gaps(words, script, duration))
    diagnostics = [{"start": start, "end": end, "recovered": 0, "status": "silent"}
                   for start, end, _ in gaps]
    voiced = [(i, start, end, missing) for i, (start, end, missing) in enumerate(gaps)
              if get_speech_timestamps(audio[round(start * RATE):round(end * RATE)],
                                       VadOptions(min_speech_duration_ms=100))]
    if not voiced:
        return words, diagnostics
    forced = forced_words or []
    try:
        result = None if forced_words is not None else aligner().align(audio, script, language="zh", regroup=False, verbose=None)
        if result:
            forced = [{"text": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                      for segment in result.segments for w in segment.words if w.end > w.start]
    except Exception:
        log.exception("CloseRead full-track forced alignment failed")

    chunks, offsets, accepted = [], [], {}
    cursor = 0
    for i, start, end, missing in voiced:
        accepted[i] = [w for w, _, _ in _matched_words(
            missing, [w for w in forced if start <= w["start"] < w["end"] <= end])]
        if _normalize("".join(w["text"] for w in accepted[i])) == _normalize(missing):
            continue
        first, last = round(start * RATE), round(end * RATE)
        chunk = audio[first:last]
        offsets.append((i, cursor, cursor + len(chunk), first))
        chunks.extend((chunk, np.zeros(RATE // 2, dtype=audio.dtype)))
        cursor += len(chunk) + RATE // 2

    recognized = {}
    if chunks:
        try:
            segments, _ = model.transcribe(np.concatenate(chunks), word_timestamps=True,
                                           multilingual=True, condition_on_previous_text=False)
            for segment in segments:
                for w in segment.words or []:
                    begin, finish = round(w.start * RATE), round(w.end * RATE)
                    for i, left, right, original in offsets:
                        if left <= begin < finish <= right:
                            recognized.setdefault(i, []).append({
                                "text": w.word.strip(), "start": (original + begin - left) / RATE,
                                "end": (original + finish - left) / RATE,
                            })
                            break
        except Exception:
            log.exception("CloseRead combined gap recognition failed")

    recovered = []
    for i, start, end, missing in voiced:
        positions = []
        for source, batch in (("retry", recognized.get(i, [])), ("forced", accepted[i])):
            for word, first, last in _matched_words(missing, batch):
                if not start <= word["start"] < word["end"] <= end + 1e-9:
                    continue
                if any(not (last <= lo and word["end"] <= other["start"]
                            or first >= hi and word["start"] >= other["end"])
                       for lo, hi, other in positions):
                    continue
                positions.append((first, last, dict(word, timing_source=source)))
        additions = [word for _, _, word in sorted(positions, key=lambda entry: entry[0])]
        recovered.extend(additions)
        expected = _normalize(missing)
        actual = _normalize("".join(w["text"] for w in additions))
        remaining = sum(a1-a0 for tag, a0, a1, _, _ in
                        SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes()
                        if tag in {"delete", "replace"})
        diagnostics[i].update(recovered=len(additions), remaining_characters=remaining,
                              status="complete" if not remaining else "partial" if additions else "unresolved")
        log.info("CloseRead speech gap: %s", diagnostics[i])
    return sorted([*words, *recovered], key=lambda word: word["start"]), diagnostics


def repair_up_to_three_rounds(words, script, duration, audio, model, aligner):
    """Reuse one forced alignment across at most three batches of gap recognition."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    class CachedAlignment:
        def __init__(self):
            self.loaded = False
            self.result = None

        def align(self, *args, **kwargs):
            if not self.loaded:
                self.loaded = True
                self.result = aligner().align(*args, **kwargs)
            return self.result

    cached = CachedAlignment()
    reports = []
    for round_number in range(1, 4):
        repaired, diagnostics = repair(words, script, duration, audio, model, lambda: cached)
        reports.extend(dict(report, round=round_number) for report in diagnostics)
        if len(repaired) == len(words):
            break
        words = repaired
    for start, end, _ in _script_gaps(words, script, duration):
        if end - start > 30 and get_speech_timestamps(
            audio[round(start * RATE):round(end * RATE)], VadOptions(min_speech_duration_ms=100)
        ):
            raise ValueError(f"CloseRead speech remains unaligned at {start:.2f}–{end:.2f}s "
                             f"({end - start:.2f}s)")
    return words, reports
