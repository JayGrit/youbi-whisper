"""Recover known CloseRead speech missing from the first ASR pass."""

from difflib import SequenceMatcher
import logging
import re

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


def repair(words, script, duration, audio, model, aligner):
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    recovered, diagnostics = [], []
    for start, end, missing in _script_gaps(words, script, duration):
        diagnostic = {"start": start, "end": end, "recovered": 0}
        diagnostics.append(diagnostic)
        gap = audio[round(start * RATE):round(end * RATE)]
        if not get_speech_timestamps(gap, VadOptions(min_speech_duration_ms=100)):
            diagnostic["status"] = "silent"
            continue
        left, right = max(0, start - PADDING), min(duration, end + PADDING)
        window = audio[round(left * RATE):round(right * RATE)]
        forced = []
        try:
            language = "zh" if re.search(r"[\u3400-\u9fff]", missing) else "en"
            result = aligner().align(window, missing, language=language, regroup=False, verbose=None)
            if result:
                forced = [{"text": word.word.strip(), "start": left + word.start, "end": left + word.end}
                          for segment in result.segments for word in segment.words
                          if word.end > word.start]
        except Exception:
            log.exception("CloseRead local forced alignment failed at %.2f–%.2f", start, end)

        try:
            # Retain the gap edges so zero-duration boundary words get a second chance.
            if _normalize("".join(word["text"] for word in forced)) == _normalize(missing):
                left = max(left, min(w["start"] for w in forced) - PADDING)
                right = min(right, max(w["end"] for w in forced) + PADDING)
            segments, _ = model.transcribe(
                audio[round(left * RATE):round(right * RATE)], word_timestamps=True,
                multilingual=True, condition_on_previous_text=False,
            )
            candidates = [{"text": word.word.strip(), "start": left + word.start, "end": left + word.end}
                          for segment in segments for word in segment.words or []]
        except Exception:
            log.exception("CloseRead local recognition failed at %.2f–%.2f", start, end)
            candidates = []

        accepted = []
        positions = []
        for source, batch in (("retry", candidates), ("forced", forced)):
            for word, first, last in _matched_words(missing, batch):
                if not start <= word["start"] < word["end"] <= end + 1e-9:
                    continue
                if any(not (last <= lo and word["end"] <= other["start"]
                            or first >= hi and word["start"] >= other["end"])
                       for lo, hi, other in positions):
                    continue
                positions.append((first, last, word))
                accepted.append(dict(word, timing_source=source))
        recovered.extend(accepted)
        expected = _normalize(missing)
        actual = "".join(_normalize(w["text"]) for w in sorted(accepted, key=lambda w: w["start"]))
        remaining = sum(a1-a0 for tag, a0, a1, _, _ in
                        SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes()
                        if tag in {"delete", "replace"})
        diagnostic.update(recovered=len(accepted), remaining_characters=remaining,
                          status="complete" if not remaining else "partial" if accepted else "unresolved")
        log.info("CloseRead speech gap: %s", diagnostic)
    return sorted([*words, *recovered], key=lambda word: word["start"]), diagnostics
