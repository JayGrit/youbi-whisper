"""Podcast dialogue subtitles from final Whisper ASR segments."""

from __future__ import annotations

import math


PODCAST_CUE_MAX_CHARS = 90
PODCAST_LINE_TARGET_CHARS = 50


def _word_text(word: dict) -> str:
    return str(word.get("text") or word.get("word") or "").strip()


def _joined(words: list[dict]) -> str:
    return " ".join(_word_text(word) for word in words)


def _break_score(words: list[dict], index: int, target: float) -> tuple[float, int]:
    left = _joined(words[:index])
    punctuation = _word_text(words[index - 1]).endswith((",", ";", ":", ".", "?", "!"))
    return (abs(len(left) - target) - (7 if punctuation else 0), -index)


def _line_break(words: list[dict]) -> str:
    content = _joined(words)
    if len(content) <= PODCAST_LINE_TARGET_CHARS:
        return content
    choices = [index for index in range(1, len(words))]
    if not choices:
        raise ValueError(f"Podcast subtitle has no word boundary for line break: {content!r}")
    balanced = [index for index in choices
                if max(len(_joined(words[:index])), len(_joined(words[index:]))) <= PODCAST_LINE_TARGET_CHARS]
    candidates = balanced or choices

    def score(index: int) -> tuple[int, int, int]:
        left = _joined(words[:index])
        right = _joined(words[index:])
        # Prefer a natural punctuation boundary only when it does not create
        # noticeably uneven lines; line balance is the primary criterion.
        punctuation = _word_text(words[index - 1]).endswith((",", ";", ":", ".", "?", "!"))
        return abs(len(left) - len(right)), 0 if punctuation else 1, abs(len(left) - len(content) / 2)

    split_at = min(candidates, key=score)
    return _joined(words[:split_at]) + "\n" + _joined(words[split_at:])


def _word_groups(words: list[dict], max_chars: int) -> list[list[dict]]:
    groups = []
    remaining = words
    while remaining:
        if len(_joined(remaining)) <= max_chars:
            groups.append(remaining)
            break
        candidates = [index for index in range(1, len(remaining))
                      if len(_joined(remaining[:index])) <= max_chars]
        if not candidates:
            raise ValueError(f"Podcast subtitle word exceeds {max_chars} characters: {_word_text(remaining[0])!r}")
        target = len(_joined(remaining)) / math.ceil(len(_joined(remaining)) / max_chars)
        split_at = min(candidates, key=lambda index: _break_score(remaining, index, target))
        groups.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return groups


def split_podcast_dialogue_segments(segments: list[dict], *, max_chars: int = PODCAST_CUE_MAX_CHARS) -> list[dict]:
    """Use aligned word boundaries for cues and a visible line break within long cues."""
    if max_chars < PODCAST_LINE_TARGET_CHARS:
        raise ValueError("Podcast subtitle maximum must allow the requested line length")
    result = []
    for segment in segments:
        original = str(segment.get("text") or "").strip()
        words = list(segment.get("words") or [])
        if not original:
            raise ValueError("Podcast subtitle text is empty")
        if not words or _joined(words) != original.replace("\n", " "):
            if len(original) > max_chars:
                raise ValueError(f"Podcast subtitle over {max_chars} characters lacks matching timed words: {original!r}")
            untimed_words = [{"text": token} for token in original.split()]
            result.append({**segment, "text": _line_break(untimed_words)})
            continue
        groups = _word_groups(words, max_chars)
        for index, group in enumerate(groups):
            start = int(segment["start_time"] if index == 0 else group[0]["start_time"])
            end = int(segment["end_time"] if index == len(groups) - 1 else group[-1]["end_time"])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid podcast subtitle word timing: {_joined(group)!r}")
            text = _line_break(group)
            if len(text.replace("\n", " ")) > max_chars:
                raise ValueError(f"Podcast subtitle exceeds {max_chars} characters: {text!r}")
            result.append({**segment, "text": text, "start_time": start, "end_time": end,
                           "words": group})
    return result


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render_dialogue_srt(segments: list[dict]) -> str:
    if not segments:
        raise ValueError("No Whisper dialogue segments are available")
    cues: list[str] = []
    previous_start = -1
    for index, segment in enumerate(segments, 1):
        start = int(segment["start_time"])
        end = int(segment["end_time"])
        content = str(segment.get("text") or "").strip()
        if not content or start < 0 or end <= start or start < previous_start:
            raise ValueError(f"Invalid Whisper dialogue segment {index}")
        speaker = str(segment.get("speaker") or "").strip()
        line = f"{speaker}: {content}" if speaker else content
        cues.append(f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{line}")
        previous_start = start
    return "\n\n".join(cues) + "\n"
