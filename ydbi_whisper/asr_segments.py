from __future__ import annotations


def _start_pad(idx: int, utts: list, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_start = utts[idx]["start_time"]
    if idx == 0:
        return max(0, orig_start - start_pad)

    prev_end = utts[idx - 1]["end_time"]
    gap = orig_start - prev_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_start - start_pad
    if gap > min_gap:
        share = int((gap - min_gap) * start_pad / total)
        return orig_start - share
    return prev_end + gap // 2


def _end_pad(idx: int, utts: list, duration: int, start_pad: int, end_pad: int, min_gap: int) -> int:
    orig_end = utts[idx]["end_time"]
    if idx == len(utts) - 1:
        return min(duration, orig_end + end_pad) if duration else orig_end + end_pad

    next_start = utts[idx + 1]["start_time"]
    gap = next_start - orig_end
    total = start_pad + end_pad

    if gap >= total + min_gap:
        return orig_end + end_pad
    if gap > min_gap:
        share = int((gap - min_gap) * end_pad / total)
        return orig_end + share
    return orig_end + gap // 2


def _apply_padding(utts: list, duration: int, start_pad: int, end_pad: int) -> list:
    if not utts:
        return utts

    min_gap = 50
    result = []
    for idx in range(len(utts)):
        new_start = _start_pad(idx, utts, start_pad, end_pad, min_gap)
        new_end = _end_pad(idx, utts, duration, start_pad, end_pad, min_gap)
        clamped_end = min(duration, new_end) if duration else new_end
        result.append({
            **utts[idx],
            "start_time": max(0, new_start),
            "end_time": clamped_end,
        })
    return result


def _speaker(utterance: dict) -> str | None:
    speaker = str(utterance.get("speaker") or "").strip()
    if speaker:
        return speaker
    additions = utterance.get("additions") or {}
    if isinstance(additions, dict):
        speaker = str(additions.get("speaker") or "").strip()
    return speaker or None


def normalize_asr_segments(utterances: list[dict]) -> list[dict]:
    return [
        {
            "text": text,
            "start_time": int(u.get("start_time") or 0),
            "end_time": int(u.get("end_time") or 0),
            "speaker": _speaker(u),
            "words": u.get("words"),
            "words_json": u.get("words_json"),
        }
        for u in utterances
        if (text := str(u.get("text") or "").strip())
    ]


def fix_asr_segment_rows(
    utterances: list[dict],
    duration: int,
    start_pad: int = 100,
    end_pad: int = 300,
) -> list[dict]:
    new_utts = normalize_asr_segments(utterances)
    if not new_utts:
        raise RuntimeError("ASR result has no utterances.")
    return _apply_padding(new_utts, duration, start_pad, end_pad)
