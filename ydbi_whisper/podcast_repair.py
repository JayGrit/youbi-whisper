"""Quality gate and bounded re-recognition for podcast dialogue subtitles."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path

from . import db
from .asr_segments import fix_asr_segment_rows
from .podcast_dialogue import split_podcast_dialogue_segments
from .whisper_asr import NoSpeechDetected, recognize_speech

log = logging.getLogger(__name__)

MIN_WORDS_PER_SECOND = 0.3
MAX_LOCAL_WINDOWS = 4
MAX_LOCAL_WINDOW_MS = 90_000
CONTEXT_MS = 500


def prepare_segments(data: dict, run_id: int) -> list[dict]:
    result = data.get("result") or {}
    duration_ms = int((data.get("audio_info") or {}).get("duration") or 0)
    segments = fix_asr_segment_rows(result.get("utterances") or [], duration_ms)
    segments = split_podcast_dialogue_segments(segments)
    return [{**item, "_whisper_run_id": run_id} for item in segments]


def slow_segments(segments: list[dict]) -> list[tuple[int, float]]:
    found = []
    for index, item in enumerate(segments):
        duration_ms = int(item["end_time"]) - int(item["start_time"])
        if duration_ms <= 0:
            raise ValueError(f"Podcast subtitle {index} has invalid timing")
        word_count = len(str(item.get("text") or "").split())
        if word_count == 0:
            raise ValueError(f"Podcast subtitle {index} is empty")
        words_per_second = word_count * 1000 / duration_ms
        if words_per_second < MIN_WORDS_PER_SECOND:
            found.append((index, words_per_second))
    return found


def _groups(indexes: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    for index in indexes:
        if groups and index == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], index)
        else:
            groups.append((index, index))
    return groups


def _caption_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _anchor_index(segments: list[dict], original: dict) -> int:
    key = _caption_key(str(original["text"]))
    matches = [index for index, item in enumerate(segments)
               if _caption_key(str(item.get("text") or "")) == key]
    if len(matches) != 1:
        raise ValueError(f"Podcast repair anchor is missing or ambiguous: {original['text']!r}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop_audio(source: Path, destination: Path, start_ms: int, end_ms: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_ms / 1000:.3f}", "-i", str(source),
        "-t", f"{(end_ms - start_ms) / 1000:.3f}",
        "-vn", "-ac", "1", "-ar", "16000", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Podcast audio crop failed: {completed.stderr.strip()}")


def _recognize_attempt(
    *, task_id: str, language: str, source_url: str, input_audio_url: str,
    audio: Path, session: Path, parent_run_id: int, start_ms: int, end_ms: int,
    allow_no_speech: bool = False, persist: bool = True,
) -> list[dict]:
    run_id = (
        db.create_whisper_run(
            task_id=task_id,
            language=language,
            source_url=source_url,
            input_audio_url=input_audio_url,
            input_local_path=str(audio),
            input_file_size=audio.stat().st_size,
            input_sha256=_sha256(audio),
            sub_stage="dialogue",
        )
        if persist else None
    )
    try:
        if run_id is not None:
            db.annotate_whisper_repair_run(run_id, parent_run_id, start_ms, end_ms)
        try:
            data = recognize_speech(
                audio, session, language=language,
                task_id=task_id if persist else None, run_id=run_id,
                diarize=False,
            )
        except NoSpeechDetected:
            if not allow_no_speech:
                raise
            segments: list[dict] = []
        else:
            segments = prepare_segments(data, run_id or 0)
        if run_id is not None:
            db.finish_whisper_run(run_id, "success")
        return segments
    except Exception as exc:
        if run_id is not None:
            db.finish_whisper_run(run_id, "failed", str(exc))
        raise


def _shift_segments(segments: list[dict], offset_ms: int) -> list[dict]:
    shifted = []
    for item in segments:
        words = [
            {**word,
             "start_time": int(word["start_time"]) + offset_ms,
             "end_time": int(word["end_time"]) + offset_ms}
            for word in item.get("words") or []
        ]
        shifted.append({
            **item,
            "start_time": int(item["start_time"]) + offset_ms,
            "end_time": int(item["end_time"]) + offset_ms,
            "words": words,
        })
    return shifted


def _repair_group(
    segments: list[dict], first: int, last: int, *,
    task_id: str, language: str, source_url: str, input_audio_url: str,
    vocals: Path, session: Path, parent_run_id: int, duration_ms: int,
    persist: bool = True,
) -> list[dict]:
    left = segments[first - 1] if first else None
    right = segments[last + 1] if last + 1 < len(segments) else None
    start_ms = max(0, int(left["start_time"] if left else segments[first]["start_time"]) - CONTEXT_MS)
    end_ms = min(duration_ms, int(right["end_time"] if right else segments[last]["end_time"]) + CONTEXT_MS)
    if end_ms <= start_ms or end_ms - start_ms > MAX_LOCAL_WINDOW_MS:
        raise ValueError(f"Podcast repair window is invalid or too long: {start_ms}-{end_ms}")

    clip = session / "repair" / f"{first}-{last}-{start_ms}-{end_ms}.wav"
    _crop_audio(vocals, clip, start_ms, end_ms)
    log.warning(
        "Podcast %s: retrying subtitle %s-%s in audio window %s-%s ms",
        task_id, first, last, start_ms, end_ms,
    )
    candidate = _recognize_attempt(
        task_id=task_id, language=language, source_url=source_url,
        input_audio_url=input_audio_url, audio=clip, session=session,
        parent_run_id=parent_run_id, start_ms=start_ms, end_ms=end_ms,
        allow_no_speech=True, persist=persist,
    )
    candidate = _shift_segments(candidate, start_ms)
    left_index = _anchor_index(candidate, left) if left else -1
    right_index = _anchor_index(candidate, right) if right else len(candidate)
    if left_index >= right_index:
        raise ValueError("Podcast repair anchors are out of order")
    replacement = candidate[left_index + 1:right_index]
    if not replacement and any(len(str(item["text"]).split()) != 1 for item in segments[first:last + 1]):
        raise ValueError("Podcast repair would remove a multiword subtitle")
    lower_bound = int(left["end_time"]) + 50 if left else 0
    upper_bound = int(right["start_time"]) - 50 if right else duration_ms
    if replacement and int(replacement[0]["start_time"]) < lower_bound:
        if lower_bound - int(replacement[0]["start_time"]) > 250:
            raise ValueError("Podcast repair overlaps the previous subtitle")
        replacement[0]["start_time"] = lower_bound
        for word in replacement[0].get("words") or []:
            word["start_time"] = max(int(word["start_time"]), lower_bound)
    if replacement and int(replacement[-1]["end_time"]) > upper_bound:
        if int(replacement[-1]["end_time"]) - upper_bound > 250:
            raise ValueError("Podcast repair overlaps the next subtitle")
        replacement[-1]["end_time"] = upper_bound
        for word in replacement[-1].get("words") or []:
            word["end_time"] = min(int(word["end_time"]), upper_bound)
    if any(int(item["start_time"]) < lower_bound or int(item["end_time"]) > upper_bound
           or int(item["end_time"]) <= int(item["start_time"]) for item in replacement):
        raise ValueError("Podcast repair would overlap a preserved subtitle")
    return segments[:first] + replacement + segments[last + 1:]


def repair_podcast_segments(
    segments: list[dict], *, task_id: str, language: str, source_url: str,
    input_audio_url: str, vocals: Path, session: Path, parent_run_id: int,
    duration_ms: int, persist: bool = True,
) -> list[dict]:
    if language.lower() != "en":
        return segments
    anomalies = slow_segments(segments)
    if not anomalies:
        return segments
    log.warning("Podcast %s: %s subtitles below %.3f words/s: %s",
                task_id, len(anomalies), MIN_WORDS_PER_SECOND, anomalies)
    original_error: Exception | None = None
    groups = _groups([index for index, _ in anomalies])
    if len(groups) <= MAX_LOCAL_WINDOWS:
        try:
            repaired = segments
            for first, last in reversed(groups):
                repaired = _repair_group(
                    repaired, first, last, task_id=task_id, language=language,
                    source_url=source_url, input_audio_url=input_audio_url,
                    vocals=vocals, session=session, parent_run_id=parent_run_id,
                    duration_ms=duration_ms, persist=persist,
                )
            if not slow_segments(repaired):
                log.info("Podcast %s: local subtitle repair accepted; segments %s -> %s",
                         task_id, len(segments), len(repaired))
                return repaired
            raise ValueError(f"Podcast local repair left slow subtitles: {slow_segments(repaired)}")
        except Exception as exc:
            original_error = exc
            log.warning("Podcast %s: local repair failed; retrying full audio: %s", task_id, exc)
    else:
        original_error = ValueError(f"Too many local repair windows: {len(groups)}")

    try:
        full = _recognize_attempt(
            task_id=task_id, language=language, source_url=source_url,
            input_audio_url=input_audio_url, audio=vocals, session=session,
            parent_run_id=parent_run_id, start_ms=0, end_ms=duration_ms,
            persist=persist,
        )
        remaining = slow_segments(full)
        if remaining:
            raise ValueError(f"Full retry left slow subtitles: {remaining}")
        log.info("Podcast %s: full recognition retry accepted; segments=%s", task_id, len(full))
        return full
    except Exception as exc:
        raise RuntimeError(
            f"Podcast subtitle quality check failed: local={original_error}; full={exc}"
        ) from exc
