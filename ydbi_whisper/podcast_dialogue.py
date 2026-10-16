"""Podcast dialogue subtitles from persisted Whisper aligned segments."""

from __future__ import annotations


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render_dialogue_srt(segments: list[dict]) -> str:
    if not segments:
        raise ValueError("No Whisper aligned dialogue segments are available")
    cues: list[str] = []
    previous_start = -1
    for index, segment in enumerate(segments, 1):
        start = int(segment["start_time"])
        end = int(segment["end_time"])
        content = str(segment.get("text") or "").strip()
        if not content or start < 0 or end <= start or start < previous_start:
            raise ValueError(f"Invalid Whisper aligned dialogue segment {index}")
        speaker = str(segment.get("speaker") or "").strip()
        line = f"{speaker}: {content}" if speaker else content
        cues.append(f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{line}")
        previous_start = start
    return "\n\n".join(cues) + "\n"
