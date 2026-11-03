"""Lightweight two-voice labeling for NotebookLM podcasts.

This relies on the task contract: exactly one noticeably lower-pitched male
voice and one higher-pitched female voice. Labels are strings because the
database speaker column is VARCHAR: female="0", male="1".
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
FRAME_LENGTH = 2048
HOP_LENGTH = 320


def _decode_mono(audio_path: Path) -> np.ndarray:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Could not decode podcast audio: {result.stderr.decode(errors='replace')[-1000:]}")
    audio = np.frombuffer(result.stdout, dtype="<f4")
    if len(audio) < SAMPLE_RATE:
        raise ValueError("Podcast audio is shorter than one second")
    return audio


def _cue_pitch(audio: np.ndarray, start_ms: int, end_ms: int) -> float | None:
    duration = end_ms - start_ms
    if duration < 350:
        return None
    margin = min(100, duration // 8)
    start = max(0, (start_ms + margin) * SAMPLE_RATE // 1000)
    end = min(len(audio), (end_ms - margin) * SAMPLE_RATE // 1000)
    if end - start < SAMPLE_RATE * 3 // 10:
        return None
    # Four seconds is enough to measure a voice without letting long cues
    # dominate processing time or include a later speaker turn.
    if end - start > 4 * SAMPLE_RATE:
        midpoint = (start + end) // 2
        start, end = midpoint - 2 * SAMPLE_RATE, midpoint + 2 * SAMPLE_RATE
    clip = audio[start:end]
    pitches = librosa.yin(
        clip, sr=SAMPLE_RATE, fmin=65, fmax=450,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    rms = librosa.feature.rms(
        y=clip, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )[0]
    count = min(len(pitches), len(rms))
    pitches, rms = pitches[:count], rms[:count]
    if not count or rms.max() < 0.001:
        return None
    usable = np.isfinite(pitches) & (rms >= max(np.quantile(rms, 0.3), rms.max() * 0.08))
    if np.count_nonzero(usable) < 5:
        return None
    return float(np.median(pitches[usable]))


def _two_pitch_centers(pitches: list[float]) -> tuple[float, float]:
    values = np.log2(np.asarray(pitches, dtype=np.float64))
    if len(values) < 20:
        raise ValueError(f"Only {len(values)} usable podcast speech cues; need at least 20")
    centers = np.quantile(values, [0.1, 0.9])
    for _ in range(30):
        groups = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        if min(np.count_nonzero(groups == 0), np.count_nonzero(groups == 1)) < 5:
            raise ValueError("Podcast does not contain two measurable voice groups")
        updated = np.array([np.median(values[groups == index]) for index in (0, 1)])
        if np.allclose(updated, centers, atol=1e-5):
            break
        centers = updated
    low, high = sorted(2 ** centers)
    if high / low < 1.25:
        raise ValueError(f"Podcast voice pitch groups are too close: {low:.1f} Hz / {high:.1f} Hz")
    return float(low), float(high)


def label_podcast_segments(audio_path: Path, segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a copy of every subtitle with speaker 0 (female) or 1 (male)."""
    if not segments:
        raise ValueError("Podcast has no subtitle segments")
    audio = _decode_mono(audio_path)
    duration_ms = round(len(audio) * 1000 / SAMPLE_RATE)
    if max(int(item["end_time"]) for item in segments) > duration_ms + 500:
        raise ValueError("Podcast subtitle timestamps extend beyond the supplied audio")

    pitches: list[float | None] = []
    for item in segments:
        start, end = int(item["start_time"]), int(item["end_time"])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid podcast subtitle timing: {start}-{end}")
        pitches.append(_cue_pitch(audio, start, end))
    training = [pitch for item, pitch in zip(segments, pitches)
                if pitch is not None and int(item["end_time"]) - int(item["start_time"]) >= 1000]
    low, high = _two_pitch_centers(training)
    threshold = (low * high) ** 0.5

    labels: list[str | None] = [None if pitch is None else ("1" if pitch < threshold else "0")
                                for pitch in pitches]
    measured = [(index, (int(item["start_time"]) + int(item["end_time"])) // 2)
                for index, item in enumerate(segments) if labels[index] is not None]
    if not measured:
        raise ValueError("Podcast has no measurable speech cues")
    for index, label in enumerate(labels):
        if label is not None:
            continue
        midpoint = (int(segments[index]["start_time"]) + int(segments[index]["end_time"])) // 2
        nearest = min(measured, key=lambda entry: abs(entry[1] - midpoint))[0]
        labels[index] = labels[nearest]
    result = [{**item, "speaker": label} for item, label in zip(segments, labels)]
    if {item["speaker"] for item in result} != {"0", "1"}:
        raise ValueError("Podcast labels did not include both voices")
    summary = {
        "female_median_hz": round(high, 1),
        "male_median_hz": round(low, 1),
        "measured_cues": sum(pitch is not None for pitch in pitches),
        "total_cues": len(segments),
    }
    log.info("Podcast voice labels: %s", summary)
    return result, summary
