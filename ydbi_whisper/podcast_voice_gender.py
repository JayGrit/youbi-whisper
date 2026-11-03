"""Diarize a two-voice podcast, then map diarized speakers to voice labels."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from . import whisper_asr as _core
from .whisperx_alignment import _load_diarization_pipeline

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
FRAME_LENGTH = 2048
HOP_LENGTH = 320


def _decode_mono(audio_path: Path) -> np.ndarray:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(audio_path),
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Could not decode podcast audio: {result.stderr.decode(errors='replace')[-1000:]}")
    audio = np.frombuffer(result.stdout, dtype="<f4")
    if len(audio) < SAMPLE_RATE:
        raise ValueError("Podcast audio is shorter than one second")
    return audio


def _interval_pitch(audio: np.ndarray, start: float, end: float) -> float | None:
    start_ms, end_ms = round(start * 1000), round(end * 1000)
    duration = end_ms - start_ms
    if duration < 350:
        return None
    margin = min(100, duration // 8)
    left = max(0, (start_ms + margin) * SAMPLE_RATE // 1000)
    right = min(len(audio), (end_ms - margin) * SAMPLE_RATE // 1000)
    if right - left < SAMPLE_RATE * 3 // 10:
        return None
    if right - left > 4 * SAMPLE_RATE:
        center = (left + right) // 2
        left, right = center - 2 * SAMPLE_RATE, center + 2 * SAMPLE_RATE
    clip = audio[left:right]
    pitch = librosa.yin(clip, sr=SAMPLE_RATE, fmin=65, fmax=450,
                        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    energy = librosa.feature.rms(y=clip, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    count = min(len(pitch), len(energy))
    pitch, energy = pitch[:count], energy[:count]
    if count == 0 or energy.max() < 0.001:
        return None
    voiced = np.isfinite(pitch) & (energy >= max(np.quantile(energy, 0.3), energy.max() * 0.08))
    return float(np.median(pitch[voiced])) if np.count_nonzero(voiced) >= 5 else None


def identify_two_voices(audio_path: Path, segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run pyannote diarization for exactly two speakers, then map female=0/male=1."""
    if not segments:
        raise ValueError("Podcast has no subtitle segments")
    audio = _decode_mono(audio_path)
    duration = len(audio) / SAMPLE_RATE
    if max(int(item["end_time"]) for item in segments) > round(duration * 1000) + 500:
        raise ValueError("Subtitle timestamps extend beyond the supplied audio")

    runtime_device = _core._runtime_device_for_engine()
    diarizer = _load_diarization_pipeline(runtime_device, "podcast gender labeling")
    diarization = diarizer(audio, min_speakers=2, max_speakers=2)
    rows = diarization.to_dict("records") if hasattr(diarization, "to_dict") else []
    intervals: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        speaker = str(row.get("speaker") or "").strip()
        start = float(row.get("start") or 0)
        end = float(row.get("end") or start)
        if speaker and end > start:
            intervals.setdefault(speaker, []).append((start, end))
    if len(intervals) != 2:
        raise ValueError(f"Expected exactly two diarized speakers; got {sorted(intervals)}")

    pitch_by_speaker: dict[str, list[float]] = {speaker: [] for speaker in intervals}
    for speaker, turns in intervals.items():
        for start, end in turns:
            pitch = _interval_pitch(audio, start, end)
            if pitch is not None:
                pitch_by_speaker[speaker].append(pitch)
    if any(len(values) < 5 for values in pitch_by_speaker.values()):
        raise ValueError(f"Too few voiced diarization turns to map gender: { {k: len(v) for k, v in pitch_by_speaker.items()} }")
    medians = {speaker: float(np.median(values)) for speaker, values in pitch_by_speaker.items()}
    low_speaker, high_speaker = sorted(medians, key=medians.get)
    if medians[high_speaker] / medians[low_speaker] < 1.25:
        raise ValueError(f"Diarized speaker pitch groups too close: {medians}")
    # Stable labels: female="0", male="1".
    mapping = {high_speaker: "0", low_speaker: "1"}

    labeled: list[dict[str, Any]] = []
    for segment in segments:
        start = int(segment["start_time"]) / 1000
        end = int(segment["end_time"]) / 1000
        overlaps: dict[str, float] = {}
        for speaker, turns in intervals.items():
            overlaps[speaker] = sum(max(0.0, min(end, turn_end) - max(start, turn_start))
                                    for turn_start, turn_end in turns)
        speaker = max(overlaps, key=overlaps.get)
        if overlaps[speaker] <= 0:
            midpoint = (start + end) / 2
            speaker = min(
                intervals,
                key=lambda candidate: min(
                    abs(midpoint - boundary)
                    for turn in intervals[candidate] for boundary in turn
                ),
            )
        labeled.append({**segment, "speaker": mapping[speaker], "diarized_speaker": speaker})
    summary = {
        "diarized_speakers": {speaker: round(medians[speaker], 1) for speaker in intervals},
        "female_speaker": high_speaker,
        "male_speaker": low_speaker,
        "female_median_hz": round(medians[high_speaker], 1),
        "male_median_hz": round(medians[low_speaker], 1),
        "speaker_count": 2,
    }
    log.info("WhisperX diarization and pitch mapping: %s", summary)
    return labeled, summary
