"""Compare the two labeled voices in podcast-6 using their median pitch.

This is a quick check for a recording known to contain one male and one female
voice. It estimates perceived voice type; it cannot establish a person's sex.

Example:
    python scripts/podcast6_voice_pitch.py /path/to/podcast-6-audio.mp3
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import librosa
import numpy as np


SAMPLE_RATE = 16_000
DEFAULT_SRT = Path.home() / "Desktop/podcast-speaker-srt/podcast-6.with-speakers.srt"
TIMING = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$")
SPEAKER = re.compile(r"^(SPEAKER_\d+):\s*")


def _milliseconds(parts: tuple[str, ...]) -> int:
    hours, minutes, seconds, millis = map(int, parts)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def read_labeled_srt(path: Path) -> dict[str, list[tuple[int, int]]]:
    cues: dict[str, list[tuple[int, int]]] = {}
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        timing = TIMING.fullmatch(lines[1].strip())
        speaker = SPEAKER.match(lines[2])
        if not timing or not speaker:
            continue
        start = _milliseconds(timing.groups()[:4])
        end = _milliseconds(timing.groups()[4:])
        if end - start >= 1_500:
            cues.setdefault(speaker.group(1), []).append((start, end))
    if len(cues) != 2:
        raise ValueError(f"Expected exactly two labeled speakers; found {sorted(cues)}")
    return cues


def spread_samples(cues: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    """Take speech throughout the episode instead of only its opening."""
    if len(cues) <= count:
        return cues
    indexes = np.linspace(0, len(cues) - 1, count, dtype=int)
    return [cues[index] for index in indexes]


def decode_clip(audio_path: Path, start_ms: int, end_ms: int) -> np.ndarray:
    # Stay inside the subtitle interval and cap long cues at four seconds.
    start = (start_ms + 200) / 1000
    duration = min((end_ms - start_ms - 400) / 1000, 4.0)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(audio_path), "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return np.frombuffer(result.stdout, dtype="<f4")


def clip_pitch(audio: np.ndarray) -> float | None:
    if len(audio) < SAMPLE_RATE:
        return None
    pitches = librosa.yin(
        audio, sr=SAMPLE_RATE, fmin=65, fmax=450,
        frame_length=2048, hop_length=320,
    )
    energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=320)[0]
    length = min(len(pitches), len(energy))
    pitches, energy = pitches[:length], energy[:length]
    # Ignore quiet frames and use one median per cue so long cues do not dominate.
    usable = np.isfinite(pitches) & (energy >= max(np.quantile(energy, 0.3), energy.max() * 0.08))
    return float(np.median(pitches[usable])) if np.count_nonzero(usable) >= 8 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Podcast-6 source audio/video or clean vocal track")
    parser.add_argument("--srt", type=Path, default=DEFAULT_SRT, help=f"Labeled SRT (default: {DEFAULT_SRT})")
    parser.add_argument("--samples", type=int, default=24, help="Maximum speech cues sampled per speaker (default: 24)")
    args = parser.parse_args()
    if not args.audio.is_file() or not args.srt.is_file():
        parser.error("Audio and labeled SRT files must exist")
    if args.samples < 1:
        parser.error("--samples must be positive")

    pitches_by_speaker: dict[str, float] = {}
    for speaker, cues in read_labeled_srt(args.srt).items():
        samples = []
        for start, end in spread_samples(cues, args.samples):
            pitch = clip_pitch(decode_clip(args.audio, start, end))
            if pitch is not None:
                samples.append(pitch)
        if len(samples) < 3:
            raise RuntimeError(f"Only {len(samples)} usable speech clips for {speaker}")
        median = float(np.median(samples))
        pitches_by_speaker[speaker] = median
        print(f"{speaker}: median F0 {median:.1f} Hz from {len(samples)} clips")

    lower, higher = sorted(pitches_by_speaker, key=pitches_by_speaker.get)
    ratio = pitches_by_speaker[higher] / pitches_by_speaker[lower]
    print(f"Estimated mapping (assuming one male and one female voice): {lower}=male, {higher}=female")
    print(f"Median pitch ratio: {ratio:.2f}x")
    if ratio < 1.25:
        print("Caution: pitch separation is small; listen to representative clips before using these labels.")


if __name__ == "__main__":
    main()
