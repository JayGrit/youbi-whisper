from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

from .config import WHISPER_DOWNLOAD_ROOT, WHISPER_MODEL, device

_MODEL = None
log = logging.getLogger(__name__)


def _load_model():
    global _MODEL
    if _MODEL is None:
        import whisper

        _MODEL = whisper.load_model(
            WHISPER_MODEL,
            device=device(),
            download_root=WHISPER_DOWNLOAD_ROOT or None,
        )
    return _MODEL


def _to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def _convert_words(words: list) -> list:
    return [
        {
            "text": w.get("word", ""),
            "start_time": _to_ms(w.get("start", 0.0)),
            "end_time": _to_ms(w.get("end", 0.0)),
        }
        for w in words or []
    ]


def _convert_segments(segments: list) -> list:
    return [
        {
            "text": seg.get("text", "").strip(),
            "start_time": _to_ms(seg.get("start", 0.0)),
            "end_time": _to_ms(seg.get("end", 0.0)),
            "words": _convert_words(seg.get("words", [])),
        }
        for seg in segments
    ]


def recognize_speech(vocals_file: Path, session: Path, language: str) -> dict:
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model()
    runtime_device = str(getattr(model, "device", device())).lower()
    word_timestamps = "mps" not in runtime_device
    if not word_timestamps:
        log.warning("Whisper is running on MPS; word timestamps are disabled to avoid MPS float64 DTW failure.")
    result = model.transcribe(
        str(vocals_file),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
    )

    utterances = _convert_segments(result.get("segments", []))
    if not utterances:
        raise RuntimeError("Whisper did not return any segments.")

    duration_ms = len(AudioSegment.from_file(vocals_file))
    payload = {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": (result.get("text") or "").strip(),
            "utterances": utterances,
        },
    }
    return payload
