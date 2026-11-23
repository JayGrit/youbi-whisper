"""Transcribe mixed-language CloseRead speech with native word timestamps."""

import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from . import db, storage, whisper_asr


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    device = whisper_asr._runtime_device_for_engine()
    return WhisperModel(whisper_asr._model_name_or_path(), device=device,
                        compute_type="int8" if device == "cpu" else "float16",
                        download_root=whisper_asr.WHISPER_DOWNLOAD_ROOT)


def transcribe(task_id):
    with db.connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT audio_url FROM product_closeread_artifact WHERE task_id=%s", (task_id,))
        audio_url = cur.fetchone()[0]
    with TemporaryDirectory() as work:
        directory = Path(work)
        audio = storage.download(audio_url, directory / "speech")
        segments, info = _model().transcribe(str(audio), word_timestamps=True, multilingual=True,
                                            condition_on_previous_text=False)
        duration = float(info.duration)
        timeline = []
        database_segments = []
        for segment in segments:
            words = []
            for word in segment.words or []:
                start, end = max(0.0, word.start), min(duration, word.end)
                if not word.word.strip() or start >= end:
                    continue
                text = word.word.strip()
                words.append({"word": text, "start": start, "end": end})
                timeline.append({"word_id": len(timeline) + 1, "text": text, "start": start, "end": end})
            database_segments.append({"words": words})
        if not timeline:
            raise ValueError("CloseRead audio has no recognized words")
        output = directory / "words.json"
        output.write_text(json.dumps({"duration": duration, "words": timeline}, ensure_ascii=False),
                          encoding="utf-8")
        words_url = storage.upload(output, f"{task_id}/closeread/words.json", "application/json")
        db.save_word_timestamps(task_id, database_segments)
    with db.connect() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE product_closeread_artifact SET words_url=%s WHERE task_id=%s",
                    (words_url, task_id))
        conn.commit()
    return {}
