"""Transcribe mixed-language CloseRead speech with native word timestamps."""

import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from . import db, storage, whisper_asr
from .closeread_repair import repair_up_to_three_rounds


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    device = whisper_asr._runtime_device_for_engine()
    return WhisperModel(whisper_asr._model_name_or_path(), device=device, num_workers=2,
                        compute_type="int8" if device == "cpu" else "float16",
                        download_root=whisper_asr.WHISPER_DOWNLOAD_ROOT)


@lru_cache(maxsize=1)
def _aligner():
    from stable_whisper import load_faster_whisper

    device = whisper_asr._runtime_device_for_engine()
    return load_faster_whisper(whisper_asr._model_name_or_path(), device=device,
                               compute_type="int8" if device == "cpu" else "float16",
                               download_root=whisper_asr.WHISPER_DOWNLOAD_ROOT)


def transcribe(task_id):
    with db.connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT audio_url, speech_blocks, generation FROM product_closeread_artifact WHERE task_id=%s", (task_id,))
        audio_url, speech_blocks, generation = cur.fetchone()
        cur.execute("""SELECT body FROM product_closeread_reading_part
                       WHERE task_id=%s AND generation=%s ORDER BY position""", (task_id, generation))
        script = "\n".join(body for (body,) in cur.fetchall())
        if not script:
            speech = json.loads(speech_blocks or "null")
            if isinstance(speech, dict) and speech.get("parts"):
                script = "\n".join(part["text"] for part in speech["parts"])
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
        diagnostics = []
        if script:
            from faster_whisper.audio import decode_audio

            timeline, diagnostics = repair_up_to_three_rounds(
                timeline, script, duration, decode_audio(str(audio)), _model(), _aligner,
            )
            for index, word in enumerate(timeline, 1):
                word["word_id"] = index
            recovered = [{"words": [{"word": word["text"], "start": word["start"], "end": word["end"]}]}
                         for word in timeline if "timing_source" in word]
            database_segments.extend(recovered)
            database_segments.sort(key=lambda segment: segment["words"][0]["start"]
                                   if segment["words"] else duration)
        output = directory / "words.json"
        output.write_text(json.dumps({"duration": duration, "words": timeline, "gap_repairs": diagnostics}, ensure_ascii=False),
                          encoding="utf-8")
        words_url = storage.upload(output, f"{task_id}/closeread/words.json", "application/json")
        db.save_word_timestamps(task_id, database_segments)
    with db.connect() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE product_closeread_artifact SET words_url=%s WHERE task_id=%s",
                    (words_url, task_id))
        conn.commit()
    return {}
