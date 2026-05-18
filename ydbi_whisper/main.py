from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ydbi_whisper import db
from ydbi_whisper.asr_segments import fix_asr_segment_rows
from ydbi_whisper.config import task_work_dir
from ydbi_whisper.sources import detect_source
from ydbi_whisper.storage import download
from ydbi_whisper.whisper_asr import recognize_speech
from ydbi_whisper.worker import run_polling_worker

log = logging.getLogger(__name__)


def _download_destination(session: Path, source_ref: str) -> Path:
    suffix = Path(source_ref.split("?", 1)[0]).suffix or ".wav"
    return session / "media" / f"audio_vocals{suffix}"


def _vocals_input_for(row: dict, session: Path) -> Path:
    task_id = row["task_id"]
    audio_vocals_url = str(row.get("audio_vocals_url") or "").strip()
    if not audio_vocals_url:
        raise FileNotFoundError(f"audio_vocals_url is missing for task: {task_id}")

    destination = _download_destination(session, audio_vocals_url)
    log.info(
        "whisper task=%s downloading vocals from minio url=%s destination=%s",
        task_id,
        audio_vocals_url,
        destination,
    )
    return download(audio_vocals_url, destination)


def handle(row: dict) -> dict[str, str]:
    task_id = row["task_id"]
    session = task_work_dir(task_id)
    try:
        vocals = _vocals_input_for(row, session)

        task = db.get_task(task_id)
        source = detect_source(task["source_url"])
        log.info(
            "whisper task=%s vocals=%s language=%s",
            task_id,
            vocals,
            source.asr_language,
        )
        data = recognize_speech(vocals, session, language=source.asr_language)
        raw_segments = db.save_asr_result(task_id, source.asr_language, data, "raw")
        duration_ms = int((data.get("audio_info") or {}).get("duration") or 0)
        full_text = str((data.get("result") or {}).get("text") or "")
        fixed_segments = fix_asr_segment_rows(raw_segments, duration_ms)
        db.save_asr_segments(
            task_id,
            "fixed",
            fixed_segments,
            language=source.asr_language,
            duration_ms=duration_ms,
            full_text=full_text,
        )
        word_count = sum(len(item.get("words") or []) for item in raw_segments)
        log.info(
            "whisper recognized task=%s raw_segments=%d fixed_segments=%d words=%d",
            task_id,
            len(raw_segments),
            len(fixed_segments),
            word_count,
        )
    finally:
        shutil.rmtree(session, ignore_errors=True)

    asr_ref = f"db://yd_asr_segment/{task_id}/raw"
    fixed_asr_ref = f"db://yd_asr_segment/{task_id}/fixed"
    log.info("whisper output task=%s asr_ref=%s fixed_asr_ref=%s", task_id, asr_ref, fixed_asr_ref)
    return {"asr_json_path": asr_ref, "asr_fixed_json_path": fixed_asr_ref}


def main() -> None:
    run_polling_worker("whisper", handle)


if __name__ == "__main__":
    main()
