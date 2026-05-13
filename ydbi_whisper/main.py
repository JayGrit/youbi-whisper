from __future__ import annotations

import logging
from pathlib import Path

from ydbi_whisper import db
from ydbi_whisper.sources import detect_source
from ydbi_whisper.whisper_asr import recognize_speech
from ydbi_whisper.worker import run_polling_worker

log = logging.getLogger(__name__)


def _ensure_existing_file(path: str | Path, field_name: str) -> Path:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"{field_name} does not exist or is empty: {file_path}")
    return file_path


def handle(row: dict) -> dict[str, str]:
    task_id = row["task_id"]
    vocals = _ensure_existing_file(row["audio_vocals_path"], "audio_vocals_path")
    session = db.session_path_for(task_id)

    task = db.get_task(task_id)
    source = detect_source(task["source_url"])
    log.info(
        "whisper task=%s vocals=%s language=%s",
        task_id,
        vocals,
        source.asr_language,
    )
    data = recognize_speech(vocals, session, language=source.asr_language)
    db.save_asr_result(task_id, source.asr_language, data, "raw")
    utterances = data["result"]["utterances"]
    word_count = sum(len(item.get("words") or []) for item in utterances)
    log.info("whisper recognized task=%s segments=%d words=%d", task_id, len(utterances), word_count)

    asr_ref = f"db://yd_asr_segment/{task_id}/raw"
    log.info("whisper output task=%s asr_ref=%s", task_id, asr_ref)
    db.set_translator_asr_json_path(task_id, asr_ref)
    return {"asr_json_path": asr_ref}


def main() -> None:
    run_polling_worker("whisper", handle)


if __name__ == "__main__":
    main()
