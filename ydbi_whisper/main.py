from __future__ import annotations

import logging
from pathlib import Path

from ydbi_whisper import db
from ydbi_whisper.config import task_work_dir
from ydbi_whisper.sources import detect_source
from ydbi_whisper.storage import download
from ydbi_whisper.whisper_asr import recognize_speech
from ydbi_whisper.worker import run_polling_worker

log = logging.getLogger(__name__)


def _ensure_existing_file(path: str | Path, field_name: str) -> Path:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"{field_name} does not exist or is empty: {file_path}")
    return file_path


def _download_destination(session: Path, source_ref: str) -> Path:
    suffix = Path(source_ref.split("?", 1)[0]).suffix or ".wav"
    return session / "media" / f"audio_vocals{suffix}"


def _vocals_input_for(row: dict, session: Path) -> Path:
    task_id = row["task_id"]
    local_vocals = row.get("audio_vocals_path")
    demucs_operator = db.demucs_operator_for(task_id)
    current_operator = db.current_operator()

    if demucs_operator == current_operator and local_vocals:
        return _ensure_existing_file(local_vocals, "audio_vocals_path")

    audio_vocals_url = str(row.get("audio_vocals_url") or "").strip()
    if not audio_vocals_url:
        if local_vocals:
            return _ensure_existing_file(local_vocals, "audio_vocals_path")
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
    db.save_asr_result(task_id, source.asr_language, data, "raw")
    utterances = data["result"]["utterances"]
    word_count = sum(len(item.get("words") or []) for item in utterances)
    log.info("whisper recognized task=%s segments=%d words=%d", task_id, len(utterances), word_count)

    asr_ref = f"db://yd_asr_segment/{task_id}/raw"
    log.info("whisper output task=%s asr_ref=%s", task_id, asr_ref)
    return {"asr_json_path": asr_ref}


def main() -> None:
    run_polling_worker("whisper", handle)


if __name__ == "__main__":
    main()
