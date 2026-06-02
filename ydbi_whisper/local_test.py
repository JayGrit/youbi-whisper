from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import db
from .config import task_work_dir
from .whisper_asr import current_asr_config, recognize_speech

log = logging.getLogger(__name__)


def configure_logging(*, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in (
        "faster_whisper",
        "huggingface_hub",
        "pyannote",
        "speechbrain",
        "torch",
        "urllib3",
        "whisperx",
    ):
        logging.getLogger(name).setLevel(level)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db_counts(task_id: str) -> dict[str, int]:
    tables = (
        "whisper_run",
        "whisper_raw_segment",
        "whisper_aligned_segment",
        "whisper_aligned_word",
        "whisper_pysbd_segment",
        "whisper_long_split_segment",
        "yd_asr_segment",
        "whisper_word_timestamp",
    )
    with db.connect() as conn:
        cur = conn.cursor()
        counts = {}
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE task_id = %s", (task_id,))
            counts[table] = int(cur.fetchone()[0] or 0)
        return counts


def run_local_asr(
    audio_path: str | Path,
    *,
    language: str = "en",
    output_dir: str | Path = "test_outputs",
    task_id: str | None = None,
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")

    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = audio.stem
    payload_path = target_dir / f"{stem}.asr_payload.json"
    meta_path = target_dir / f"{stem}.asr_meta.json"
    task_id = task_id or f"local-test-{uuid.uuid4().hex}-{language}"
    session = task_work_dir(task_id)
    run_id: int | None = None
    log.info(
        "local asr test start task_id=%s audio=%s language=%s output_dir=%s session=%s",
        task_id,
        audio,
        language,
        target_dir,
        session,
    )

    try:
        run_id = db.create_whisper_run(
            task_id=task_id,
            language=language,
            source_url="local:test",
            input_audio_url=f"local:{audio}",
            input_local_path=str(audio),
            input_file_size=audio.stat().st_size,
            input_sha256=_sha256(audio),
        )
        data = recognize_speech(audio, session, language=language, task_id=task_id, run_id=run_id)
        final_segments = db.save_asr_result(task_id, language, data, run_id=run_id)
        db.finish_whisper_run(run_id, "success")

        utterances = (data.get("result") or {}).get("utterances") or []
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": task_id,
            "run_id": run_id,
            "input_file": str(audio),
            "language": language,
            "config": current_asr_config(language=language),
            "segments": len(utterances),
            "final_segments": len(final_segments),
            "text_chars": len(str((data.get("result") or {}).get("text") or "").strip()),
            "asr_ref": f"db://yd_asr_segment/{task_id}",
            "db_counts": _db_counts(task_id),
            "payload_path": str(payload_path.resolve()),
            "meta_path": str(meta_path.resolve()),
        }

        payload_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(
            "local asr test done task_id=%s run_id=%s audio=%s segments=%s text_chars=%s payload=%s meta=%s",
            task_id,
            run_id,
            audio,
            meta["segments"],
            meta["text_chars"],
            payload_path.resolve(),
            meta_path.resolve(),
        )

        return {"payload": data, "meta": meta}
    except Exception as exc:
        if run_id is not None:
            db.finish_whisper_run(run_id, "failed", str(exc))
        raise
    finally:
        log.info("local asr cleanup session=%s", session)
        shutil.rmtree(session, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local YouBi Whisper ASR test for one audio file.")
    parser.add_argument("audio_path", help="Absolute path to the local audio file.")
    parser.add_argument("--language", default="en", help="ASR language code. Default: en.")
    parser.add_argument("--output-dir", default="test_outputs", help="Directory for JSON outputs. Default: test_outputs.")
    parser.add_argument("--task-id", default=None, help="Task id used for database rows. Default: generated local-test id.")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG logs from WhisperX and dependency libraries.")
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    result = run_local_asr(args.audio_path, language=args.language, output_dir=args.output_dir, task_id=args.task_id)
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
