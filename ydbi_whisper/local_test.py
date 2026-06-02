from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
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
    stat = audio.stat()
    digest = hashlib.sha256()
    with audio.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    audio_sha256 = digest.hexdigest()
    stable_task_id = task_id or f"local-test-{audio_sha256[:20]}-{language}"
    session = task_work_dir(stable_task_id)
    db.upsert_whisper_local_test_task(
        stable_task_id,
        audio_path=str(audio),
        audio_size=stat.st_size,
        audio_mtime_ns=stat.st_mtime_ns,
        audio_sha256=audio_sha256,
        language=language,
        status="running",
    )
    log.info(
        "local asr test start task=%s audio=%s language=%s output_dir=%s session=%s",
        stable_task_id,
        audio,
        language,
        target_dir,
        session,
    )

    try:
        data = recognize_speech(audio, session, language=language, task_id=stable_task_id)
        utterances = (data.get("result") or {}).get("utterances") or []
        semantic_debug_path = str((data.get("result") or {}).get("semantic_caption_debug_path") or "").strip()
        semantic_debug_output_path = target_dir / f"{stem}.semantic_caption_segmentation.json"
        if semantic_debug_path and Path(semantic_debug_path).is_file():
            shutil.copy2(semantic_debug_path, semantic_debug_output_path)
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": stable_task_id,
            "input_file": str(audio),
            "language": language,
            "config": current_asr_config(language=language),
            "segments": len(utterances),
            "text_chars": len(str((data.get("result") or {}).get("text") or "").strip()),
            "payload_path": str(payload_path.resolve()),
            "meta_path": str(meta_path.resolve()),
            "semantic_caption_debug_path": str(semantic_debug_output_path.resolve()) if semantic_debug_output_path.is_file() else semantic_debug_path,
        }

        payload_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        db.upsert_whisper_local_test_task(
            stable_task_id,
            audio_path=str(audio),
            audio_size=stat.st_size,
            audio_mtime_ns=stat.st_mtime_ns,
            audio_sha256=audio_sha256,
            language=language,
            status="success",
            payload_path=str(payload_path.resolve()),
            meta_path=str(meta_path.resolve()),
            semantic_debug_path=meta["semantic_caption_debug_path"],
        )
        log.info(
            "local asr test done task=%s audio=%s segments=%s text_chars=%s payload=%s meta=%s",
            stable_task_id,
            audio,
            meta["segments"],
            meta["text_chars"],
            payload_path.resolve(),
            meta_path.resolve(),
        )

        return {"payload": data, "meta": meta}
    finally:
        log.info("local asr cleanup session=%s", session)
        shutil.rmtree(session, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local YouBi Whisper ASR test for one audio file.")
    parser.add_argument("audio_path", help="Absolute path to the local audio file.")
    parser.add_argument("--language", default="en", help="ASR language code. Default: en.")
    parser.add_argument("--output-dir", default="test_outputs", help="Directory for JSON outputs. Default: test_outputs.")
    parser.add_argument("--task-id", default="", help="Optional stable task_id for resumable local tests.")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG logs from WhisperX and dependency libraries.")
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    result = run_local_asr(args.audio_path, language=args.language, output_dir=args.output_dir, task_id=args.task_id or None)
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
