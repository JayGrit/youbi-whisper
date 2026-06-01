from __future__ import annotations

import argparse
import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import task_work_dir
from .whisper_asr import current_asr_config, recognize_speech

log = logging.getLogger(__name__)


def run_local_asr(
    audio_path: str | Path,
    *,
    language: str = "en",
    output_dir: str | Path = "test_outputs",
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")

    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = audio.stem
    payload_path = target_dir / f"{stem}.asr_payload.json"
    meta_path = target_dir / f"{stem}.asr_meta.json"
    session = task_work_dir(f"local-test-{uuid.uuid4().hex}")

    try:
        data = recognize_speech(audio, session, language=language)
        utterances = (data.get("result") or {}).get("utterances") or []
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_file": str(audio),
            "language": language,
            "config": current_asr_config(language=language),
            "segments": len(utterances),
            "text_chars": len(str((data.get("result") or {}).get("text") or "").strip()),
            "payload_path": str(payload_path.resolve()),
            "meta_path": str(meta_path.resolve()),
        }

        payload_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"payload": data, "meta": meta}
    finally:
        shutil.rmtree(session, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local YouBi Whisper ASR test for one audio file.")
    parser.add_argument("audio_path", help="Absolute path to the local audio file.")
    parser.add_argument("--language", default="en", help="ASR language code. Default: en.")
    parser.add_argument("--output-dir", default="test_outputs", help="Directory for JSON outputs. Default: test_outputs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_local_asr(args.audio_path, language=args.language, output_dir=args.output_dir)
    print(json.dumps(result["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
