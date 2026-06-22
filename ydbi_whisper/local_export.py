from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from .asr_segments import fix_asr_segment_rows
from .config import task_work_dir
from .logging_utils import configure_dependency_logging
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
    configure_dependency_logging(debug=debug)


def _srt_timestamp(ms: int) -> str:
    value = max(0, int(ms or 0))
    hours, remainder = divmod(value, 60 * 60 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    index = 1
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = _srt_timestamp(int(segment.get("start_time") or 0))
        end = _srt_timestamp(int(segment.get("end_time") or 0))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_txt(segments: list[dict[str, Any]], fallback_text: str = "") -> str:
    lines = [str(segment.get("text") or "").strip() for segment in segments]
    text = "\n".join(line for line in lines if line).strip()
    if not text:
        text = fallback_text.strip()
    return text + ("\n" if text else "")


def transcribe_local_audio(
    audio_path: str | Path,
    *,
    language: str = "en",
    output_dir: str | Path | None = "test_outputs",
    task_id: str | None = None,
    write_files: bool = True,
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"audio file not found: {audio}")

    task_id = task_id or f"local-export-{uuid.uuid4().hex}-{language}"
    session = task_work_dir(task_id)
    log.info("local export start task_id=%s audio=%s language=%s session=%s", task_id, audio, language, session)

    try:
        data = recognize_speech(audio, session, language=language)
        duration_ms = int((data.get("audio_info") or {}).get("duration") or 0)
        result = data.get("result") or {}
        segments = fix_asr_segment_rows(result.get("utterances") or [], duration_ms)
        srt = render_srt(segments)
        txt = render_txt(segments, str(result.get("text") or ""))

        response: dict[str, Any] = {
            "task_id": task_id,
            "input_file": str(audio),
            "language": language,
            "config": current_asr_config(language=language),
            "segments": segments,
            "srt": srt,
            "txt": txt,
        }

        if write_files:
            if output_dir is None:
                raise ValueError("output_dir is required when write_files=True")
            target_dir = Path(output_dir).expanduser()
            target_dir.mkdir(parents=True, exist_ok=True)
            srt_path = target_dir / f"{audio.stem}.srt"
            txt_path = target_dir / f"{audio.stem}.txt"
            srt_path.write_text(srt, encoding="utf-8")
            txt_path.write_text(txt, encoding="utf-8")
            response["srt_path"] = str(srt_path.resolve())
            response["txt_path"] = str(txt_path.resolve())
            log.info("local export done task_id=%s srt=%s txt=%s", task_id, srt_path.resolve(), txt_path.resolve())
        else:
            log.info("local export done task_id=%s segments=%s", task_id, len(segments))

        return response
    finally:
        shutil.rmtree(session, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a local audio file and export SRT/TXT without writing DB rows.")
    parser.add_argument("audio_path", help="Path to the local audio file.")
    parser.add_argument("--language", default="en", help="ASR language code. Default: en.")
    parser.add_argument("--output-dir", default="test_outputs", help="Directory for SRT/TXT outputs. Default: test_outputs.")
    parser.add_argument("--task-id", default=None, help="Task id used for temporary work dir. Default: generated local-export id.")
    parser.add_argument("--no-files", action="store_true", help="Do not write SRT/TXT files; print both strings as JSON.")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG logs.")
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    result = transcribe_local_audio(
        args.audio_path,
        language=args.language,
        output_dir=args.output_dir,
        task_id=args.task_id,
        write_files=not args.no_files,
    )
    print(json.dumps({
        "task_id": result["task_id"],
        "input_file": result["input_file"],
        "language": result["language"],
        "segments": len(result["segments"]),
        "srt_path": result.get("srt_path"),
        "txt_path": result.get("txt_path"),
        "srt": result["srt"] if args.no_files else None,
        "txt": result["txt"] if args.no_files else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
