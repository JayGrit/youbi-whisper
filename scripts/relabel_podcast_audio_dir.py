#!/usr/bin/env python3
"""Label locally collected podcast subtitles, one task at a time.

Run with the project's virtualenv. This script changes only
whisper_asr_segment.speaker; it never changes distributor stage states.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ydbi_whisper import db  # noqa: E402
from ydbi_whisper.podcast_voice_gender import identify_two_voices  # noqa: E402

DEFAULT_AUDIO_DIR = Path("/Users/hoshuuch/Desktop/podcast_all_notebooklm/audio")
LOG = logging.getLogger("relabel_podcast_audio_dir")


def subtitle_rows(task_id: str) -> tuple[str, list[dict]]:
    with db.connect() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT t.task_type, w.status AS whisper_status
               FROM task t
               LEFT JOIN distributor_task_stages w
                 ON w.task_id=t.id AND w.stage_name='whisper' AND w.sub_stage='dialogue'
               WHERE t.id=%s""",
            (task_id,),
        )
        stage = cursor.fetchone()
        if not stage or stage["task_type"] not in {"podcast", "podcast_new"}:
            raise ValueError(f"{task_id}: not a podcast/podcast_new task")
        cursor.execute(
            """SELECT item_index, text, start_time, end_time, speaker
               FROM whisper_asr_segment WHERE task_id=%s ORDER BY item_index""",
            (task_id,),
        )
        return str(stage["whisper_status"] or ""), cursor.fetchall()


def signature(rows: list[dict]) -> list[tuple]:
    return [
        (row["item_index"], row["text"], row["start_time"], row["end_time"])
        for row in rows
    ]


def process(audio: Path, output_dir: Path, *, force: bool = False) -> dict:
    task_id = audio.stem
    stage_status, rows = subtitle_rows(task_id)
    if stage_status != "success":
        return {"status": "skipped_whisper_not_success", "whisper_status": stage_status}
    if not rows:
        return {"status": "skipped_no_subtitles"}
    if not force and all(str(row["speaker"]) in {"0", "1"} for row in rows):
        return {"status": "already_labeled", "cues": len(rows)}

    backup = output_dir / "backups" / f"{task_id}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    LOG.info("%s: identifying two speakers from %s", task_id, audio)
    labeled, summary = identify_two_voices(audio, rows)
    if len(labeled) != len(rows) or any(str(row.get("speaker")) not in {"0", "1"} for row in labeled):
        raise RuntimeError(f"{task_id}: labeling did not produce 0/1 for every cue")

    with db.connect() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT item_index, text, start_time, end_time, speaker
               FROM whisper_asr_segment WHERE task_id=%s ORDER BY item_index FOR UPDATE""",
            (task_id,),
        )
        current = cursor.fetchall()
        if signature(current) != signature(rows):
            conn.rollback()
            raise RuntimeError(f"{task_id}: subtitles changed during analysis")
        cursor.executemany(
            "UPDATE whisper_asr_segment SET speaker=%s WHERE task_id=%s AND item_index=%s",
            [(row["speaker"], task_id, row["item_index"]) for row in labeled],
        )
        cursor.execute(
            """SELECT COUNT(*) AS total, SUM(speaker IN ('0','1')) AS labeled
               FROM whisper_asr_segment WHERE task_id=%s""",
            (task_id,),
        )
        check = cursor.fetchone()
        if check["total"] != len(rows) or check["labeled"] != len(rows):
            conn.rollback()
            raise RuntimeError(f"{task_id}: database speaker verification failed: {check}")
        conn.commit()
    LOG.info("%s: labeled %d cues; %s", task_id, len(rows), summary)
    return {"status": "labeled", "cues": len(rows), "voice_summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--only", help="Process one podcast task ID instead of the whole directory")
    parser.add_argument("--force", action="store_true", help="Recheck tasks already labeled 0/1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    audio_dir = args.audio_dir.expanduser().resolve()
    if args.only and not re.fullmatch(r"podcast-\d+", args.only):
        parser.error("--only must be a podcast-N task ID")
    files = sorted(audio_dir.glob("podcast-*.m4a"), key=lambda path: int(path.stem.split("-", 1)[1]))
    if args.only:
        files = [audio_dir / f"{args.only}.m4a"]
    if not files or any(not path.is_file() or path.stat().st_size == 0 for path in files):
        parser.error(f"audio is missing or empty in {audio_dir}")

    output_dir = audio_dir.parent
    report_path = output_dir / "relabel_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    failures = []
    for index, audio in enumerate(files, 1):
        try:
            report[audio.stem] = process(audio, output_dir, force=args.force)
            LOG.info("%d/%d %s: %s", index, len(files), audio.stem, report[audio.stem]["status"])
        except Exception as exc:
            LOG.exception("%d/%d %s: failed", index, len(files), audio.stem)
            report[audio.stem] = {"status": "failed", "error": str(exc)}
            failures.append(audio.stem)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"Failed: {', '.join(failures)}; see {report_path}")


if __name__ == "__main__":
    main()
