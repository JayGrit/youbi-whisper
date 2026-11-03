#!/usr/bin/env python3
"""Poll and label podcast subtitles whose uploader stage is not complete."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ydbi_whisper import db
from ydbi_whisper.podcast_voice_gender import identify_two_voices
from ydbi_whisper.storage import download

LOG = logging.getLogger("relabel_unfinished_podcasts")


def claimable_tasks() -> list[dict]:
    with db.connect() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT t.id AS task_id, ts.audio_source_url "
            "FROM task t "
            "JOIN task_source ts ON ts.task_id=t.id "
            "JOIN distributor_task_stages w ON w.task_id=t.id "
            "AND w.stage_name='whisper' AND w.sub_stage='dialogue' AND w.status='success' "
            "LEFT JOIN distributor_task_stages u ON u.task_id=t.id "
            "AND u.stage_name='uploader' AND u.sub_stage='main' "
            "WHERE t.id REGEXP '^podcast-[0-9]+$' AND t.id<>'podcast-23' "
            "AND (u.status IS NULL OR u.status NOT IN ('success','skipped')) "
            "AND COALESCE(ts.audio_source_url,'')<>'' "
            "AND EXISTS (SELECT 1 FROM whisper_asr_segment a WHERE a.task_id=t.id) "
            "ORDER BY CAST(SUBSTRING_INDEX(t.id,'-',-1) AS UNSIGNED)"
        )
        return cursor.fetchall()


def process_task(row: dict, work_dir: Path) -> None:
    task_id = str(row["task_id"])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    audio_path = download(str(row["audio_source_url"]), work_dir / f"{safe_id}.audio")
    with db.connect() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT item_index, text, start_time, end_time, speaker "
            "FROM whisper_asr_segment WHERE task_id=%s ORDER BY item_index",
            (task_id,),
        )
        segments = cursor.fetchall()
    if not segments:
        raise RuntimeError(f"{task_id}: no Whisper subtitles")
    labeled, summary = identify_two_voices(audio_path, segments)
    # Re-check uploader status in the write transaction. A task may have
    # completed while this script downloaded or analyzed its audio.
    with db.connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM distributor_task_stages WHERE task_id=%s "
            "AND stage_name='uploader' AND sub_stage='main' FOR UPDATE",
            (task_id,),
        )
        stage = cursor.fetchone()
        status = stage[0] if stage else None
        if status in ("success", "skipped"):
            conn.rollback()
            LOG.info("%s: uploader completed during analysis; skipped", task_id)
            return
        cursor.executemany(
            "UPDATE whisper_asr_segment SET speaker=%s "
            "WHERE task_id=%s AND item_index=%s",
            [(item["speaker"], task_id, item["item_index"]) for item in labeled],
        )
        conn.commit()
    LOG.info("%s: labeled %d cues (%s)", task_id, len(labeled), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ydbi-podcast-gender"))
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/ydbi-podcast-gender/processed.json"))
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.state_file.read_text() if args.state_file.exists() else "{}")
    if isinstance(state, list):
        state = {task_id: "done" for task_id in state}
    while True:
        rows = [row for row in claimable_tasks() if row["task_id"] not in state]
        if not rows:
            LOG.info("No eligible podcast tasks; polling again in %s seconds", args.poll_seconds)
        for row in rows:
            try:
                process_task(row, args.work_dir)
                state[row["task_id"]] = "done"
                args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            except FileNotFoundError as exc:
                LOG.error("%s: audio object missing; skipping until audio is restored or state cleared: %s",
                          row["task_id"], exc)
                state[row["task_id"]] = "missing_audio"
                args.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            except Exception:
                LOG.exception("%s: voice labeling failed; will retry on next poll", row["task_id"])
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
