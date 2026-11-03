#!/usr/bin/env python3
"""Re-label podcast-23 subtitles using its locally supplied NotebookLM audio."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ydbi_whisper import db
from ydbi_whisper.podcast_voice_gender import identify_two_voices

AUDIO = Path("/Users/hoshuuch/Downloads/Why_High_IQ_Won_t_Save_Your_Finances.m4a")
TASK_ID = "podcast-23"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not AUDIO.is_file():
        raise FileNotFoundError(f"podcast-23 local audio not found: {AUDIO}")
    with db.connect() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT item_index, text, start_time, end_time, speaker "
            "FROM whisper_asr_segment WHERE task_id=%s ORDER BY item_index",
            (TASK_ID,),
        )
        segments = cursor.fetchall()
    if not segments:
        raise RuntimeError(f"No Whisper subtitles found for {TASK_ID}")
    labeled, summary = identify_two_voices(AUDIO, segments)
    with db.connect() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE whisper_asr_segment SET speaker=%s WHERE task_id=%s AND item_index=%s",
            [(item["speaker"], TASK_ID, item["item_index"]) for item in labeled],
        )
        conn.commit()
    logging.info("Updated %d podcast-23 subtitles: %s", len(labeled), summary)


if __name__ == "__main__":
    main()
