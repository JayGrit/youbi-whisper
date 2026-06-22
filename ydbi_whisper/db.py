from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import mysql.connector

from . import video_info
from .asr_segments import fix_asr_segment_rows
from .config import (
    DEVICE,
    MYSQL_CONFIG,
    WHISPER_ENGINE,
    WHISPER_MODEL,
    WHISPER_RUNTIME_DEVICE,
    WHISPERX_ALIGN,
    WHISPERX_ALIGN_INTERPOLATE_METHOD,
    WHISPERX_ALIGN_MODEL,
    WHISPERX_ALIGN_MODEL_DIR,
    WHISPERX_BATCH_SIZE,
    WHISPERX_CHUNK_SIZE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_REGROUP_MAX_CHARS,
    WHISPERX_REGROUP_MAX_DURATION_MS,
    WHISPERX_VAD_METHOD,
    WHISPERX_VAD_OFFSET,
    WHISPERX_VAD_ONSET,
)
from .service import FAILED, READY, RUNNING, SERVICE_NAME, SERVICE_TABLE, SUCCESS

HEARTBEAT_TABLE = "service_heartbeat"
SUBMISSION_TABLE = "downloader_submission"
UPLOADER_ACCOUNT_TABLE = "uploader_account"
UPLOAD_SUBMISSION_TABLES = (
    "uploader_task_bilibili",
    "uploader_task_douyin",
    "uploader_task_xiaohongshu",
    "uploader_task_shipinhao",
    "uploader_task_kuaishou",
    "uploader_task_jinritoutiao",
)
HEARTBEAT_DEVICE_COLUMNS = ("Macbook Air M4", "Macmini M2", "LPXB", "MY_HP", "LPXB_HP", "TXY")
OPERATOR_COLUMN = "operator"
OPERATOR_COLUMN_DEFINITION = "VARCHAR(128) NULL"
STAGE_RUNNING_TIMEOUT_SECONDS = 2 * 60 * 60
_heartbeat_schema_ready = False


def _row_value(row: Any, index: int = 0) -> Any:
    if isinstance(row, Mapping):
        return list(row.values())[index]
    return row[index]


def _service_table_for(stage_name: str) -> str:
    if stage_name != SERVICE_NAME:
        raise ValueError(f"{SERVICE_NAME} service cannot handle stage: {stage_name}")
    return SERVICE_TABLE


def _is_false(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "off"}
    return value is False or value == 0


def _staged_table_exists_cur(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """,
        (table,),
    )
    row = cur.fetchone()
    return bool(row and int(_row_value(row)) > 0)


def _staged_column_exists_cur(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    row = cur.fetchone()
    return bool(row and int(_row_value(row)) > 0)


def _ensure_staged_account_columns_cur(cur) -> bool:
    if not _staged_table_exists_cur(cur, UPLOADER_ACCOUNT_TABLE):
        return False
    if not _staged_column_exists_cur(cur, UPLOADER_ACCOUNT_TABLE, "staged_running_count"):
        cur.execute(
            f"""
            ALTER TABLE {UPLOADER_ACCOUNT_TABLE}
            ADD COLUMN staged_running_count INT NOT NULL DEFAULT 0
            """
        )
    if not _staged_column_exists_cur(cur, UPLOADER_ACCOUNT_TABLE, "staged_failed_count"):
        cur.execute(
            f"""
            ALTER TABLE {UPLOADER_ACCOUNT_TABLE}
            ADD COLUMN staged_failed_count INT NOT NULL DEFAULT 0
            """
        )
    return True


def _task_has_upload_submission_cur(cur, task_id: str, account_key: str) -> bool:
    if not task_id or not account_key:
        return False
    for table in UPLOAD_SUBMISSION_TABLES:
        if not _staged_table_exists_cur(cur, table):
            continue
        cur.execute(
            f"""
            SELECT 1
            FROM {table}
            WHERE task_id = %s AND account_key = %s
            LIMIT 1
            """,
            (task_id, account_key),
        )
        if cur.fetchone():
            return True
    return False


def _apply_staged_pipeline_failure_cur(cur, task_id: str, old_task_status: str | None) -> None:
    if str(old_task_status or "").strip().lower() == FAILED:
        return
    if not _ensure_staged_account_columns_cur(cur) or not _staged_table_exists_cur(cur, SUBMISSION_TABLE):
        return
    cur.execute(
        f"""
        SELECT type
        FROM {SUBMISSION_TABLE}
        WHERE task_id = %s
          AND status = %s
          AND NULLIF(type, '') IS NOT NULL
        FOR UPDATE
        """,
        (task_id, SUCCESS),
    )
    row = cur.fetchone()
    account_key = str(_row_value(row) if row else "").strip()
    if not account_key or _task_has_upload_submission_cur(cur, task_id, account_key):
        return
    cur.execute(
        f"""
        UPDATE {UPLOADER_ACCOUNT_TABLE}
        SET staged_running_count = GREATEST(staged_running_count - 1, 0),
            staged_failed_count = staged_failed_count + 1,
            metrics_updated_at = NOW(),
            updated_at = NOW()
        WHERE account_key = %s
        """,
        (account_key,),
    )


def _ensure_columns(cur, table: str, columns: Mapping[str, str]) -> None:
    return

def _table_columns(cur, table: str) -> set[str]:
    return set()

def _table_exists(cur, table: str) -> bool:
    return False

def _table_indexes(cur, table: str) -> set[str]:
    return set()

def _drop_index_if_exists(cur, table: str, index_name: str) -> None:
    return

def _drop_column_if_exists(cur, table: str, column_name: str) -> None:
    return

def _drop_table_if_exists(cur, table: str) -> None:
    return

def _migrate_asr_schema(cur) -> None:
    return

def ensure_asr_schema() -> None:
    return

def create_whisper_run(
    *,
    task_id: str,
    language: str,
    source_url: str | None,
    input_audio_url: str | None,
    input_local_path: str,
    input_file_size: int,
    input_sha256: str,
    sub_stage: str = "main",
) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO whisper_run
              (task_id, sub_stage, language, source_url, engine, model, configured_device, input_audio_url,
               input_local_path, input_file_size, input_sha256, batch_size, chunk_size,
               compute_type, vad_method, vad_onset, vad_offset, align_enabled,
               align_model, align_model_dir, align_interpolate_method, regroup_max_chars,
               regroup_max_duration_ms, status, operator)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
               %s, %s, %s, %s, %s, 'running', %s)
            """,
            (
                task_id,
                sub_stage,
                language,
                source_url,
                WHISPER_ENGINE,
                WHISPER_MODEL,
                WHISPER_RUNTIME_DEVICE,
                input_audio_url,
                input_local_path,
                input_file_size,
                input_sha256,
                WHISPERX_BATCH_SIZE if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_CHUNK_SIZE if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_COMPUTE_TYPE if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_VAD_METHOD if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_VAD_ONSET if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_VAD_OFFSET if WHISPER_ENGINE == "whisperx" else None,
                1 if WHISPER_ENGINE == "whisperx" and WHISPERX_ALIGN else 0,
                WHISPERX_ALIGN_MODEL or None,
                WHISPERX_ALIGN_MODEL_DIR if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_ALIGN_INTERPOLATE_METHOD if WHISPER_ENGINE == "whisperx" else None,
                WHISPERX_REGROUP_MAX_CHARS,
                WHISPERX_REGROUP_MAX_DURATION_MS,
                _operator_value(),
            ),
        )
        run_id = int(cur.lastrowid)
        conn.commit()
        return run_id


def update_whisper_run_runtime(
    run_id: int,
    *,
    runtime_device: str,
    model_path: str,
    input_duration_ms: int | None = None,
) -> None:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE whisper_run
            SET runtime_device = %s,
                model_path = %s,
                input_duration_ms = COALESCE(%s, input_duration_ms)
            WHERE id = %s
            """,
            (runtime_device, model_path, input_duration_ms, run_id),
        )
        conn.commit()


def finish_whisper_run(run_id: int, status: str, error_message: str | None = None) -> None:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE whisper_run
            SET status = %s, error_message = %s, finished_at = NOW()
            WHERE id = %s
            """,
            (status, error_message, run_id),
        )
        conn.commit()


def _seconds_to_ms(value: Any) -> int:
    return int(round(float(value or 0) * 1000))


def _segment_ms(segment: Mapping[str, Any], key: str) -> int:
    ms_key = f"{key}_time"
    if segment.get(ms_key) is not None:
        return int(segment.get(ms_key) or 0)
    return _seconds_to_ms(segment.get(key))


def _duration_ms(start_time: int, end_time: int) -> int:
    return max(0, int(end_time) - int(start_time))


def _word_global_index(word: Mapping[str, Any]) -> int | None:
    value = word.get("_whisper_word_global_index")
    return int(value) if value is not None else None


def _segment_word_ids(segment: Mapping[str, Any], word_ids: Mapping[int, int]) -> tuple[int | None, int | None]:
    indexes = [
        global_index
        for word in segment.get("words") or []
        if (global_index := _word_global_index(word)) is not None and global_index in word_ids
    ]
    if not indexes:
        return None, None
    return word_ids[min(indexes)], word_ids[max(indexes)]


def save_whisper_raw_segments(run_id: int, task_id: str, segments: list[dict[str, Any]]) -> dict[int, int]:
    ids: dict[int, int] = {}
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM whisper_raw_segment WHERE run_id = %s", (run_id,))
        for index, segment in enumerate(segments):
            start_time = _segment_ms(segment, "start")
            end_time = _segment_ms(segment, "end")
            cur.execute(
                """
                INSERT INTO whisper_raw_segment
                  (run_id, task_id, raw_index, text, start_time, end_time, duration_ms,
                   seek_offset, temperature, avg_logprob, compression_ratio, no_speech_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    task_id,
                    index,
                    str(segment.get("text") or "").strip(),
                    start_time,
                    end_time,
                    _duration_ms(start_time, end_time),
                    segment.get("seek"),
                    segment.get("temperature"),
                    segment.get("avg_logprob"),
                    segment.get("compression_ratio"),
                    segment.get("no_speech_prob"),
                ),
            )
            ids[index] = int(cur.lastrowid)
        conn.commit()
    return ids


def save_whisper_aligned_segments(
    run_id: int,
    task_id: str,
    segments: list[dict[str, Any]],
    raw_segment_ids: Mapping[int, int] | None = None,
) -> dict[int, int]:
    ids: dict[int, int] = {}
    raw_segment_ids = raw_segment_ids or {}
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM whisper_aligned_segment WHERE run_id = %s", (run_id,))
        for index, segment in enumerate(segments):
            start_time = _segment_ms(segment, "start")
            end_time = _segment_ms(segment, "end")
            cur.execute(
                """
                INSERT INTO whisper_aligned_segment
                  (run_id, task_id, aligned_index, raw_segment_id, text, start_time,
                   end_time, duration_ms, speaker)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    task_id,
                    index,
                    raw_segment_ids.get(index),
                    str(segment.get("text") or "").strip(),
                    start_time,
                    end_time,
                    _duration_ms(start_time, end_time),
                    str(segment.get("speaker") or "") or None,
                ),
            )
            ids[index] = int(cur.lastrowid)
        conn.commit()
    return ids


def save_whisper_aligned_words(
    run_id: int,
    task_id: str,
    segments: list[dict[str, Any]],
    aligned_segment_ids: Mapping[int, int],
) -> dict[int, int]:
    ids: dict[int, int] = {}
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM whisper_aligned_word WHERE run_id = %s", (run_id,))
        for segment_index, segment in enumerate(segments):
            aligned_segment_id = aligned_segment_ids.get(segment_index)
            for segment_word_index, word in enumerate(segment.get("words") or []):
                global_word_index = _word_global_index(word)
                if global_word_index is None:
                    continue
                text = str(word.get("word") or word.get("text") or "").strip()
                if not text:
                    continue
                start_time = _segment_ms(word, "start") if word.get("start") is not None or word.get("start_time") is not None else None
                end_time = _segment_ms(word, "end") if word.get("end") is not None or word.get("end_time") is not None else None
                cur.execute(
                    """
                    INSERT INTO whisper_aligned_word
                      (run_id, task_id, aligned_segment_id, global_word_index,
                       segment_word_index, text, start_time, end_time, duration_ms,
                       score, speaker)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        task_id,
                        aligned_segment_id,
                        global_word_index,
                        segment_word_index,
                        text[:255],
                        start_time,
                        end_time,
                        _duration_ms(start_time, end_time) if start_time is not None and end_time is not None else None,
                        word.get("score"),
                        str(word.get("speaker") or "") or None,
                    ),
                )
                ids[global_word_index] = int(cur.lastrowid)
        conn.commit()
    return ids


def save_whisper_pysbd_segments(
    run_id: int,
    task_id: str,
    segments: list[dict[str, Any]],
    word_ids: Mapping[int, int],
) -> dict[int, int]:
    ids: dict[int, int] = {}
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM whisper_pysbd_segment WHERE run_id = %s", (run_id,))
        for index, segment in enumerate(segments):
            start_time = _segment_ms(segment, "start")
            end_time = _segment_ms(segment, "end")
            first_word_id, last_word_id = _segment_word_ids(segment, word_ids)
            cur.execute(
                """
                INSERT INTO whisper_pysbd_segment
                  (run_id, task_id, pysbd_index, aligned_segment_id, text, start_time,
                   end_time, duration_ms, first_word_id, last_word_id, word_count,
                   pysbd_language, segmentation_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    task_id,
                    index,
                    segment.get("_whisper_aligned_segment_id"),
                    str(segment.get("text") or "").strip(),
                    start_time,
                    end_time,
                    _duration_ms(start_time, end_time),
                    first_word_id,
                    last_word_id,
                    len(segment.get("words") or []),
                    segment.get("_whisper_pysbd_language"),
                    segment.get("_whisper_segmentation_method") or "pysbd",
                ),
            )
            ids[index] = int(cur.lastrowid)
        conn.commit()
    return ids


def save_whisper_splits(
    run_id: int,
    task_id: str,
    segments: list[dict[str, Any]],
    pysbd_segment_ids: Mapping[int, int],
    word_ids: Mapping[int, int],
) -> dict[int, int]:
    ids: dict[int, int] = {}
    with connect() as conn:
        cur = conn.cursor()
        _ensure_columns(
            cur,
            "whisper_split",
            {
                "split_trigger": "VARCHAR(64) NULL",
                "split_method": "VARCHAR(64) NULL",
                "split_conjunction": "VARCHAR(64) NULL",
                "original_text": "MEDIUMTEXT NULL",
                "original_part_index": "INT NULL",
                "original_part_count": "INT NULL",
            },
        )
        cur.execute("DELETE FROM whisper_split WHERE run_id = %s", (run_id,))
        for index, segment in enumerate(segments):
            if not segment.get("_whisper_split_applied"):
                continue
            start_time = _segment_ms(segment, "start")
            end_time = _segment_ms(segment, "end")
            first_word_id, last_word_id = _segment_word_ids(segment, word_ids)
            source_index_value = segment.get("_whisper_pysbd_index")
            source_index = int(source_index_value) if source_index_value is not None else index
            cur.execute(
                """
                INSERT INTO whisper_split
                  (run_id, task_id, split_index, pysbd_segment_id, text, start_time,
                   end_time, duration_ms, first_word_id, last_word_id, word_count,
                   split_reason, split_at_word_index, split_punctuation, max_chars,
                   max_duration_ms, split_trigger, split_method, split_conjunction,
                   original_text, original_part_index, original_part_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    task_id,
                    index,
                    pysbd_segment_ids[source_index],
                    str(segment.get("text") or "").strip(),
                    start_time,
                    end_time,
                    _duration_ms(start_time, end_time),
                    first_word_id,
                    last_word_id,
                    len(segment.get("words") or []),
                    segment.get("_whisper_split_reason") or "none",
                    segment.get("_whisper_split_at_word_index"),
                    segment.get("_whisper_split_punctuation"),
                    WHISPERX_REGROUP_MAX_CHARS,
                    WHISPERX_REGROUP_MAX_DURATION_MS,
                    segment.get("_whisper_split_trigger") or segment.get("_whisper_split_reason") or "none",
                    segment.get("_whisper_split_method") or "unknown",
                    segment.get("_whisper_split_conjunction"),
                    segment.get("_whisper_original_text"),
                    segment.get("_whisper_original_part_index"),
                    segment.get("_whisper_original_part_count"),
                ),
            )
            ids[index] = int(cur.lastrowid)
        conn.commit()
    return ids


def _word_timestamp_rows(task_id: str, segments: list[dict[str, Any]], run_id: int | None = None) -> list[tuple]:
    def time_ms(word: dict[str, Any], ms_key: str, seconds_key: str) -> int:
        if word.get(ms_key) is not None:
            return int(word.get(ms_key) or 0)
        return int(round(float(word.get(seconds_key) or 0) * 1000))

    rows = []
    for segment_index, segment in enumerate(segments):
        for word_index, word in enumerate(segment.get("words") or []):
            text = str(word.get("text") or word.get("word") or "").strip()
            if not text:
                continue
            rows.append(
                (
                    task_id,
                    segment_index,
                    word_index,
                    text[:255],
                    time_ms(word, "start_time", "start"),
                    time_ms(word, "end_time", "end"),
                    run_id,
                    word.get("_whisper_aligned_word_id"),
                )
            )
    return rows


def save_word_timestamps(task_id: str, segments: list[dict[str, Any]], run_id: int | None = None) -> int:
    ensure_asr_schema()
    rows = _word_timestamp_rows(task_id, segments, run_id)
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM whisper_word_timestamp WHERE task_id = %s", (task_id,))
        if rows:
            cur.executemany(
                """
                INSERT INTO whisper_word_timestamp
                  (task_id, segment_index, word_index, text, start_time, end_time,
                   whisper_run_id, whisper_aligned_word_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  text = VALUES(text),
                  start_time = VALUES(start_time),
                  end_time = VALUES(end_time),
                  whisper_run_id = VALUES(whisper_run_id),
                  whisper_aligned_word_id = VALUES(whisper_aligned_word_id)
                """,
                rows,
            )
        conn.commit()
    return len(rows)


def save_asr_segments(
    task_id: str,
    segments: list[dict[str, Any]],
    run_id: int | None = None,
) -> None:
    ensure_asr_schema()
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM asr_segment WHERE task_id = %s", (task_id,))
        for index, item in enumerate(segments):
            cur.execute(
                """
                INSERT INTO asr_segment
                  (task_id, item_index, text, start_time, end_time, speaker,
                   whisper_run_id, whisper_split_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  text = VALUES(text),
                  start_time = VALUES(start_time),
                  end_time = VALUES(end_time),
                  speaker = VALUES(speaker),
                  whisper_run_id = VALUES(whisper_run_id),
                  whisper_split_id = VALUES(whisper_split_id)
                """,
                (
                    task_id,
                    index,
                    str(item.get("text") or "").strip(),
                    int(item.get("start_time") or 0),
                    int(item.get("end_time") or 0),
                    str(item.get("speaker") or "") or None,
                    run_id,
                    item.get("_whisper_split_id"),
                ),
            )
        conn.commit()


def save_asr_result(task_id: str, language: str, payload: dict[str, Any], run_id: int | None = None) -> list[dict[str, Any]]:
    audio_info = payload.get("audio_info") or {}
    result = payload.get("result") or {}
    duration_ms = int(audio_info.get("duration") or 0)
    segments = fix_asr_segment_rows(result.get("utterances") or [], duration_ms)
    save_asr_segments(task_id, segments, run_id)
    save_word_timestamps(task_id, segments, run_id)
    return segments


def connect():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    return conn


def _dict_cursor(conn):
    return conn.cursor(dictionary=True)


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _heartbeat_device_column() -> str | None:
    return DEVICE if DEVICE in HEARTBEAT_DEVICE_COLUMNS else None


def _operator_value() -> str:
    return DEVICE


def current_operator() -> str:
    return _operator_value()


def _ensure_operator_columns(cur, tables: tuple[str, ...]) -> None:
    return


def ensure_service_heartbeat_schema() -> None:
    global _heartbeat_schema_ready
    _heartbeat_schema_ready = True


def record_service_poll() -> None:
    column = _heartbeat_device_column()
    if not column:
        return

    ensure_service_heartbeat_schema()
    quoted_column = _quote_identifier(column)
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {HEARTBEAT_TABLE} (service_name, {quoted_column})
            VALUES (%s, NOW())
            ON DUPLICATE KEY UPDATE {quoted_column} = VALUES({quoted_column})
            """,
            ("whisper",),
        )
        conn.commit()


def get_task(task_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        cur = _dict_cursor(conn)
        cur.execute("SELECT * FROM task WHERE id = %s", (task_id,))
        task = cur.fetchone()
        if not task:
            return None
        info = video_info.get(task_id) or {}
        task["video_info"] = info
        for key, value in info.items():
            if value is not None:
                task[key] = value
        return task


def list_narration_alignment_segments(task_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        cur = _dict_cursor(conn)
        cur.execute(
            """
            SELECT ss.item_index, ss.status, ss.dst_text AS text,
                   cvs.start_time, cvs.end_time
            FROM speaker_segment ss
            LEFT JOIN combiner_vocal_speed cvs
              ON cvs.task_id = ss.task_id
             AND cvs.segment_id = ss.item_index
            WHERE ss.task_id = %s
            ORDER BY ss.item_index ASC
            """,
            (task_id,),
        )
        return list(cur.fetchall())


def demucs_operator_for(task_id: str) -> str | None:
    with connect() as conn:
        cur = _dict_cursor(conn)
        _ensure_operator_columns(cur, ("demucs",))
        cur.execute("SELECT `operator` FROM demucs WHERE task_id = %s", (task_id,))
        row = cur.fetchone()
        if not row:
            return None
        operator = row.get("operator")
        return str(operator).strip() if operator else None


def find_ready(stage_name: str) -> dict[str, Any] | None:
    table = _service_table_for(stage_name)
    with connect() as conn:
        cur = _dict_cursor(conn)
        cur.execute(
            f"""
            SELECT s.*
            FROM {table} s
            JOIN task t ON t.id = s.task_id
            WHERE s.status = %s
              AND t.status <> 'failed'
            ORDER BY s.task_id ASC, s.sub_stage ASC
            LIMIT 1
            """,
            (READY,),
        )
        return video_info.merge_into(cur.fetchone())


def mark_running(stage_name: str, task_id: str, sub_stage: str = "main") -> bool:
    table = _service_table_for(stage_name)
    operator = _operator_value()
    with connect() as conn:
        cur = conn.cursor()
        _ensure_operator_columns(cur, (table,))
        cur.execute(
            f"""
            UPDATE {table}
            SET status = %s,
                started_at = COALESCE(started_at, NOW()),
                error_message = NULL,
                `operator` = %s
            WHERE task_id = %s AND sub_stage = %s AND status = %s
              AND EXISTS (
                  SELECT 1 FROM task t
                  WHERE t.id = %s AND t.status <> 'failed'
              )
            """,
            (RUNNING, operator, task_id, sub_stage, READY, task_id),
        )
        stage_updated = cur.rowcount == 1
        if stage_updated:
            cur.execute(
                """
                UPDATE task
                SET status = 'running',
                    current_stage = %s,
                    started_at = COALESCE(started_at, NOW())
                WHERE id = %s
                """,
                (stage_name, task_id),
            )
        conn.commit()
        return stage_updated


def recycle_stale_running(stage_name: str) -> int:
    table = _service_table_for(stage_name)
    timeout_seconds = STAGE_RUNNING_TIMEOUT_SECONDS
    message = f"{stage_name} task timed out after {timeout_seconds}s; retrying"
    with connect() as conn:
        cur = conn.cursor()
        _ensure_operator_columns(cur, (table,))
        cur.execute(
            f"""
            UPDATE {table}
            SET status = %s,
                started_at = NULL,
                completed_at = NULL,
                error_message = %s,
                `operator` = NULL
            WHERE status = %s
              AND started_at IS NOT NULL
              AND TIMESTAMPDIFF(SECOND, started_at, NOW()) > %s
            """,
            (READY, message, RUNNING, timeout_seconds),
        )
        recycled = cur.rowcount
        conn.commit()
        return int(recycled)


def _update_stage_fields(stage_name: str, task_id: str, fields: Mapping[str, Any]) -> None:
    table = _service_table_for(stage_name)
    assignments = ", ".join(f"{key} = %s" for key in fields)
    values = list(fields.values()) + [task_id]
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE {table} SET {assignments} WHERE task_id = %s", values)
        conn.commit()


def set_translator_asr_json_path(task_id: str, asr_json_path: str) -> None:
    video_info.upsert(task_id, {"asr_json_path": asr_json_path})


def mark_success(stage_name: str, task_id: str, outputs: Mapping[str, Any] | None = None, sub_stage: str = "main") -> None:
    table = _service_table_for(stage_name)
    fields = dict(outputs or {})
    stage_fields = {key: value for key, value in fields.items() if key not in video_info.COLUMNS}
    assignments = ["status = %s", "completed_at = NOW()", "error_message = NULL"]
    values: list[Any] = [SUCCESS]
    for key, value in stage_fields.items():
        assignments.append(f"{key} = %s")
        values.append(value)
    values.extend((task_id, sub_stage))

    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM task WHERE id = %s", (task_id,))
        task_row = cur.fetchone()
        if not task_row:
            conn.commit()
            return
        # A different pipeline stage may fail while Whisper is still running.
        # Keep the task-level failure, but persist Whisper's own successful
        # result and output references so the stage does not remain running.
        video_info.upsert(task_id, fields, cur)
        cur.execute(
            f"UPDATE {table} SET {', '.join(assignments)} WHERE task_id = %s AND sub_stage = %s",
            values,
        )
        conn.commit()


def mark_failed(stage_name: str, task_id: str, message: str, sub_stage: str = "main") -> None:
    table = _service_table_for(stage_name)
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM task WHERE id = %s FOR UPDATE", (task_id,))
        task_row = cur.fetchone()
        old_task_status = _row_value(task_row) if task_row else None
        cur.execute(
            f"""
            UPDATE {table}
            SET status = %s, error_message = %s, completed_at = NOW()
            WHERE task_id = %s AND sub_stage = %s
            """,
            (FAILED, message, task_id, sub_stage),
        )
        cur.execute(
            """
            UPDATE task
            SET status = 'failed', current_stage = %s, error_message = %s, completed_at = NOW()
            WHERE id = %s
            """,
            (stage_name, message, task_id),
        )
        _apply_staged_pipeline_failure_cur(cur, task_id, old_task_status)
        conn.commit()
