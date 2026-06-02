from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mysql.connector

from . import video_info
from .asr_segments import fix_asr_segment_rows
from .config import MYSQL_CONFIG
from .stages import FAILED, READY, RUNNING, SUCCESS, stage_for

HEARTBEAT_TABLE = "yd_service_heartbeat"
HEARTBEAT_DEVICE_COLUMNS = ("Macbook Air M4", "Macmini M2", "LPXB", "MY_HP", "LPXB_HP", "TXY")
OPERATOR_COLUMN = "operator"
OPERATOR_COLUMN_DEFINITION = "VARCHAR(128) NULL"
STAGE_RUNNING_TIMEOUT_SECONDS = 2 * 60 * 60
_heartbeat_schema_ready = False


def _row_value(row: Any, index: int = 0) -> Any:
    if isinstance(row, Mapping):
        return list(row.values())[index]
    return row[index]


def _ensure_columns(cur, table: str, columns: Mapping[str, str]) -> None:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        (table,),
    )
    existing = {_row_value(row) for row in cur.fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _table_columns(cur, table: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        (table,),
    )
    return {_row_value(row) for row in cur.fetchall()}


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        (table,),
    )
    return int(_row_value(cur.fetchone()) or 0) > 0


def _table_indexes(cur, table: str) -> set[str]:
    cur.execute(
        """
        SELECT DISTINCT index_name
        FROM information_schema.statistics
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        (table,),
    )
    return {_row_value(row) for row in cur.fetchall()}


def _drop_index_if_exists(cur, table: str, index_name: str) -> None:
    if index_name in _table_indexes(cur, table):
        cur.execute(f"ALTER TABLE {table} DROP INDEX {index_name}")


def _drop_column_if_exists(cur, table: str, column_name: str) -> None:
    if column_name in _table_columns(cur, table):
        cur.execute(f"ALTER TABLE {table} DROP COLUMN {column_name}")


def _drop_table_if_exists(cur, table: str) -> None:
    if _table_exists(cur, table):
        cur.execute(f"DROP TABLE {table}")


def _migrate_asr_schema(cur) -> None:
    if "segment_type" in _table_columns(cur, "yd_asr_segment"):
        cur.execute(
            """
            DELETE old_segment
            FROM yd_asr_segment old_segment
            JOIN yd_asr_segment fixed_segment
              ON fixed_segment.task_id = old_segment.task_id
             AND fixed_segment.item_index = old_segment.item_index
             AND fixed_segment.segment_type = 'fixed'
            WHERE old_segment.segment_type <> 'fixed'
            """
        )
        cur.execute(
            """
            DELETE duplicate_segment
            FROM yd_asr_segment duplicate_segment
            JOIN yd_asr_segment keep_segment
              ON keep_segment.task_id = duplicate_segment.task_id
             AND keep_segment.item_index = duplicate_segment.item_index
             AND keep_segment.id < duplicate_segment.id
            """
        )
    _drop_index_if_exists(cur, "yd_asr_segment", "uk_asr_segment")
    _drop_index_if_exists(cur, "yd_asr_segment", "idx_asr_segment_task")
    _drop_column_if_exists(cur, "yd_asr_segment", "segment_type")
    _drop_column_if_exists(cur, "yd_asr_segment", "words_json")
    if "uk_asr_segment" not in _table_indexes(cur, "yd_asr_segment"):
        cur.execute("ALTER TABLE yd_asr_segment ADD UNIQUE KEY uk_asr_segment (task_id, item_index)")
    if "idx_asr_segment_task" not in _table_indexes(cur, "yd_asr_segment"):
        cur.execute("ALTER TABLE yd_asr_segment ADD KEY idx_asr_segment_task (task_id, item_index)")

    if "segment_type" in _table_columns(cur, "whisper_word_timestamp"):
        cur.execute("DELETE FROM whisper_word_timestamp WHERE segment_type <> 'raw'")
    _drop_index_if_exists(cur, "whisper_word_timestamp", "uk_whisper_word")
    _drop_index_if_exists(cur, "whisper_word_timestamp", "idx_whisper_word_task")
    _drop_column_if_exists(cur, "whisper_word_timestamp", "segment_type")
    if "uk_whisper_word" not in _table_indexes(cur, "whisper_word_timestamp"):
        cur.execute("ALTER TABLE whisper_word_timestamp ADD UNIQUE KEY uk_whisper_word (task_id, segment_index, word_index)")
    if "idx_whisper_word_task" not in _table_indexes(cur, "whisper_word_timestamp"):
        cur.execute("ALTER TABLE whisper_word_timestamp ADD KEY idx_whisper_word_task (task_id, start_time, end_time)")


def ensure_asr_schema() -> None:
    with connect() as conn:
        cur = conn.cursor()
        _drop_table_if_exists(cur, "yd_asr_result")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS yd_asr_segment (
              id BIGINT PRIMARY KEY AUTO_INCREMENT,
              task_id VARCHAR(64) NOT NULL,
              item_index INT NOT NULL,
              text MEDIUMTEXT NOT NULL,
              start_time INT NOT NULL,
              end_time INT NOT NULL,
              speaker VARCHAR(64),
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              UNIQUE KEY uk_asr_segment (task_id, item_index),
              KEY idx_asr_segment_task (task_id, item_index)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS whisper_word_timestamp (
              id BIGINT PRIMARY KEY AUTO_INCREMENT,
              task_id VARCHAR(64) NOT NULL,
              segment_index INT NOT NULL,
              word_index INT NOT NULL,
              text VARCHAR(255) NOT NULL,
              start_time INT NOT NULL,
              end_time INT NOT NULL,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              UNIQUE KEY uk_whisper_word (task_id, segment_index, word_index),
              KEY idx_whisper_word_task (task_id, start_time, end_time)
            )
            """
        )
        _ensure_columns(
            cur,
            "yd_asr_segment",
            {
                "speaker": "VARCHAR(64)",
            },
        )
        _migrate_asr_schema(cur)
        conn.commit()


def _word_timestamp_rows(task_id: str, segments: list[dict[str, Any]]) -> list[tuple]:
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
                )
            )
    return rows


def save_word_timestamps(task_id: str, segments: list[dict[str, Any]]) -> int:
    ensure_asr_schema()
    rows = _word_timestamp_rows(task_id, segments)
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM whisper_word_timestamp WHERE task_id = %s", (task_id,))
        if rows:
            cur.executemany(
                """
                INSERT INTO whisper_word_timestamp
                  (task_id, segment_index, word_index, text, start_time, end_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  text = VALUES(text),
                  start_time = VALUES(start_time),
                  end_time = VALUES(end_time)
                """,
                rows,
            )
        conn.commit()
    return len(rows)


def save_asr_segments(
    task_id: str,
    segments: list[dict[str, Any]],
) -> None:
    ensure_asr_schema()
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM yd_asr_segment WHERE task_id = %s", (task_id,))
        for index, item in enumerate(segments):
            cur.execute(
                """
                INSERT INTO yd_asr_segment
                  (task_id, item_index, text, start_time, end_time, speaker)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  text = VALUES(text),
                  start_time = VALUES(start_time),
                  end_time = VALUES(end_time),
                  speaker = VALUES(speaker)
                """,
                (
                    task_id,
                    index,
                    str(item.get("text") or "").strip(),
                    int(item.get("start_time") or 0),
                    int(item.get("end_time") or 0),
                    str(item.get("speaker") or "") or None,
                ),
            )
        conn.commit()


def save_asr_result(task_id: str, language: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    audio_info = payload.get("audio_info") or {}
    result = payload.get("result") or {}
    duration_ms = int(audio_info.get("duration") or 0)
    segments = fix_asr_segment_rows(result.get("utterances") or [], duration_ms)
    save_asr_segments(task_id, segments)
    save_word_timestamps(task_id, segments)
    return segments


def connect():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    return conn


def _dict_cursor(conn):
    return conn.cursor(dictionary=True)


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _heartbeat_device_column() -> str | None:
    device = os.environ.get("DEVICE", "").strip() or "Macbook Air M4"
    return device if device in HEARTBEAT_DEVICE_COLUMNS else None


def _operator_value() -> str:
    return os.environ.get("DEVICE", "").strip() or "Macbook Air M4"


def current_operator() -> str:
    return _operator_value()


def _ensure_operator_columns(cur, tables: tuple[str, ...]) -> None:
    for table in tables:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
            """,
            (table,),
        )
        if _row_value(cur.fetchone()) == 0:
            continue

        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
            """,
            (table, OPERATOR_COLUMN),
        )
        if _row_value(cur.fetchone()) > 0:
            continue

        try:
            cur.execute(
                f"ALTER TABLE {_quote_identifier(table)} "
                f"ADD COLUMN {_quote_identifier(OPERATOR_COLUMN)} {OPERATOR_COLUMN_DEFINITION}"
            )
        except mysql.connector.Error as exc:
            if getattr(exc, "errno", None) != 1060:
                raise


def ensure_service_heartbeat_schema() -> None:
    global _heartbeat_schema_ready
    if _heartbeat_schema_ready:
        return

    columns_sql = ",\n                ".join(
        f"{_quote_identifier(column)} DATETIME NULL" for column in HEARTBEAT_DEVICE_COLUMNS
    )
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {HEARTBEAT_TABLE} (
                service_name VARCHAR(64) NOT NULL PRIMARY KEY,
                {columns_sql},
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
            """,
            (HEARTBEAT_TABLE,),
        )
        existing = {_row_value(row) for row in cur.fetchall()}
        for column in HEARTBEAT_DEVICE_COLUMNS:
            if column not in existing:
                try:
                    cur.execute(f"ALTER TABLE {HEARTBEAT_TABLE} ADD COLUMN {_quote_identifier(column)} DATETIME NULL")
                except mysql.connector.Error as exc:
                    if getattr(exc, "errno", None) != 1060:
                        raise
        conn.commit()
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
        cur.execute("SELECT * FROM yd_task WHERE id = %s", (task_id,))
        task = cur.fetchone()
        if not task:
            return None
        task["video_info"] = video_info.get(task_id)
        return task


def demucs_operator_for(task_id: str) -> str | None:
    with connect() as conn:
        cur = _dict_cursor(conn)
        _ensure_operator_columns(cur, ("yd_demucs",))
        cur.execute("SELECT `operator` FROM yd_demucs WHERE task_id = %s", (task_id,))
        row = cur.fetchone()
        if not row:
            return None
        operator = row.get("operator")
        return str(operator).strip() if operator else None


def find_ready(stage_name: str) -> dict[str, Any] | None:
    stage = stage_for(stage_name)
    with connect() as conn:
        cur = _dict_cursor(conn)
        cur.execute(
            f"""
            SELECT s.*
            FROM {stage.table} s
            JOIN yd_task t ON t.id = s.task_id
            WHERE s.status = %s
              AND t.status <> 'failed'
            ORDER BY s.task_id ASC
            LIMIT 1
            """,
            (READY,),
        )
        return video_info.merge_into(cur.fetchone())


def mark_running(stage_name: str, task_id: str) -> bool:
    stage = stage_for(stage_name)
    operator = _operator_value()
    with connect() as conn:
        cur = conn.cursor()
        _ensure_operator_columns(cur, ("yd_task", stage.table))
        cur.execute(
            f"""
            UPDATE {stage.table}
            SET status = %s,
                started_at = COALESCE(started_at, NOW()),
                error_message = NULL,
                `operator` = %s
            WHERE task_id = %s AND status = %s
              AND EXISTS (
                  SELECT 1 FROM yd_task t
                  WHERE t.id = %s AND t.status <> 'failed'
              )
            """,
            (RUNNING, operator, task_id, READY, task_id),
        )
        stage_updated = cur.rowcount == 1
        if stage_updated:
            cur.execute(
                """
                UPDATE yd_task
                SET status = 'running',
                    current_stage = %s,
                    started_at = COALESCE(started_at, NOW()),
                    `operator` = %s
                WHERE id = %s
                """,
                (stage_name, operator, task_id),
            )
        conn.commit()
        return stage_updated


def recycle_stale_running(stage_name: str) -> int:
    stage = stage_for(stage_name)
    timeout_seconds = STAGE_RUNNING_TIMEOUT_SECONDS
    message = f"{stage_name} task timed out after {timeout_seconds}s; retrying"
    with connect() as conn:
        cur = conn.cursor()
        _ensure_operator_columns(cur, (stage.table,))
        cur.execute(
            f"""
            UPDATE {stage.table}
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
    stage = stage_for(stage_name)
    assignments = ", ".join(f"{key} = %s" for key in fields)
    values = list(fields.values()) + [task_id]
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE {stage.table} SET {assignments} WHERE task_id = %s", values)
        conn.commit()


def set_translator_asr_json_path(task_id: str, asr_json_path: str) -> None:
    video_info.upsert(task_id, {"asr_json_path": asr_json_path})


def session_path_for(task_id: str) -> Path:
    task = get_task(task_id)
    if not task:
        raise RuntimeError(f"Task not found: {task_id}")
    info = task.get("video_info") or {}
    session_path = info.get("session_path")
    if not session_path:
        raise RuntimeError(f"Task missing downloader session_path: {task_id}")
    return Path(session_path)


def mark_success(stage_name: str, task_id: str, outputs: Mapping[str, Any] | None = None) -> None:
    stage = stage_for(stage_name)
    fields = dict(outputs or {})
    stage_fields = {key: value for key, value in fields.items() if key not in video_info.COLUMNS}
    assignments = ["status = %s", "completed_at = NOW()", "error_message = NULL"]
    values: list[Any] = [SUCCESS]
    for key, value in stage_fields.items():
        assignments.append(f"{key} = %s")
        values.append(value)
    values.append(task_id)

    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM yd_task WHERE id = %s", (task_id,))
        task_row = cur.fetchone()
        if not task_row or task_row[0] == "failed":
            conn.commit()
            return
        video_info.upsert(task_id, fields, cur)
        cur.execute(
            f"UPDATE {stage.table} SET {', '.join(assignments)} WHERE task_id = %s",
            values,
        )
        if stage.next_table:
            cur.execute(
                f"UPDATE {stage.next_table} SET status = %s WHERE task_id = %s AND status = 'pending'",
                (READY, task_id),
            )
            cur.execute(
                "UPDATE yd_task SET current_stage = %s WHERE id = %s",
                (stage.next_name, task_id),
            )
        else:
            cur.execute(
                """
                UPDATE yd_task
                SET status = 'success', current_stage = 'done', completed_at = NOW(), error_message = NULL
                WHERE id = %s
                """,
                (task_id,),
            )
        conn.commit()


def mark_failed(stage_name: str, task_id: str, message: str) -> None:
    stage = stage_for(stage_name)
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            UPDATE {stage.table}
            SET status = %s, error_message = %s, completed_at = NOW()
            WHERE task_id = %s
            """,
            (FAILED, message, task_id),
        )
        cur.execute(
            """
            UPDATE yd_task
            SET status = 'failed', current_stage = %s, error_message = %s, completed_at = NOW()
            WHERE id = %s
            """,
            (stage_name, message, task_id),
        )
        conn.commit()
