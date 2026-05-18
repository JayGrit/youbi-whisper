from __future__ import annotations

import os
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mysql.connector

from . import video_info
from .asr_segments import normalize_asr_segments
from .config import MYSQL_CONFIG
from .stages import FAILED, READY, RUNNING, SUCCESS, stage_for

HEARTBEAT_TABLE = "yd_service_heartbeat"
HEARTBEAT_DEVICE_COLUMNS = ("Macbook Air M4", "Macmini M2", "LPXB", "MY_HP", "LPXB_HP", "TXY")
OPERATOR_COLUMN = "operator"
OPERATOR_COLUMN_DEFINITION = "VARCHAR(128) NULL"
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


def ensure_asr_schema() -> None:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS yd_asr_result (
              task_id VARCHAR(64) PRIMARY KEY,
              language VARCHAR(16),
              duration_ms INT NOT NULL DEFAULT 0,
              full_text MEDIUMTEXT,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS yd_asr_segment (
              id BIGINT PRIMARY KEY AUTO_INCREMENT,
              task_id VARCHAR(64) NOT NULL,
              segment_type VARCHAR(16) NOT NULL,
              item_index INT NOT NULL,
              text MEDIUMTEXT NOT NULL,
              start_time INT NOT NULL,
              end_time INT NOT NULL,
              speaker VARCHAR(64),
              words_json MEDIUMTEXT,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              UNIQUE KEY uk_asr_segment (task_id, segment_type, item_index),
              KEY idx_asr_segment_task (task_id, segment_type, item_index)
            )
            """
        )
        _ensure_columns(
            cur,
            "yd_asr_segment",
            {
                "speaker": "VARCHAR(64)",
                "words_json": "MEDIUMTEXT",
            },
        )
        conn.commit()


def save_asr_segments(
    task_id: str,
    segment_type: str,
    segments: list[dict[str, Any]],
    *,
    language: str | None = None,
    duration_ms: int | None = None,
    full_text: str | None = None,
) -> None:
    ensure_asr_schema()
    with connect() as conn:
        cur = conn.cursor()
        if language is not None or duration_ms is not None or full_text is not None:
            cur.execute(
                """
                INSERT INTO yd_asr_result (task_id, language, duration_ms, full_text)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  language = COALESCE(VALUES(language), language),
                  duration_ms = IF(VALUES(duration_ms) = 0, duration_ms, VALUES(duration_ms)),
                  full_text = COALESCE(VALUES(full_text), full_text)
                """,
                (task_id, language, int(duration_ms or 0), full_text),
            )
        cur.execute(
            "DELETE FROM yd_asr_segment WHERE task_id = %s AND segment_type = %s",
            (task_id, segment_type),
        )
        for index, item in enumerate(segments):
            words = item.get("words")
            words_json = item.get("words_json")
            if words_json is None and words is not None:
                words_json = json.dumps(words, ensure_ascii=False)
            cur.execute(
                """
                INSERT INTO yd_asr_segment
                  (task_id, segment_type, item_index, text, start_time, end_time, speaker, words_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  text = VALUES(text),
                  start_time = VALUES(start_time),
                  end_time = VALUES(end_time),
                  speaker = VALUES(speaker),
                  words_json = VALUES(words_json)
                """,
                (
                    task_id,
                    segment_type,
                    index,
                    str(item.get("text") or "").strip(),
                    int(item.get("start_time") or 0),
                    int(item.get("end_time") or 0),
                    str(item.get("speaker") or "") or None,
                    words_json,
                ),
            )
        conn.commit()


def save_asr_result(task_id: str, language: str, payload: dict[str, Any], segment_type: str = "raw") -> list[dict[str, Any]]:
    audio_info = payload.get("audio_info") or {}
    result = payload.get("result") or {}
    duration_ms = int(audio_info.get("duration") or 0)
    full_text = str(result.get("text") or "")
    segments = normalize_asr_segments(result.get("utterances") or [])
    save_asr_segments(
        task_id,
        segment_type,
        segments,
        language=language,
        duration_ms=duration_ms,
        full_text=full_text,
    )
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


def _stage_running_timeout_seconds() -> int:
    value = os.environ.get("YDBI_STAGE_RUNNING_TIMEOUT_SECONDS", "").strip()
    if not value:
        return 14400
    try:
        return max(1, int(value))
    except ValueError:
        return 14400


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


def record_service_poll(stage_name: str) -> None:
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
            (stage_name,),
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
            f"SELECT * FROM {stage.table} WHERE status = %s ORDER BY task_id ASC LIMIT 1",
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
            """,
            (RUNNING, operator, task_id, READY),
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
    timeout_seconds = _stage_running_timeout_seconds()
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
