from __future__ import annotations

import unittest
from unittest.mock import patch

from ydbi_whisper import db


class _Cursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str, _params=None) -> None:
        self.statements.append(" ".join(sql.split()))

    def fetchone(self):
        return ("failed",)


class _Connection:
    def __init__(self) -> None:
        self.cursor_instance = _Cursor()
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def cursor(self) -> _Cursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True


class FailedTaskCompletionTest(unittest.TestCase):
    def test_whisper_success_is_persisted_after_another_stage_failed_task(self) -> None:
        connection = _Connection()
        outputs = {"asr_json_path": "db://asr_segment/task-1"}
        with (
            patch.object(db, "connect", return_value=connection),
            patch.object(db.video_info, "upsert") as upsert,
        ):
            db.mark_success("whisper", "task-1", outputs)

        sql = "\n".join(connection.cursor_instance.statements)
        self.assertIn("UPDATE whisper SET status = %s", sql)
        self.assertTrue(connection.committed)
        upsert.assert_called_once_with("task-1", outputs, connection.cursor_instance)


if __name__ == "__main__":
    unittest.main()
