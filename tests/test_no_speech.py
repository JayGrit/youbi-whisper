from __future__ import annotations

import unittest
from unittest.mock import patch

from ydbi_whisper import db, whisper_asr


class _Cursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple | list | None]] = []

    def execute(self, sql: str, params=None) -> None:
        self.statements.append((" ".join(sql.split()), params))

    def fetchone(self):
        return ("running",)


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


class NoSpeechPipelineTest(unittest.TestCase):
    def test_whisper_success_skips_subtitle_stages_and_readies_uploader(self) -> None:
        connection = _Connection()
        with (
            patch.object(db, "connect", return_value=connection),
            patch.object(db.video_info, "upsert") as upsert,
        ):
            db.mark_success(
                "whisper",
                "task-1",
                {"need_subtitle": 0, "need_dubbing": 0},
            )

        sql = "\n".join(statement for statement, _ in connection.cursor_instance.statements)
        self.assertIn("UPDATE translator", sql)
        self.assertIn("UPDATE speaker", sql)
        self.assertIn("UPDATE combiner", sql)
        self.assertIn("UPDATE uploader", sql)
        self.assertIn("current_stage = 'uploader'", sql)
        self.assertNotIn("UPDATE translator SET status = %s WHERE task_id = %s AND status = 'pending'", sql)
        self.assertTrue(connection.committed)
        upsert.assert_called_once_with(
            "task-1",
            {"need_subtitle": 0, "need_dubbing": 0},
            connection.cursor_instance,
        )

    def test_unrelated_index_error_is_not_treated_as_empty_vad_batch(self) -> None:
        try:
            [][0]
        except IndexError as exc:
            self.assertFalse(whisper_asr._is_empty_whisperx_batch_error(exc))


if __name__ == "__main__":
    unittest.main()
