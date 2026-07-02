from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import mysql.connector  # noqa: F401
except ModuleNotFoundError:
    mysql_module = types.ModuleType("mysql")
    connector_module = types.ModuleType("mysql.connector")
    mysql_module.connector = connector_module
    sys.modules["mysql"] = mysql_module
    sys.modules["mysql.connector"] = connector_module

try:
    import minio  # noqa: F401
except ModuleNotFoundError:
    minio_module = types.ModuleType("minio")
    minio_error_module = types.ModuleType("minio.error")

    class _Minio:
        pass

    class _S3Error(Exception):
        code = "stub"

    minio_module.Minio = _Minio
    minio_error_module.S3Error = _S3Error
    sys.modules["minio"] = minio_module
    sys.modules["minio.error"] = minio_error_module

from ydbi_whisper import main


class DubbingChunkAlignedWhisperTest(unittest.TestCase):
    def test_dubbing_alignment_uses_dubbing_audio_and_saves_asr_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            vocals = session / "audio_dubbing.wav"
            vocals.parent.mkdir(parents=True, exist_ok=True)
            vocals.write_bytes(b"audio")
            aligned_payload = {
                "audio_info": {"duration": 1200},
                "result": {"text": "你好", "utterances": [{"text": "你好", "start_time": 250, "end_time": 900}]},
            }
            saved_segments = [{"text": "你好", "start_time": 250, "end_time": 900, "words": []}]
            row = {
                "task_id": "task-1",
                "task_type": main.DUBBING_CHUNK_ALIGNED_TASK_TYPE,
                "sub_stage": "dubbing_alignment",
                "audio_dubbing_url": "minio://audio_dubbing.wav",
            }
            with (
                patch("ydbi_whisper.main.task_work_dir", return_value=session),
                patch("ydbi_whisper.main.download", return_value=vocals) as download,
                patch("ydbi_whisper.main.db.get_task", return_value={"source_url": ""}),
                patch("ydbi_whisper.main.db.create_whisper_run", return_value=7),
                patch(
                    "ydbi_whisper.main.db.list_dubbing_alignment_segments",
                    return_value=[{"item_index": 0, "text": "你好", "start_time": 0, "end_time": 1200}],
                ),
                patch("ydbi_whisper.main.align_known_text", return_value=aligned_payload),
                patch("ydbi_whisper.main.db.save_asr_result", return_value=saved_segments) as save_asr,
                patch("ydbi_whisper.main.db.save_dubbing_alignment_result") as save_old_alignment,
                patch("ydbi_whisper.main.db.finish_whisper_run"),
            ):
                outputs = main.handle(row)

        download.assert_called_once()
        save_asr.assert_called_once_with("task-1", "zh", aligned_payload, run_id=7)
        save_old_alignment.assert_not_called()
        self.assertEqual({"asr_json_path": "db://whisper_asr_segment/task-1"}, outputs)


if __name__ == "__main__":
    unittest.main()
