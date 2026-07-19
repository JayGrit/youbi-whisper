from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ydbi_whisper import main
from ydbi_whisper.whisper_asr import NoSpeechDetected


class SourceTranscriptionTest(unittest.TestCase):
    def test_source_transcription_uploads_txt_without_writing_asr_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.m4a"
            audio.write_bytes(b"audio")
            with (
                mock.patch.object(main, "task_work_dir", return_value=Path(tmp)),
                mock.patch.object(main, "_vocals_input_for", return_value=audio),
                mock.patch.object(main.db, "get_task", return_value={"source_url": "https://youtu.be/dQw4w9WgXcQ"}),
                mock.patch.object(main.db, "create_whisper_run", return_value=7),
                mock.patch.object(main, "recognize_speech", return_value={
                    "audio_info": {"duration": 2000},
                    "result": {"text": "hello world", "utterances": [{"text": "hello world", "start_time": 0, "end_time": 2000}]},
                }),
                mock.patch.object(main, "upload", return_value="https://minio/transcript.txt") as upload,
                mock.patch.object(main.db, "save_asr_result") as save_asr,
                mock.patch.object(main.db, "finish_whisper_run"),
            ):
                result = main.handle({"task_id": "task-1", "sub_stage": "source_transcription", "task_type": "narration", "audio_source_url": "minio://audio"})

        self.assertEqual(result, {"source_transcript_txt_url": "https://minio/transcript.txt"})
        upload.assert_called_once()
        save_asr.assert_not_called()

    def test_source_transcription_fails_on_no_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.m4a"
            audio.write_bytes(b"audio")
            with (
                mock.patch.object(main, "task_work_dir", return_value=Path(tmp)),
                mock.patch.object(main, "_vocals_input_for", return_value=audio),
                mock.patch.object(main.db, "get_task", return_value={"source_url": "https://youtu.be/dQw4w9WgXcQ"}),
                mock.patch.object(main.db, "create_whisper_run", return_value=7),
                mock.patch.object(main, "recognize_speech", side_effect=NoSpeechDetected("none")),
                mock.patch.object(main.db, "finish_whisper_run"),
            ):
                with self.assertRaisesRegex(RuntimeError, "无人声"):
                    main.handle({"task_id": "task-1", "sub_stage": "source_transcription", "task_type": "narration", "audio_source_url": "minio://audio"})

    def test_dubbing_fails_on_no_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.m4a"
            audio.write_bytes(b"audio")
            with (
                mock.patch.object(main, "task_work_dir", return_value=Path(tmp)),
                mock.patch.object(main, "_vocals_input_for", return_value=audio),
                mock.patch.object(main.db, "get_task", return_value={"source_url": "https://youtu.be/dQw4w9WgXcQ"}),
                mock.patch.object(main.db, "create_whisper_run", return_value=7),
                mock.patch.object(main, "recognize_speech", side_effect=NoSpeechDetected("none")),
                mock.patch.object(main.db, "finish_whisper_run") as finish_run,
            ):
                with self.assertRaisesRegex(RuntimeError, "无人声"):
                    main.handle({"task_id": "task-1", "sub_stage": "main", "task_type": "dubbing", "audio_vocals_url": "minio://audio"})

        finish_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
