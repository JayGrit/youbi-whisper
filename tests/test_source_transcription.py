from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ydbi_whisper import main
from ydbi_whisper import asr_orchestration
from ydbi_whisper.whisper_asr import NoSpeechDetected


class SourceTranscriptionTest(unittest.TestCase):
    def test_text_only_asr_skips_alignment_and_segment_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = mock.Mock(device='cpu')
            with (
                mock.patch.object(asr_orchestration._core, '_load_model', return_value=model),
                mock.patch.object(asr_orchestration, '_transcribe', return_value=({'text': 'hello world'}, object())),
                mock.patch.object(asr_orchestration._core.db, 'update_whisper_run_runtime'),
                mock.patch.object(asr_orchestration._core.db, 'save_whisper_raw_segments') as save_raw,
                mock.patch.object(asr_orchestration._core, '_align_whisperx_result') as align,
            ):
                result = asr_orchestration.recognize_speech(
                    Path(tmp) / 'vocals.wav', Path(tmp), 'en', task_id='task-1',
                    run_id=7, text_only=True,
                )
            self.assertEqual({'result': {'text': 'hello world'}}, result)
            self.assertFalse((Path(tmp) / 'metadata').exists())
            save_raw.assert_not_called()
            align.assert_not_called()

    def test_podcast_test_uploads_only_text_without_segment_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'vocals.wav'
            audio.write_bytes(b'audio')
            with (
                mock.patch.object(main, 'task_work_dir', return_value=Path(tmp)),
                mock.patch.object(main, '_vocals_input_for', return_value=audio),
                mock.patch.object(main.db, 'get_task', return_value={'source_language': 'en'}),
                mock.patch.object(main.db, 'create_whisper_run', return_value=7),
                mock.patch.object(main, 'recognize_speech', return_value={'result': {'text': 'hello world'}}) as recognize,
                mock.patch.object(main, 'render_txt') as render_txt,
                mock.patch.object(main, 'upload', return_value='s3://bucket/source.txt'),
                mock.patch.object(main.db, 'save_asr_result') as save_asr,
                mock.patch.object(main.db, 'finish_whisper_run'),
            ):
                result = main.handle({
                    'task_id': 'task-1', 'task_type': 'podcast_test',
                    'sub_stage': 'source_transcription',
                    'audio_vocals_url': 's3://bucket/vocals.wav',
                })
            self.assertEqual({'source_transcript_txt_url': 's3://bucket/source.txt'}, result)
            self.assertTrue(recognize.call_args.kwargs['text_only'])
            self.assertFalse(recognize.call_args.kwargs['diarize'])
            render_txt.assert_not_called()
            save_asr.assert_not_called()

    def test_podcast_test_requires_separated_vocals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'audio_vocals_url is missing'):
                main._vocals_input_for({
                    'task_id': 'podcast-test-1',
                    'task_type': 'podcast_test',
                    'sub_stage': 'source_transcription',
                    'has_background_audio': 0,
                    'audio_source_url': 'minio://source-audio',
                }, Path(tmp))

    def test_source_transcription_uploads_txt_without_writing_asr_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.m4a"
            audio.write_bytes(b"audio")
            with (
                mock.patch.object(main, "task_work_dir", return_value=Path(tmp)),
                mock.patch.object(main, "_vocals_input_for", return_value=audio),
                mock.patch.object(main.db, "get_task", return_value={"source_url": "https://youtu.be/dQw4w9WgXcQ", "source_language": "en"}),
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
                mock.patch.object(main.db, "get_task", return_value={"source_url": "https://youtu.be/dQw4w9WgXcQ", "source_language": "en"}),
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
                mock.patch.object(main.db, "get_task", return_value={"source_url": "https://youtu.be/dQw4w9WgXcQ", "source_language": "en"}),
                mock.patch.object(main.db, "create_whisper_run", return_value=7),
                mock.patch.object(main, "recognize_speech", side_effect=NoSpeechDetected("none")),
                mock.patch.object(main.db, "finish_whisper_run") as finish_run,
            ):
                with self.assertRaisesRegex(RuntimeError, "无人声"):
                    main.handle({"task_id": "task-1", "sub_stage": "main", "task_type": "dubbing", "audio_vocals_url": "minio://audio"})

        finish_run.assert_called_once_with(7, "failed", "无人声")


if __name__ == "__main__":
    unittest.main()
