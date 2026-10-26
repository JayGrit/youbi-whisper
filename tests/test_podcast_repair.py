from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from ydbi_whisper import db, main, podcast_repair


def cue(text: str, start: int, end: int, run_id: int = 1) -> dict:
    return {
        "text": text, "start_time": start, "end_time": end,
        "words": [], "_whisper_run_id": run_id,
    }


class PodcastRepairTest(unittest.TestCase):
    def test_low_speed_gate_finds_only_severe_segment(self) -> None:
        segments = [cue("Wow.", 0, 22640), cue("No.", 30000, 31842)]
        self.assertEqual([index for index, _ in podcast_repair.slow_segments(segments)], [0])

    def test_local_repair_adds_missing_opening_and_keeps_right_anchor(self) -> None:
        original = [cue("Wow.", 0, 22640), cue("It is a complete inversion.", 22691, 25165)]
        candidate = [
            cue("Picture this.", 0, 1210, 2),
            cue("Wow.", 19632, 19985, 2),
            cue("10 hours just to secure a single fare.", 19985, 22701, 2),
            cue("It is a complete inversion.", 22752, 25119, 2),
        ]
        with (
            mock.patch.object(podcast_repair, "_crop_audio"),
            mock.patch.object(podcast_repair, "_recognize_attempt", return_value=candidate),
        ):
            repaired = podcast_repair._repair_group(
                original, 0, 0,
                task_id="podcast-18", language="en", source_url="source",
                input_audio_url="audio", vocals=Path("audio.m4a"),
                session=Path("/tmp"), parent_run_id=1, duration_ms=26000,
            )
        self.assertEqual([item["text"] for item in repaired[:3]], [
            "Picture this.", "Wow.", "10 hours just to secure a single fare.",
        ])
        self.assertEqual(repaired[2]["end_time"], 22641)
        self.assertEqual(repaired[3], original[1])
        self.assertEqual([item["_whisper_run_id"] for item in repaired], [2, 2, 2, 1])
        self.assertEqual(podcast_repair.slow_segments(repaired), [])

    def test_unresolved_local_repair_falls_back_to_full_run(self) -> None:
        original = [cue("Wow.", 0, 22640)]
        full = [cue("Picture this.", 0, 1200, 3)]
        with (
            mock.patch.object(podcast_repair, "_repair_group", side_effect=ValueError("anchor")),
            mock.patch.object(podcast_repair, "_recognize_attempt", return_value=full) as recognize,
        ):
            repaired = podcast_repair.repair_podcast_segments(
                original, task_id="podcast-18", language="en", source_url="source",
                input_audio_url="audio", vocals=Path("audio.m4a"),
                session=Path("/tmp"), parent_run_id=1, duration_ms=26000,
            )
        self.assertEqual(repaired, full)
        recognize.assert_called_once()
        self.assertEqual(recognize.call_args.kwargs["start_ms"], 0)
        self.assertEqual(recognize.call_args.kwargs["end_ms"], 26000)

    def test_full_run_still_slow_fails_quality_gate(self) -> None:
        original = [cue("Wow.", 0, 22640)]
        with (
            mock.patch.object(podcast_repair, "_repair_group", side_effect=ValueError("anchor")),
            mock.patch.object(podcast_repair, "_recognize_attempt", return_value=original),
        ):
            with self.assertRaisesRegex(RuntimeError, "Full retry left slow subtitles"):
                podcast_repair.repair_podcast_segments(
                    original, task_id="podcast-18", language="en", source_url="source",
                    input_audio_url="audio", vocals=Path("audio.m4a"),
                    session=Path("/tmp"), parent_run_id=1, duration_ms=26000,
                )

    def test_saved_segments_keep_their_source_run(self) -> None:
        connection = mock.MagicMock()
        connection.__enter__.return_value = connection
        cursor = connection.cursor.return_value
        with mock.patch.object(db, "connect", return_value=connection):
            db.save_asr_segments("podcast-18", [cue("Picture this.", 0, 1210, 2)], 1)
        insert = cursor.execute.call_args_list[1]
        self.assertEqual(insert.args[1][-2], 2)
        connection.commit.assert_called_once()

    def test_podcast_handle_repairs_before_saving_final_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.m4a"
            audio.write_bytes(b"audio")
            repaired = [cue("Picture this.", 0, 1210, 2)]
            with (
                mock.patch.object(main, "task_work_dir", return_value=Path(directory)),
                mock.patch.object(main, "_vocals_input_for", return_value=audio),
                mock.patch.object(main.db, "get_task", return_value={"source_language": "en"}),
                mock.patch.object(main.db, "create_whisper_run", return_value=1),
                mock.patch.object(main, "recognize_speech", return_value={
                    "audio_info": {"duration": 26000},
                    "result": {"utterances": [
                        cue("Wow.", 0, 22640),
                        cue("It is a complete inversion.", 22722, 25119),
                    ]},
                }),
                mock.patch.object(main, "repair_podcast_segments", return_value=repaired) as repair,
                mock.patch.object(main.db, "save_asr_segments") as save,
                mock.patch.object(main.db, "finish_whisper_run"),
            ):
                result = main.handle({
                    "task_id": "podcast-18", "task_type": "podcast", "sub_stage": "dialogue",
                    "audio_source_url": "minio://audio", "source_language": "en",
                })
        self.assertEqual(result, {"asr_json_path": "db://whisper_asr_segment/podcast-18"})
        repair.assert_called_once()
        save.assert_called_once_with("podcast-18", repaired, 1)


if __name__ == "__main__":
    unittest.main()
