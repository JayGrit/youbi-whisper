from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from ydbi_whisper import whisper_asr


class KnownTextAlignmentTests(unittest.TestCase):
    def test_chinese_words_are_joined_without_spaces(self) -> None:
        segment = whisper_asr._segment_from_words(
            [
                {"word": "其", "start": 0.1, "end": 0.2},
                {"word": "实", "start": 0.2, "end": 0.3},
                {"word": "。", "start": 0.3, "end": 0.4},
            ],
            "zh",
        )

        self.assertEqual("其实。", segment["text"])

    def test_known_text_uses_database_timing_as_align_input(self) -> None:
        fake_whisperx = types.SimpleNamespace(load_audio=mock.Mock(return_value="audio"))
        aligned = {
            "segments": [
                {
                    "text": "其实。",
                    "start": 0.25,
                    "end": 1.1,
                    "words": [
                        {"word": "其", "start": 0.25, "end": 0.5},
                        {"word": "实", "start": 0.5, "end": 0.8},
                        {"word": "。", "start": 0.8, "end": 1.1},
                    ],
                }
            ],
            "language": "zh",
        }
        fake_pydub = types.SimpleNamespace(
            AudioSegment=types.SimpleNamespace(
                from_file=mock.Mock(return_value=bytes(2000))
            )
        )

        with (
            mock.patch.dict(
                sys.modules,
                {"whisperx": fake_whisperx, "pydub": fake_pydub},
            ),
            mock.patch.object(
                whisper_asr,
                "_align_whisperx_result",
                return_value=aligned,
            ) as align_result,
            mock.patch.object(
                whisper_asr,
                "_regroup_aligned_segments",
                return_value=(aligned["segments"], aligned["segments"]),
            ),
        ):
            payload = whisper_asr.align_known_text(
                Path("/tmp/narration.wav"),
                [
                    {
                        "item_index": 0,
                        "text": "其实。",
                        "start_time": 250,
                        "end_time": 1100,
                    }
                ],
                "zh",
            )

        transcript = align_result.call_args.args[1]["segments"]
        self.assertEqual(
            [{"text": "其实。", "start": 0.25, "end": 1.1}],
            transcript,
        )
        self.assertEqual("其实。", payload["result"]["text"])


if __name__ == "__main__":
    unittest.main()
