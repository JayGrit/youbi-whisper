from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ydbi_whisper import whisper_asr


class KnownTextAlignmentTests(unittest.TestCase):
    def test_old_whisperx_loads_huggingface_model_from_local_snapshot(self) -> None:
        calls = []

        def load_align_model(*args, **kwargs):
            calls.append((args, kwargs))
            if "model_cache_only" in kwargs:
                raise TypeError(
                    "load_align_model() got an unexpected keyword argument "
                    "'model_cache_only'"
                )
            return "model", {"language": "zh"}

        fake_whisperx = types.SimpleNamespace(
            load_align_model=load_align_model,
            alignment=types.SimpleNamespace(
                DEFAULT_ALIGN_MODELS_HF={"zh": "owner/chinese-align"}
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "models--owner--chinese-align"
            (repo / "refs").mkdir(parents=True)
            (repo / "refs" / "main").write_text("revision-1", encoding="utf-8")
            snapshot = repo / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            with mock.patch.object(
                whisper_asr, "WHISPERX_ALIGN_MODEL_DIR", temp_dir
            ):
                result = whisper_asr._load_align_model(fake_whisperx, "zh", "cpu")

        self.assertEqual(("model", {"language": "zh"}), result)
        self.assertEqual(2, len(calls))
        self.assertEqual(str(snapshot), calls[1][1]["model_name"])
        self.assertNotIn("model_cache_only", calls[1][1])

    def test_align_model_uses_local_cache_and_is_reused(self) -> None:
        fake_whisperx = types.SimpleNamespace(
            load_align_model=mock.Mock(return_value=("model", {"language": "zh"})),
            align=mock.Mock(
                return_value={"segments": [], "word_segments": []}
            ),
        )
        whisper_asr._ALIGN_MODELS.clear()

        for _ in range(2):
            whisper_asr._align_whisperx_result(
                fake_whisperx,
                {
                    "language": "zh",
                    "segments": [{"text": "其实。", "start": 0.0, "end": 1.0}],
                },
                "audio",
                "zh",
                "cpu",
            )

        fake_whisperx.load_align_model.assert_called_once()
        self.assertTrue(
            fake_whisperx.load_align_model.call_args.kwargs["model_cache_only"]
        )

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
