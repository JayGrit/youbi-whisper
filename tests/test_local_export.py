from __future__ import annotations

import unittest

from ydbi_whisper.local_export import render_speaker_srt, render_srt, render_txt


class LocalExportFormatTest(unittest.TestCase):
    def test_render_srt_uses_srt_timestamps(self) -> None:
        self.assertEqual(
            render_srt([
                {"start_time": 1234, "end_time": 65000, "text": "Hello"},
                {"start_time": 3661007, "end_time": 3662555, "text": "World"},
            ]),
            "1\n00:00:01,234 --> 00:01:05,000\nHello\n\n"
            "2\n01:01:01,007 --> 01:01:02,555\nWorld\n",
        )

    def test_render_txt_uses_one_segment_per_line(self) -> None:
        self.assertEqual(
            render_txt([
                {"text": "Hello"},
                {"text": " "},
                {"text": "World"},
            ]),
            "Hello\nWorld\n",
        )

    def test_render_srt_skips_empty_segments_with_continuous_indexes(self) -> None:
        self.assertEqual(
            render_srt([
                {"start_time": 0, "end_time": 1000, "text": "Hello"},
                {"start_time": 1000, "end_time": 2000, "text": " "},
                {"start_time": 2000, "end_time": 3000, "text": "World"},
            ]),
            "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nWorld\n",
        )

    def test_render_speaker_srt_prefixes_speaker_labels(self) -> None:
        self.assertEqual(
            render_speaker_srt([
                {"start_time": 0, "end_time": 1000, "speaker": "SPEAKER_00", "text": "Hello"},
                {"start_time": 1000, "end_time": 2000, "speaker": "SPEAKER_01", "text": "World"},
            ]),
            "1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00: Hello\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nSPEAKER_01: World\n",
        )

    def test_render_txt_falls_back_to_full_text(self) -> None:
        self.assertEqual(render_txt([], "Hello world"), "Hello world\n")


if __name__ == "__main__":
    unittest.main()
