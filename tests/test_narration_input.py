from __future__ import annotations

import unittest
from unittest import mock

from ydbi_whisper import video_info


class NarrationInputTests(unittest.TestCase):
    def test_video_info_merge_includes_narration_task_type(self) -> None:
        row = {"task_id": "narration-6", "status": "ready"}
        with mock.patch.object(
            video_info,
            "get",
            return_value={
                "task_id": "narration-6",
                "task_type": "narration",
                "audio_dubbing_url": "https://example.test/narration.wav",
            },
        ):
            merged = video_info.merge_into(row)

        self.assertEqual("narration", merged["task_type"])
        self.assertEqual(
            "https://example.test/narration.wav",
            merged["audio_dubbing_url"],
        )


if __name__ == "__main__":
    unittest.main()
