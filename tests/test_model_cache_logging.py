from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ydbi_whisper import whisper_asr


class SemanticModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        whisper_asr._WTPSPLIT_MODEL = None
        whisper_asr._WTPSPLIT_UNAVAILABLE = False

    def tearDown(self) -> None:
        whisper_asr._WTPSPLIT_MODEL = None
        whisper_asr._WTPSPLIT_UNAVAILABLE = False

    def test_complete_cache_is_loaded_by_local_path(self) -> None:
        calls = []

        def fake_sat(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return "model"

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            tokenizer = root / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            (model / "config.json").touch()
            (model / "model.safetensors").touch()
            (tokenizer / "tokenizer_config.json").touch()
            (tokenizer / "tokenizer.json").touch()

            fake_hub = types.SimpleNamespace(
                snapshot_download=mock.Mock(side_effect=[str(model), str(tokenizer)])
            )
            fake_wtpsplit = types.SimpleNamespace(SaT=fake_sat)
            with mock.patch.dict(
                sys.modules,
                {"huggingface_hub": fake_hub, "wtpsplit": fake_wtpsplit},
            ), mock.patch.object(whisper_asr.log, "info") as info:
                result = whisper_asr._wtpsplit_model()

        self.assertEqual("model", result)
        self.assertEqual(Path(model), calls[0][0])
        self.assertEqual(Path(tokenizer), calls[0][1]["tokenizer_name_or_path"])
        self.assertEqual(
            {"local_files_only": True}, calls[0][1]["from_pretrained_kwargs"]
        )
        info.assert_not_called()

    def test_missing_cache_prints_one_chinese_download_message(self) -> None:
        calls = []

        def fake_sat(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return "model"

        fake_hub = types.SimpleNamespace(
            snapshot_download=mock.Mock(side_effect=FileNotFoundError)
        )
        fake_wtpsplit = types.SimpleNamespace(SaT=fake_sat)
        with mock.patch.dict(
            sys.modules,
            {"huggingface_hub": fake_hub, "wtpsplit": fake_wtpsplit},
        ), mock.patch.object(whisper_asr.log, "info") as info:
            result = whisper_asr._wtpsplit_model()

        self.assertEqual("model", result)
        self.assertEqual("sat-3l-sm", calls[0][0])
        self.assertEqual(
            "facebookAI/xlm-roberta-base",
            calls[0][1]["tokenizer_name_or_path"],
        )
        info.assert_called_once_with("正在下载语义分句模型 sat-3l-sm（首次使用）")


if __name__ == "__main__":
    unittest.main()
