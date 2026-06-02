from __future__ import annotations

import os
import tempfile
from pathlib import Path


MYSQL_CONFIG = {
    "host": "120.53.92.66",
    "port": 3306,
    "user": "hoshuuch",
    "password": "490229",
    "database": "youbi",
}

WORK_ROOT = Path(tempfile.gettempdir()) / "ydbi"
WORKFOLDER = WORK_ROOT
WORK_DIR = WORK_ROOT / "whisper"
POLL_INTERVAL_SECONDS = 10
SERVICE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("YDBI_WHISPER_MODEL_ROOT", SERVICE_ROOT / "models")).expanduser()
TORCH_HOME = Path(os.environ.get("TORCH_HOME", MODEL_ROOT / "torch")).expanduser()
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

COOKIE_DIR = Path("/Users/hoshuuch/Money/YouBi/data/cookies").expanduser()

STORAGE_BACKEND = "minio"
MINIO_ENDPOINT = "http://120.53.92.66:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "ydbi"
MINIO_PUBLIC_BASE = "/minio"
MINIO_FULL_BASE_URL = "https://120.53.92.66/minio"
MINIO_SECURE = False

DEVICE = "auto"
WHISPER_MODEL = "large-v3-turbo"
WHISPER_DOWNLOAD_ROOT = os.environ.get("YDBI_WHISPER_DOWNLOAD_ROOT", str(MODEL_ROOT / "openai-whisper"))
WHISPER_ENGINE = os.environ.get("YDBI_WHISPER_ENGINE", "whisperx").strip().lower()
WHISPERX_MODEL_PATH = os.environ.get("YDBI_WHISPERX_MODEL_PATH", "").strip()
WHISPERX_COMPUTE_TYPE = os.environ.get("YDBI_WHISPERX_COMPUTE_TYPE", "default").strip()
WHISPERX_BATCH_SIZE = int(os.environ.get("YDBI_WHISPERX_BATCH_SIZE", "16"))
WHISPERX_VAD_METHOD = os.environ.get("YDBI_WHISPERX_VAD_METHOD", "silero").strip().lower()
WHISPERX_VAD_ONSET = float(os.environ.get("YDBI_WHISPERX_VAD_ONSET", "0.5"))
WHISPERX_VAD_OFFSET = float(os.environ.get("YDBI_WHISPERX_VAD_OFFSET", "0.363"))
WHISPERX_CHUNK_SIZE = int(os.environ.get("YDBI_WHISPERX_CHUNK_SIZE", "30"))
TEST_API_HOST = os.environ.get("YDBI_WHISPER_TEST_API_HOST", "127.0.0.1")
TEST_API_PORT = int(os.environ.get("YDBI_WHISPER_TEST_API_PORT", "8213"))


def device() -> str:
    if DEVICE.lower() != "auto":
        return DEVICE

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def task_work_dir(task_id: str) -> Path:
    path = WORK_DIR / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path
