from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MYSQL_CONFIG = {
    "host": "120.53.92.66",
    "port": 3306,
    "user": "hoshuuch",
    "password": "490229",
    "database": "youbi",
}

WORK_ROOT = Path(os.environ.get("YDBI_WORK_ROOT", Path(tempfile.gettempdir()) / "ydbi")).expanduser()
WORKFOLDER = WORK_ROOT
WORK_DIR = Path(os.environ.get("YDBI_WHISPER_WORK_DIR", WORK_ROOT / "whisper")).expanduser()
POLL_INTERVAL_SECONDS = 10
SERVICE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("YDBI_WHISPER_MODEL_ROOT", SERVICE_ROOT / "models")).expanduser()
TORCH_HOME = Path(os.environ.get("TORCH_HOME", MODEL_ROOT / "torch")).expanduser()
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
NLTK_DATA = Path(os.environ.get("NLTK_DATA", MODEL_ROOT / "nltk")).expanduser()
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA))
HF_HOME = Path(os.environ.get("HF_HOME", MODEL_ROOT / "huggingface")).expanduser()
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", MODEL_ROOT / "cache")).expanduser()
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

COOKIE_DIR = Path("/Users/hoshuuch/Money/YouBi/data/cookies").expanduser()

STORAGE_BACKEND = "minio"
MINIO_ENDPOINT = "http://120.53.92.66:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "ydbi"
MINIO_PUBLIC_BASE = "/minio"
MINIO_FULL_BASE_URL = "https://120.53.92.66/minio"
MINIO_SECURE = False

DEVICE = os.environ.get("DEVICE", "").strip() or "Macbook Air M4"
WHISPER_RUNTIME_DEVICE = os.environ.get("YDBI_WHISPER_RUNTIME_DEVICE", "auto").strip() or "auto"
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
WHISPERX_ALIGN = _env_bool("YDBI_WHISPERX_ALIGN", True)
WHISPERX_ALIGN_MODEL = os.environ.get("YDBI_WHISPERX_ALIGN_MODEL", "").strip()
WHISPERX_ALIGN_MODEL_DIR = os.environ.get("YDBI_WHISPERX_ALIGN_MODEL_DIR", str(MODEL_ROOT / "align"))
WHISPERX_ALIGN_LOCAL_FILES_ONLY = _env_bool("YDBI_WHISPERX_ALIGN_LOCAL_FILES_ONLY", True)
WHISPERX_ALIGN_INTERPOLATE_METHOD = os.environ.get("YDBI_WHISPERX_ALIGN_INTERPOLATE_METHOD", "nearest").strip()
WHISPERX_REGROUP_MAX_CHARS = int(os.environ.get("YDBI_WHISPERX_REGROUP_MAX_CHARS", "120"))
WHISPERX_REGROUP_MAX_DURATION_MS = int(os.environ.get("YDBI_WHISPERX_REGROUP_MAX_DURATION_MS", "8000"))
TEST_API_HOST = os.environ.get("YDBI_WHISPER_TEST_API_HOST", "127.0.0.1")
TEST_API_PORT = int(os.environ.get("YDBI_WHISPER_TEST_API_PORT", "8213"))


def device() -> str:
    if WHISPER_RUNTIME_DEVICE.lower() != "auto":
        return WHISPER_RUNTIME_DEVICE

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
