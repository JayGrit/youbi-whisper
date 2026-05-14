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

WORKFOLDER = Path("/Users/hoshuuch/Money/YouBi/workfolder").expanduser()
WORK_DIR = Path(os.environ.get("YDBI_WHISPER_WORK_DIR", Path(tempfile.gettempdir()) / "ydbi" / "whisper")).expanduser()
POLL_INTERVAL_SECONDS = 10

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
WHISPER_DOWNLOAD_ROOT = ""


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
