from __future__ import annotations

from pathlib import Path


MYSQL_CONFIG = {
    "host": "120.53.92.66",
    "port": 3306,
    "user": "hoshuuch",
    "password": "490229",
    "database": "youbi",
}

WORKFOLDER = Path("/Users/hoshuuch/Money/YouBi/workfolder").expanduser()
POLL_INTERVAL_SECONDS = 10

COOKIE_DIR = Path("/Users/hoshuuch/Money/YouBi/data/cookies").expanduser()

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
