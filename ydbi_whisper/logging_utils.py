from __future__ import annotations

import logging
import warnings


DEPENDENCY_LOGGERS = (
    "faster_whisper",
    "filelock",
    "huggingface_hub",
    "httpcore",
    "httpx",
    "pyannote",
    "speechbrain",
    "torch",
    "transformers",
    "urllib3",
    "whisperx",
    "wtpsplit",
)


def configure_dependency_logging(*, debug: bool = False) -> None:
    """Keep normal runs quiet while preserving dependency logs for --debug."""
    dependency_level = logging.DEBUG if debug else logging.ERROR
    for name in DEPENDENCY_LOGGERS:
        logging.getLogger(name).setLevel(dependency_level)

    if debug:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"When using length constraints .*",
        category=UserWarning,
        module=r"wtpsplit.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*use_return_dict.*deprecated.*",
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
