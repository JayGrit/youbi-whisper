from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .youtube import is_bilibili_url, is_youtube_url


@dataclass(frozen=True)
class SourceConfig:
    name: str
    matches: Callable[[str], bool]
    asr_language: str


SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="youtube",
        matches=is_youtube_url,
        asr_language="en",
    ),
    SourceConfig(
        name="bilibili",
        matches=is_bilibili_url,
        asr_language="zh",
    ),
]


def detect_source(url: str) -> SourceConfig:
    for source in SOURCES:
        if source.matches(url):
            return source
    raise ValueError(f"No source matches URL: {url}")
