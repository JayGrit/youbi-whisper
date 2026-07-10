from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qs, urlparse


YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
BILIBILI_HOSTS = {"bilibili.com", "www.bilibili.com", "m.bilibili.com"}
DOUYIN_HOSTS = {"douyin.com", "www.douyin.com", "m.douyin.com", "v.douyin.com", "iesdouyin.com"}


def _extract_youtube_id(parsed) -> str | None:
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host in {"youtu.be", "www.youtu.be"}:
        candidate = path.split("/")[0]
        if YOUTUBE_ID_RE.match(candidate):
            return candidate

    if "youtube.com" not in host:
        return None

    query_id = parse_qs(parsed.query).get("v", [""])[0]
    if YOUTUBE_ID_RE.match(query_id):
        return query_id

    parts = path.split("/")
    for prefix in ("shorts", "embed", "live"):
        if len(parts) >= 2 and parts[0] == prefix and YOUTUBE_ID_RE.match(parts[1]):
            return parts[1]
    return None


def is_youtube_url(url: str) -> bool:
    try:
        return _extract_youtube_id(urlparse(url.strip())) is not None
    except ValueError:
        return False


def is_bilibili_url(url: str) -> bool:
    return urlparse(url.strip()).netloc.lower() in BILIBILI_HOSTS


def is_douyin_url(url: str) -> bool:
    return urlparse(url.strip()).netloc.lower() in DOUYIN_HOSTS


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
    SourceConfig(
        name="douyin",
        matches=is_douyin_url,
        asr_language="zh",
    ),
]


def detect_source(url: str) -> SourceConfig:
    for source in SOURCES:
        if source.matches(url):
            return source
    raise ValueError(f"No source matches URL: {url}")
