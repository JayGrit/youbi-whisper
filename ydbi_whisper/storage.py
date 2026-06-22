from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from .config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MINIO_FULL_BASE_URL,
)


@dataclass(frozen=True)
class ObjectRef:
    bucket: str
    object_name: str


def _endpoint_parts() -> tuple[str, bool]:
    parsed = urlparse(MINIO_ENDPOINT)
    if parsed.scheme:
        return parsed.netloc, parsed.scheme == "https"
    return MINIO_ENDPOINT, MINIO_SECURE


def _minio_client() -> Minio:
    endpoint, secure = _endpoint_parts()
    return Minio(endpoint, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=secure)


def _strip_known_prefix(path: str) -> str:
    path = path.split("?", 1)[0].lstrip("/")
    for prefix in (f"minio/{MINIO_BUCKET}/", f"{MINIO_BUCKET}/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def parse_object_ref(ref: str) -> ObjectRef | None:
    value = str(ref or "").strip()
    if not value:
        return None

    parsed = urlparse(value)
    if parsed.scheme == "s3":
        bucket = parsed.netloc or MINIO_BUCKET
        object_name = parsed.path.lstrip("/")
        return ObjectRef(bucket, object_name) if object_name else None

    if parsed.scheme in {"http", "https"}:
        object_name = _strip_known_prefix(parsed.path)
        return ObjectRef(MINIO_BUCKET, object_name) if object_name else None

    if value.startswith("/minio/") or value.startswith(f"/{MINIO_BUCKET}/") or value.startswith(f"{MINIO_BUCKET}/"):
        object_name = _strip_known_prefix(value)
        return ObjectRef(MINIO_BUCKET, object_name) if object_name else None

    return None


def download(ref: str | Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = str(ref)
    local = Path(source.removeprefix("local:"))
    if local.exists() and local.stat().st_size > 0:
        if local.resolve() != destination.resolve():
            shutil.copy2(local, destination)
        return destination

    object_info = parse_object_ref(source)
    if not object_info:
        raise FileNotFoundError(f"input does not exist locally or in minio: {source}")

    try:
        client = _minio_client()
        client.fget_object(object_info.bucket, object_info.object_name, str(destination))
        return destination
    except S3Error as exc:
        raise FileNotFoundError(
            f"input does not exist locally or in minio: {source}; tried "
            f"s3://{object_info.bucket}/{object_info.object_name}: {exc.code}"
        ) from exc


def upload(source: Path, object_name: str, content_type: str = "application/octet-stream") -> str:
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"output does not exist or is empty: {source}")
    client = _minio_client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    client.fput_object(MINIO_BUCKET, object_name, str(source), content_type=content_type)
    return f"{MINIO_FULL_BASE_URL.rstrip('/')}/{MINIO_BUCKET}/{object_name.lstrip('/')}"
