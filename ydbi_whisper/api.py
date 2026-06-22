from __future__ import annotations

import json
import logging
import shutil
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .asr_segments import fix_asr_segment_rows
from .config import TEST_API_HOST, TEST_API_PORT, task_work_dir
from .logging_utils import configure_dependency_logging
from .whisper_asr import current_asr_config, recognize_speech

log = logging.getLogger(__name__)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _input_ref(payload: dict[str, Any]) -> str:
    for key in ("audio", "audio_path", "audio_url", "audio_vocals_url"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    raise ValueError("missing audio; provide audio, audio_path, audio_url, or audio_vocals_url")


def _download_destination(session: Path, source_ref: str) -> Path:
    suffix = Path(source_ref.split("?", 1)[0]).suffix or ".wav"
    return session / "media" / f"test_audio{suffix}"


def _recognize(payload: dict[str, Any]) -> dict[str, Any]:
    from .storage import download

    source_ref = _input_ref(payload)
    language = str(payload.get("language") or "en").strip()
    include_segments = bool(payload.get("include_segments", True))
    task_id = f"test-{uuid.uuid4().hex}"
    session = task_work_dir(task_id)

    try:
        audio = download(source_ref, _download_destination(session, source_ref))
        data = recognize_speech(audio, session, language=language)
        response: dict[str, Any] = {
            "ok": True,
            "task_id": task_id,
            "source": source_ref,
            "config": current_asr_config(language=language),
            "result": data,
        }
        if include_segments:
            duration_ms = int((data.get("audio_info") or {}).get("duration") or 0)
            utterances = (data.get("result") or {}).get("utterances") or []
            response["utterances"] = fix_asr_segment_rows(utterances, duration_ms)
        return response
    finally:
        shutil.rmtree(session, ignore_errors=True)


class WhisperTestHandler(BaseHTTPRequestHandler):
    server_version = "ydbi-whisper-test-api/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/health", "/healthz"}:
            self._send_json({"ok": True})
            return
        if path in {"/config", "/params"}:
            self._send_json({"ok": True, "config": current_asr_config(language=None)})
            return
        self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/recognize", "/test/recognize"}:
            self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            self._send_json(_recognize(payload))
        except Exception as exc:
            log.exception("test recognition failed")
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = _json_bytes(payload)
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_dependency_logging()
    server = ThreadingHTTPServer((TEST_API_HOST, TEST_API_PORT), WhisperTestHandler)
    log.info("whisper test api listening on http://%s:%s", TEST_API_HOST, TEST_API_PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
