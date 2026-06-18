from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import db
from .config import POLL_INTERVAL_SECONDS
from .service import SERVICE_NAME

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Mapping[str, Any] | None]


def _start_task_heartbeat(stage_name: str) -> threading.Event:
    stop_event = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_event.wait(POLL_INTERVAL_SECONDS):
            try:
                db.record_service_poll()
            except Exception:
                log.exception("更新任务心跳失败")

    try:
        db.record_service_poll()
    except Exception:
        log.exception("更新任务心跳失败")
    thread = threading.Thread(target=heartbeat_loop, name=f"{stage_name}-heartbeat", daemon=True)
    thread.start()
    return stop_event


def run_polling_worker(handler: Handler) -> None:
    stage_name = SERVICE_NAME
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for logger_name in ("faster_whisper", "huggingface_hub", "pyannote", "speechbrain", "torch", "urllib3", "whisperx"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    log.info("语音识别服务已启动")
    while True:
        try:
            db.record_service_poll()
            recycled = db.recycle_stale_running(stage_name)
            if recycled:
                log.warning("已回收 %d 个超时任务", recycled)
            row = db.find_ready(stage_name)
        except Exception:
            log.exception("查询待处理任务失败，%s 秒后重试", POLL_INTERVAL_SECONDS)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if not row:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        task_id = row["task_id"]
        if not db.mark_running(stage_name, task_id):
            continue

        log.info("任务 %s：开始处理", task_id)
        heartbeat_stop = _start_task_heartbeat(stage_name)
        try:
            outputs = handler(row) or {}
            db.mark_success(stage_name, task_id, outputs)
            log.info("任务 %s：处理完成", task_id)
        except Exception as exc:
            log.exception("任务 %s：处理失败", task_id)
            db.mark_failed(stage_name, task_id, str(exc))
        finally:
            heartbeat_stop.set()
