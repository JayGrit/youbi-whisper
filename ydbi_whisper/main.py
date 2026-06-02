from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ydbi_whisper import db
from ydbi_whisper.config import task_work_dir
from ydbi_whisper.sources import detect_source
from ydbi_whisper.storage import download
from ydbi_whisper.whisper_asr import recognize_speech
from ydbi_whisper.worker import run_polling_worker

# 当前模块的 logger，用于输出 whisper 阶段的运行日志
log = logging.getLogger(__name__)

# 单个 ASR 任务允许的最大分段数量
# 防止 Whisper 产出过多 segment，导致后续字幕翻译、合并、入库等流程压力过大
MAX_ASR_SEGMENTS = 300


def _download_destination(session: Path, source_ref: str) -> Path:
    # 从 source_ref 中去掉 URL 查询参数部分，只保留真实路径部分
    # 例如 xxx.wav?token=abc 会先变成 xxx.wav
    suffix = Path(source_ref.split("?", 1)[0]).suffix or ".wav"

    # 生成当前任务的人声音频下载目标路径
    # 文件名统一叫 audio_vocals，后缀沿用远程文件后缀；如果没有后缀则默认 .wav
    return session / "media" / f"audio_vocals{suffix}"


def _vocals_input_for(row: dict, session: Path) -> Path:
    # 从任务行中取出 task_id
    task_id = row["task_id"]

    # 从任务行中取出人声音频的远程地址
    # 这里一般是 MinIO 或其他对象存储中的 audio_vocals_url
    audio_vocals_url = str(row.get("audio_vocals_url") or "").strip()

    # 如果没有人声音频地址，说明上游 demucs 或下载阶段没有正确产出 vocals
    if not audio_vocals_url:
        raise FileNotFoundError(f"audio_vocals_url is missing for task: {task_id}")

    # 计算本地下载目标路径
    destination = _download_destination(session, audio_vocals_url)

    # 打印下载日志，方便排查当前任务下载的来源和目标路径
    log.info(
        "whisper task=%s downloading vocals from minio url=%s destination=%s",
        task_id,
        audio_vocals_url,
        destination,
    )

    # 将远程人声音频下载到本地 session 工作目录，并返回本地文件路径
    return download(audio_vocals_url, destination)


def handle(row: dict) -> dict[str, str]:
    # 当前 whisper 阶段处理的任务 ID
    task_id = row["task_id"]

    # 获取当前任务的本地工作目录
    # 后续会在这个目录下存放临时音频、metadata 等文件
    session = task_work_dir(task_id)

    try:
        # 下载或准备当前任务的人声音频文件
        vocals = _vocals_input_for(row, session)

        # 从数据库中读取任务主表信息
        # 主要用于拿 source_url，进而判断来源类型和 ASR 语言
        task = db.get_task(task_id)

        # 根据任务的 source_url 判断来源
        # source 中一般会包含 asr_language 等配置
        source = detect_source(task["source_url"])

        # 打印 Whisper 识别前的关键信息
        log.info(
            "whisper task=%s vocals=%s language=%s",
            task_id,
            vocals,
            source.asr_language,
        )

        # 调用 Whisper ASR 进行语音识别
        # 返回 data，结构中包含 audio_info、result.text、result.utterances 等信息
        data = recognize_speech(vocals, session, language=source.asr_language, task_id=task_id)

        # 保存后处理后的 ASR 识别结果
        # 后处理包括标准化字段、过滤空文本、给 start/end 加 padding、防止片段过紧等
        asr_segments = db.save_asr_result(task_id, source.asr_language, data)

        # 如果 Whisper 返回的分段数量过多，则直接中断任务
        # 避免后续处理超出系统设计上限
        if len(asr_segments) > MAX_ASR_SEGMENTS:
            raise ValueError(f"Whisper produced {len(asr_segments)} segments, exceeding the {MAX_ASR_SEGMENTS} limit.")

        # 统计所有 ASR segment 中的词级时间戳数量
        # 如果运行在 MPS 等不支持 word timestamps 的场景下，这里可能为 0
        word_count = sum(len(item.get("words") or []) for item in asr_segments)

        # 打印识别完成日志，包括分段数和词级数量
        log.info(
            "whisper recognized task=%s segments=%d words=%d",
            task_id,
            len(asr_segments),
            word_count,
        )

    finally:
        # 无论识别成功还是失败，都清理当前任务的本地临时工作目录
        # ignore_errors=True 表示清理失败时不再额外抛异常
        shutil.rmtree(session, ignore_errors=True)

    # ASR 结果在数据库中的引用地址
    # 这里不是实际文件路径，而是一个逻辑引用，表示从 yd_asr_segment 表读取该任务分段
    asr_ref = f"db://yd_asr_segment/{task_id}"

    # 打印当前 whisper 阶段最终产物引用
    log.info("whisper output task=%s asr_ref=%s", task_id, asr_ref)

    # 返回给 worker 框架的阶段产物
    # 一般会被写回当前阶段表或任务状态中，供后续 translator / merger 等阶段使用
    return {"asr_json_path": asr_ref}


def main() -> None:
    run_polling_worker(handle)


if __name__ == "__main__":
    main()
