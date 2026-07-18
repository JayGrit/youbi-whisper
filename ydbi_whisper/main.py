from __future__ import annotations

import logging
import shutil
import hashlib
from pathlib import Path
from typing import Any

from ydbi_whisper import db
from ydbi_whisper.config import task_work_dir
from ydbi_whisper.sources import detect_source
from ydbi_whisper.storage import download
from ydbi_whisper.storage import upload
from ydbi_whisper.local_export import render_speaker_txt, render_srt, render_txt
from ydbi_whisper.whisper_asr import NoSpeechDetected, align_known_text, recognize_speech
from ydbi_whisper.worker import run_polling_worker

# 当前模块的 logger，用于输出 whisper 阶段的运行日志
log = logging.getLogger(__name__)

DUBBING_MULTI_SEGMENT_TASK_TYPE = "dubbing_multi_segment"
DUBBING_CHUNK_ALIGNED_TASK_TYPE = "dubbing_chunk_aligned"
DUBBING_ALIGNMENT_TASK_TYPES = {
    DUBBING_MULTI_SEGMENT_TASK_TYPE,
    DUBBING_CHUNK_ALIGNED_TASK_TYPE,
}
DIALOGUE_SUB_STAGES = {"dialogue", "对话"}
PPT_ALIGNMENT_SUB_STAGE = "ppt_alignment"


def _asr_language_from_source_language(value: object) -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    if not language:
        return ""
    if language in {"中文", "汉语", "漢語", "chinese", "mandarin"} or language.startswith("zh"):
        return "zh"
    if language in {"英文", "英语", "英語", "english"} or language.startswith("en"):
        return "en"
    return language.split("-", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_destination(session: Path, source_ref: str) -> Path:
    # 从 source_ref 中去掉 URL 查询参数部分，只保留真实路径部分
    # 例如 xxx.wav?token=abc 会先变成 xxx.wav
    suffix = Path(source_ref.split("?", 1)[0]).suffix or ".wav"

    # 生成当前任务的人声音频下载目标路径
    # 文件名统一叫 audio_vocals，后缀沿用远程文件后缀；如果没有后缀则默认 .wav
    return session / "media" / f"audio_vocals{suffix}"


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _vocals_input_for(row: dict, session: Path) -> Path:
    # 从任务行中取出 task_id
    task_id = row["task_id"]

    # 从任务行中取出人声音频的远程地址
    # 这里一般是 MinIO 或其他对象存储中的 audio_vocals_url
    source_transcription = str(row.get("sub_stage") or "main") == "source_transcription"
    task_type = str(row.get("task_type") or "").strip().lower()
    dubbing_alignment = str(row.get("sub_stage") or "main") == "dubbing_alignment"
    ppt_alignment = task_type == "ppt" and str(row.get("sub_stage") or "main") == PPT_ALIGNMENT_SUB_STAGE
    narration_audio_url = (
        str(row.get("audio_dubbing_url") or "").strip()
        if task_type == "narration" and not source_transcription
        else ""
    )
    dubbing_alignment_audio_url = (
        str(row.get("audio_dubbing_url") or "").strip()
        if task_type in DUBBING_ALIGNMENT_TASK_TYPES and dubbing_alignment
        else ""
    )
    ppt_alignment_audio_url = db.get_ppt_dialogue_audio_url(task_id) if ppt_alignment else ""
    audio_vocals_url = (
        ppt_alignment_audio_url
        or dubbing_alignment_audio_url
        or narration_audio_url
        or str(row.get("audio_vocals_url") or "").strip()
    )
    has_background_audio = row.get("has_background_audio")
    source_transcription_with_vocals = (
        task_type == "narration"
        and source_transcription
        and _truthy(has_background_audio)
    )
    input_url = audio_vocals_url
    input_label = "vocals"
    if not input_url and not source_transcription_with_vocals:
        input_url = str(row.get("audio_source_url") or "").strip()
        input_label = "source audio"
    elif dubbing_alignment_audio_url:
        input_label = "dubbing alignment audio"
    elif narration_audio_url:
        input_label = "narration audio"
    elif ppt_alignment_audio_url:
        input_label = "ppt dialogue audio"

    # 如果没有人声音频地址，说明上游 demucs 或下载阶段没有正确产出 vocals
    if not input_url:
        if source_transcription_with_vocals:
            raise FileNotFoundError(f"audio_vocals_url is missing for narration source transcription task: {task_id}")
        raise FileNotFoundError(f"audio_vocals_url is missing for task: {task_id}")
    if dubbing_alignment and not dubbing_alignment_audio_url:
        raise FileNotFoundError(f"audio_dubbing_url is missing for dubbing alignment task: {task_id}")
    if ppt_alignment and not ppt_alignment_audio_url:
        raise FileNotFoundError(f"product_ppt.ppt_dialogue_audio_url is missing for ppt alignment task: {task_id}")

    # 计算本地下载目标路径
    destination = _download_destination(session, input_url)

    log.info("任务 %s：正在下载人声音频", task_id)
    log.debug(
        "任务 %s 下载详情：输入类型=%s，来源=%s，目标=%s",
        task_id,
        input_label,
        input_url,
        destination,
    )

    # 将远程人声音频下载到本地 session 工作目录，并返回本地文件路径
    return download(input_url, destination)


def handle(row: dict) -> dict[str, Any]:
    # 当前 whisper 阶段处理的任务 ID
    task_id = row["task_id"]
    sub_stage = str(row.get("sub_stage") or "main")

    # 获取当前任务的本地工作目录
    # 后续会在这个目录下存放临时音频、metadata 等文件
    session = task_work_dir(task_id)
    run_id: int | None = None
    ppt_subtitle_srt_url = ""

    try:
        # 下载或准备当前任务的人声音频文件
        vocals = _vocals_input_for(row, session)

        # 从数据库中读取任务主表信息
        # 主要用于拿 source_url，进而判断来源类型和 ASR 语言
        task = db.get_task(task_id)

        # 根据任务的 source_url 判断来源
        # source 中一般会包含 asr_language 等配置
        source_url = str(task.get("source_url") or "").strip()
        task_type = str(row.get("task_type") or "").strip().lower()
        narration_task = task_type == "narration"
        dubbing_alignment = task_type in DUBBING_ALIGNMENT_TASK_TYPES and sub_stage == "dubbing_alignment"
        ppt_alignment = task_type == "ppt" and sub_stage == PPT_ALIGNMENT_SUB_STAGE
        source_transcription = sub_stage == "source_transcription"
        dialogue_stage = sub_stage in DIALOGUE_SUB_STAGES
        if (narration_task and not source_transcription) or dubbing_alignment or ppt_alignment:
            asr_language = "zh"
            if ppt_alignment:
                source_name = "ppt_alignment"
            else:
                source_name = "dubbing_alignment" if dubbing_alignment else "narration"
        else:
            if not source_url:
                raise ValueError(f"source_url is missing for task: {task_id}")
            source = detect_source(source_url)
            asr_language = _asr_language_from_source_language(
                task.get("source_language") or row.get("source_language")
            ) or source.asr_language
            source_name = source.name
        run_id = db.create_whisper_run(
            task_id=task_id,
            language=asr_language,
            source_url=source_url or source_name,
            input_audio_url=str(
                (row.get("audio_vocals_url") if source_transcription and _truthy(row.get("has_background_audio")) else None)
                or (row.get("audio_source_url") if source_transcription else None)
                or row.get("audio_dubbing_url")
                or row.get("audio_vocals_url")
                or row.get("audio_source_url")
                or ""
            ),
            input_local_path=str(vocals),
            input_file_size=vocals.stat().st_size,
            input_sha256=_sha256(vocals),
            sub_stage=sub_stage,
        )

        log.info("任务 %s：正在识别语音", task_id)
        log.debug("任务 %s 识别参数：音频=%s，语言=%s", task_id, vocals, asr_language)

        # 调用 Whisper ASR 进行语音识别
        # 返回 data，结构中包含 audio_info、result.text、result.utterances 等信息
        try:
            if (narration_task and not source_transcription) or dubbing_alignment or ppt_alignment:
                known_segments = (
                    db.list_dubbing_alignment_segments(task_id)
                    if dubbing_alignment
                    else db.list_ppt_alignment_segments(task_id)
                    if ppt_alignment
                    else db.list_narration_alignment_segments(task_id)
                )
                if not known_segments:
                    raise RuntimeError(
                        f"known-text alignment segments are missing for task: {task_id}"
                    )
                data = align_known_text(
                    vocals,
                    known_segments,
                    language=asr_language,
                    task_id=task_id,
                    run_id=run_id,
                )
            else:
                data = recognize_speech(
                    vocals,
                    session,
                    language=asr_language,
                    task_id=task_id,
                    run_id=run_id,
                    diarize=dialogue_stage,
                )
        except NoSpeechDetected:
            if source_transcription:
                raise RuntimeError("没有原生字幕且未识别到语音")
            log.warning("任务 %s：未检测到有效语音，已跳过字幕处理", task_id)
            db.finish_whisper_run(run_id, "success")
            return {}

        # 保存后处理后的 ASR 识别结果
        # 后处理包括标准化字段、过滤空文本、给 start/end 加 padding、防止片段过紧等
        if source_transcription:
            result = data.get("result") or {}
            duration_ms = int((data.get("audio_info") or {}).get("duration") or 0)
            from ydbi_whisper.asr_segments import fix_asr_segment_rows
            asr_segments = fix_asr_segment_rows(result.get("utterances") or [], duration_ms)
            transcript = render_txt(asr_segments, str(result.get("text") or ""))
            if not transcript.strip():
                raise RuntimeError("没有原生字幕且未识别到语音")
            transcript_path = session / "source_transcription.txt"
            transcript_path.write_text(transcript, encoding="utf-8")
            transcript_url = upload(
                transcript_path,
                f"{task_id}/whisper/source_transcription.txt",
                "text/plain; charset=utf-8",
            )
        else:
            if dialogue_stage:
                result = data.get("result") or {}
                duration_ms = int((data.get("audio_info") or {}).get("duration") or 0)
                from ydbi_whisper.asr_segments import fix_asr_segment_rows
                asr_segments = fix_asr_segment_rows(result.get("utterances") or [], duration_ms)
                dialogue_txt = render_speaker_txt(asr_segments)
                if not dialogue_txt.strip():
                    raise RuntimeError("对话文本生成结果为空")
                dialogue_path = session / "dialogue.txt"
                dialogue_path.write_text(dialogue_txt, encoding="utf-8")
                dialogue_srt_url = upload(
                    dialogue_path,
                    f"{task_id}/whisper/dialogue.txt",
                    "text/plain; charset=utf-8",
                )
            elif dubbing_alignment:
                if task_type == DUBBING_CHUNK_ALIGNED_TASK_TYPE:
                    asr_segments = db.save_asr_result(task_id, asr_language, data, run_id=run_id)
                else:
                    asr_segments = db.save_dubbing_alignment_result(
                        task_id,
                        data,
                        run_id=run_id,
                        known_segments=known_segments,
                    )
            elif ppt_alignment:
                asr_segments = db.save_asr_result(task_id, asr_language, data, run_id=run_id)
                subtitle_srt = render_srt(asr_segments)
                if not subtitle_srt.strip():
                    raise RuntimeError("PPT subtitle SRT generation result is empty")
                subtitle_path = session / "ppt_subtitle.srt"
                subtitle_path.write_text(subtitle_srt, encoding="utf-8")
                ppt_subtitle_srt_url = upload(
                    subtitle_path,
                    f"{task_id}/whisper/ppt_subtitle.srt",
                    "application/x-subrip; charset=utf-8",
                )
                db.update_ppt_subtitle_srt_url(task_id, ppt_subtitle_srt_url)
            else:
                asr_segments = db.save_asr_result(task_id, asr_language, data, run_id=run_id)

        # 统计所有 ASR segment 中的词级时间戳数量
        # 如果运行在 MPS 等不支持 word timestamps 的场景下，这里可能为 0
        word_count = sum(len(item.get("words") or []) for item in asr_segments)

        log.info(
            "任务 %s：语音识别完成，共 %d 段、%d 个词",
            task_id,
            len(asr_segments),
            word_count,
        )
        db.finish_whisper_run(run_id, "success")

    except Exception as exc:
        if run_id is not None:
            db.finish_whisper_run(run_id, "failed", str(exc))
        raise
    finally:
        # 无论识别成功还是失败，都清理当前任务的本地临时工作目录
        # ignore_errors=True 表示清理失败时不再额外抛异常
        shutil.rmtree(session, ignore_errors=True)

    # ASR 结果在数据库中的引用地址
    # 这里不是实际文件路径，而是一个逻辑引用，表示从 whisper_asr_segment 表读取该任务分段
    if sub_stage == "source_transcription":
        return {"source_transcript_txt_url": transcript_url}
    if sub_stage in DIALOGUE_SUB_STAGES:
        return {"dialogue_srt_url": dialogue_srt_url}
    if sub_stage == "dubbing_alignment":
        if task_type == DUBBING_CHUNK_ALIGNED_TASK_TYPE:
            return {"asr_json_path": f"db://whisper_asr_segment/{task_id}"}
        return {"asr_json_path": f"db://speaker_multi_segment/{task_id}"}
    if sub_stage == PPT_ALIGNMENT_SUB_STAGE:
        return {
            "asr_json_path": f"db://whisper_asr_segment/{task_id}",
            "subtitle_srt_url": ppt_subtitle_srt_url,
        }
    asr_ref = f"db://whisper_asr_segment/{task_id}"

    log.debug("任务 %s 识别结果：%s", task_id, asr_ref)

    # 返回给 worker 框架的阶段产物
    # 一般会被写回当前阶段表或任务状态中，供后续 translator / merger 等阶段使用
    return {"asr_json_path": asr_ref}


def main() -> None:
    run_polling_worker(handle)


if __name__ == "__main__":
    main()
