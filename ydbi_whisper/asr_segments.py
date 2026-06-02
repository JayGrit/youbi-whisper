from __future__ import annotations


def _start_pad(idx: int, utts: list, start_pad: int, end_pad: int, min_gap: int) -> int:
    # 取出当前句子的原始开始时间
    orig_start = utts[idx]["start_time"]

    # 如果是第一句，前面没有上一句，不需要考虑和上一句的间隔
    if idx == 0:
        # 开始时间向前扩展 start_pad，但不能小于 0
        return max(0, orig_start - start_pad)

    # 取出上一句的结束时间
    prev_end = utts[idx - 1]["end_time"]

    # 当前句子开始时间 与 上一句结束时间之间的空隙
    gap = orig_start - prev_end

    # 当前句子开始 padding + 上一句结束 padding 的总需求空间
    total = start_pad + end_pad

    # 如果两句之间的间隔足够大：
    # 可以同时满足上一句 end_pad、当前句 start_pad，并且中间还保留 min_gap
    if gap >= total + min_gap:
        # 当前句子的开始时间直接向前扩展 start_pad
        return orig_start - start_pad

    # 如果间隔不够完整 padding，但仍然大于最小间隔 min_gap
    if gap > min_gap:
        # 将可用于 padding 的空间 gap - min_gap 按 start_pad 和 end_pad 的比例分配
        # 当前句子拿到属于 start_pad 的那一部分
        share = int((gap - min_gap) * start_pad / total)

        # 当前句子的开始时间向前扩展 share
        return orig_start - share

    # 如果两句之间的 gap 小于等于 min_gap，说明两句太近
    # 直接把两句之间的空隙一分为二，当前句从中点之后开始
    return prev_end + gap // 2


def _end_pad(idx: int, utts: list, duration: int, start_pad: int, end_pad: int, min_gap: int) -> int:
    # 取出当前句子的原始结束时间
    orig_end = utts[idx]["end_time"]

    # 如果是最后一句，后面没有下一句，不需要考虑和下一句的间隔
    if idx == len(utts) - 1:
        # 如果传入了音频/视频总时长 duration，则结束时间不能超过 duration
        # 如果没有 duration，则直接向后扩展 end_pad
        return min(duration, orig_end + end_pad) if duration else orig_end + end_pad

    # 取出下一句的开始时间
    next_start = utts[idx + 1]["start_time"]

    # 当前句子结束时间 与 下一句开始时间之间的空隙
    gap = next_start - orig_end

    # 当前句子结束 padding + 下一句开始 padding 的总需求空间
    total = start_pad + end_pad

    # 如果两句之间的间隔足够大：
    # 可以同时满足当前句 end_pad、下一句 start_pad，并且中间还保留 min_gap
    if gap >= total + min_gap:
        # 当前句子的结束时间直接向后扩展 end_pad
        return orig_end + end_pad

    # 如果间隔不够完整 padding，但仍然大于最小间隔 min_gap
    if gap > min_gap:
        # 将可用于 padding 的空间 gap - min_gap 按 start_pad 和 end_pad 的比例分配
        # 当前句子拿到属于 end_pad 的那一部分
        share = int((gap - min_gap) * end_pad / total)

        # 当前句子的结束时间向后扩展 share
        return orig_end + share

    # 如果两句之间的 gap 小于等于 min_gap，说明两句太近
    # 直接把两句之间的空隙一分为二，当前句在中点处结束
    return orig_end + gap // 2


def _apply_padding(utts: list, duration: int, start_pad: int, end_pad: int) -> list:
    # 如果没有任何句子，直接返回原列表
    if not utts:
        return utts

    # 两个相邻片段之间至少保留的间隔，单位通常是毫秒
    min_gap = 50

    # 用于保存加完 padding 之后的新片段列表
    result = []

    # 遍历每一个 ASR 片段
    for idx in range(len(utts)):
        # 计算当前片段新的开始时间
        new_start = _start_pad(idx, utts, start_pad, end_pad, min_gap)

        # 计算当前片段新的结束时间
        new_end = _end_pad(idx, utts, duration, start_pad, end_pad, min_gap)

        # 如果传入了音频/视频总时长 duration，则结束时间不能超过 duration
        # 如果没有 duration，则使用计算出来的 new_end
        clamped_end = min(duration, new_end) if duration else new_end

        # 保留原始片段中的其他字段，只覆盖 start_time 和 end_time
        result.append({
            **utts[idx],
            "start_time": max(0, new_start),
            "end_time": clamped_end,
        })

    # 返回处理后的片段列表
    return result


def _speaker(utterance: dict) -> str | None:
    # 优先从 utterance["speaker"] 中取说话人信息
    # 如果不存在或为空，则转成空字符串
    speaker = str(utterance.get("speaker") or "").strip()

    # 如果 speaker 非空，直接返回
    if speaker:
        return speaker

    # 如果顶层 speaker 为空，则尝试从 additions 字段中读取
    additions = utterance.get("additions") or {}

    # 只有 additions 是 dict 时，才尝试从其中取 speaker
    if isinstance(additions, dict):
        speaker = str(additions.get("speaker") or "").strip()

    # 如果最终 speaker 仍然为空，则返回 None
    return speaker or None


def normalize_asr_segments(utterances: list[dict]) -> list[dict]:
    # 将原始 ASR utterances 统一整理成后续处理需要的标准格式
    return [
        {
            # 文本内容，已经在 if 条件中去掉了首尾空白
            "text": text,

            # 片段开始时间，缺失时按 0 处理，并转成 int
            "start_time": int(u.get("start_time") or 0),

            # 片段结束时间，缺失时按 0 处理，并转成 int
            "end_time": int(u.get("end_time") or 0),

            # 说话人信息，优先取顶层 speaker，其次取 additions.speaker
            "speaker": _speaker(u),

            # 保留原始 words 字段，通常用于词级时间戳或词级信息
            "words": u.get("words"),

            # whisper 中间产物追踪字段，供最终 ASR 表反查拆分来源
            "_whisper_split_id": u.get("_whisper_split_id"),

        }

        # 遍历每一个原始 ASR 片段
        for u in utterances

        # 只保留 text 非空的片段
        # 同时使用海象运算符 := 将清洗后的文本赋值给 text
        if (text := str(u.get("text") or "").strip())
    ]


def fix_asr_segment_rows(
    utterances: list[dict],
    duration: int,
    start_pad: int = 100,
    end_pad: int = 300,
) -> list[dict]:
    # 先将原始 ASR 结果标准化，过滤掉空文本片段，并统一字段格式
    new_utts = normalize_asr_segments(utterances)

    # 如果标准化后没有任何有效片段，说明 ASR 结果不可用
    if not new_utts:
        raise RuntimeError("ASR result has no utterances.")

    # 对每个片段应用开始/结束 padding，避免字幕或音频切分过紧
    return _apply_padding(new_utts, duration, start_pad, end_pad)
