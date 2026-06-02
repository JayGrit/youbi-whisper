from __future__ import annotations

import logging
import re
import time
from typing import Any

from openai import OpenAI

from . import db
from .config import (
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    WHISPER_SEMANTIC_SEGMENT_CONTEXT_WORDS,
    WHISPER_SEMANTIC_SEGMENT_EDITABLE_WORDS,
    WHISPER_SEMANTIC_SEGMENT_ENABLED,
    WHISPER_SEMANTIC_SEGMENT_MAX_WORDS,
    WHISPER_SEMANTIC_SEGMENT_MIN_WORDS,
    WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
STRICT_ATTEMPT = 2
SENTENCE_BREAK_RE = re.compile(r"[.!?。！？]+[\"')\]}]*$")
MINOR_BREAK_RE = re.compile(r"[,;:，；：]+[\"')\]}]*$")
SOFT_BREAK_WORDS = {
    "and",
    "but",
    "or",
    "because",
    "while",
    "although",
    "however",
    "meanwhile",
    "which",
    "who",
    "that",
}


def _word_text(word: dict[str, Any]) -> str:
    return str(word.get("word") or word.get("text") or "").strip()


def _join_words(words: list[dict[str, Any]]) -> str:
    return " ".join(text for word in words if (text := _word_text(word))).strip()


def _line_tokens(line: str) -> list[str]:
    return [item for item in line.strip().split() if item]


def _segment_from_words(words: list[dict[str, Any]]) -> dict[str, Any] | None:
    timed_words = [word for word in words if "start" in word and "end" in word]
    text = _join_words(words)
    if not text or not timed_words:
        return None
    return {
        "text": text,
        "start": float(timed_words[0]["start"]),
        "end": float(timed_words[-1]["end"]),
        "words": words,
    }


def _fallback_groups_for_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in words:
        text = _word_text(word)
        if not text:
            continue
        current.append(word)
        normalized = text.strip("\"'()[]{}").lower()
        if SENTENCE_BREAK_RE.search(text) and len(current) >= 4:
            flush()
        elif MINOR_BREAK_RE.search(text) and len(current) >= WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS:
            flush()
        elif normalized in SOFT_BREAK_WORDS and len(current) >= WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS:
            flush()
        elif len(current) >= WHISPER_SEMANTIC_SEGMENT_MAX_WORDS:
            flush()

    flush()
    return groups


def _api_settings() -> dict[str, str] | None:
    try:
        rows = db.list_openai_api_keys()
    except Exception as exc:
        log.warning("semantic caption openai keys unavailable: %s", exc)
        return None

    keys: list[str] = []
    for row in rows:
        api_key = str(row.get("api_key") or "").strip()
        if api_key and api_key not in keys:
            keys.append(api_key)
    if not keys:
        return None

    first = rows[0]
    return {
        "base_url": str(first.get("base_url") or OPENAI_BASE_URL),
        "api_keys": ",".join(keys),
        "model": str(first.get("model") or OPENAI_MODEL),
        "timeout_seconds": str(OPENAI_TIMEOUT_SECONDS),
    }


def _api_keys(settings: dict[str, str]) -> list[str]:
    return [
        item.strip()
        for item in str(settings.get("api_keys") or "").replace("\n", ",").split(",")
        if item.strip()
    ]


def _system_prompt(language: str, *, strict: bool) -> str:
    strict_rule = (
        "\nThis is a retry after validation failed. Be stricter: copy the editable_zone text exactly, "
        "only insert line breaks between existing words."
        if strict
        else ""
    )
    return f"""You are a Semantic Caption Unit segmenter for ASR subtitles.

Your task is to optimize line breaks in the editable_zone only.

Hard rules:
1. Do not change, add, delete, translate, correct, or reorder any word.
2. Do not change punctuation, casing, spelling, abbreviations, or numbers.
3. Only add, remove, or move line breaks between existing words.
4. Return only the editable_zone text with line breaks. Do not return XML, JSON, explanations, or markdown.
5. The returned text, after replacing line breaks with spaces, must be exactly the original editable_zone text.

Caption goals:
1. Each line should be one complete semantic caption unit.
2. Prefer {WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS} words per line.
3. Keep most lines between 8 and 15 words.
4. Avoid lines over 20 words.
5. Never exceed {WHISPER_SEMANTIC_SEGMENT_MAX_WORDS} words unless no valid split exists.
6. Make lines easy to read as subtitles and easy to breathe for TTS.

Preferred break points:
1. Sentence endings.
2. Commas, semicolons, colons.
3. Coordinators such as and, but, or, because, while, although.
4. Clause boundaries such as who, which, that.
5. Introductory phrases and parenthetical phrases.

Do not split:
1. Person names, organization names, place names.
2. Time/date expressions and abbreviations such as U.S., U.K., Dr., Mr., Mrs., St.
3. Tight noun phrases and fixed expressions.

Language hint: {language or "unknown"}.{strict_rule}"""


def _user_prompt(left_context: str, editable_zone: str, right_context: str) -> str:
    return f"""<left_context>
{left_context}
</left_context>

<editable_zone>
{editable_zone}
</editable_zone>

<right_context>
{right_context}
</right_context>"""


def _validate_output(editable_words: list[dict[str, Any]], output: str) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    expected_tokens = [_word_text(word) for word in editable_words if _word_text(word)]
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    groups: list[list[dict[str, Any]]] = []
    cursor = 0

    for line in lines:
        tokens = _line_tokens(line)
        if not tokens:
            continue
        end = cursor + len(tokens)
        if expected_tokens[cursor:end] != tokens:
            return [], {
                "ok": False,
                "reason": "line_tokens_do_not_match_editable_words",
                "line": line,
                "cursor": cursor,
                "expected": expected_tokens[cursor:end],
                "actual": tokens,
            }
        groups.append(editable_words[cursor:end])
        cursor = end

    if cursor != len(expected_tokens):
        return [], {
            "ok": False,
            "reason": "output_did_not_consume_all_words",
            "consumed_words": cursor,
            "expected_words": len(expected_tokens),
        }

    overlong_lines = [
        {"line_index": index, "word_count": len(group)}
        for index, group in enumerate(groups)
        if len(group) > WHISPER_SEMANTIC_SEGMENT_MAX_WORDS
    ]
    if overlong_lines:
        return [], {
            "ok": False,
            "reason": "line_exceeds_max_words",
            "max_words": WHISPER_SEMANTIC_SEGMENT_MAX_WORDS,
            "overlong_lines": overlong_lines,
        }

    return groups, {
        "ok": True,
        "line_count": len(groups),
        "word_counts": [len(group) for group in groups],
        "overlong_lines": [],
    }


def _call_llm(
    settings: dict[str, str],
    api_key: str,
    system: str,
    user: str,
    *,
    temperature: float,
) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=settings["base_url"],
        timeout=float(settings.get("timeout_seconds") or OPENAI_TIMEOUT_SECONDS),
    )
    response = client.chat.completions.create(
        model=settings["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _append_summary(diagnostics: dict[str, Any], key: str, count: int = 1) -> None:
    summary = diagnostics.setdefault("summary", {})
    summary[key] = int(summary.get(key) or 0) + count


def _semantic_groups_for_chunk(
    chunk_index: int,
    all_words: list[dict[str, Any]],
    start: int,
    end: int,
    language: str,
    settings: dict[str, str],
    api_keys: list[str],
    diagnostics: dict[str, Any],
) -> list[list[dict[str, Any]]] | None:
    left_start = max(0, start - WHISPER_SEMANTIC_SEGMENT_CONTEXT_WORDS)
    right_end = min(len(all_words), end + WHISPER_SEMANTIC_SEGMENT_CONTEXT_WORDS)
    left_context = _join_words(all_words[left_start:start])
    editable_zone = _join_words(all_words[start:end])
    right_context = _join_words(all_words[end:right_end])

    chunk_log: dict[str, Any] = {
        "chunk_index": chunk_index,
        "word_start": start,
        "word_end": end,
        "left_context": left_context,
        "editable_zone": editable_zone,
        "right_context": right_context,
        "attempts": [],
    }
    diagnostics.setdefault("chunks", []).append(chunk_log)

    if not editable_zone:
        chunk_log["status"] = "skipped"
        chunk_log["reason"] = "empty_editable_zone"
        return None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        strict = attempt >= STRICT_ATTEMPT
        system = _system_prompt(language, strict=strict)
        user = _user_prompt(left_context, editable_zone, right_context)
        attempt_log: dict[str, Any] = {
            "attempt": attempt,
            "temperature": 0.0 if strict else 0.1,
            "strict": strict,
            "system_prompt": system,
            "user_prompt": user,
        }
        started_at = time.monotonic()
        try:
            raw = _call_llm(
                settings,
                api_keys[(attempt - 1) % len(api_keys)],
                system,
                user,
                temperature=float(attempt_log["temperature"]),
            )
            attempt_log["raw_output"] = raw
            groups, validation = _validate_output(all_words[start:end], raw)
            attempt_log["validation"] = validation
            attempt_log["duration_seconds"] = round(time.monotonic() - started_at, 3)
            chunk_log["attempts"].append(attempt_log)
            if groups and validation.get("ok"):
                chunk_log["status"] = "accepted"
                chunk_log["accepted_attempt"] = attempt
                _append_summary(diagnostics, "accepted_chunks")
                return groups
        except Exception as exc:
            attempt_log["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            attempt_log["duration_seconds"] = round(time.monotonic() - started_at, 3)
            chunk_log["attempts"].append(attempt_log)

    chunk_log["status"] = "fallback"
    chunk_log["reason"] = "all_attempts_failed"
    _append_summary(diagnostics, "fallback_chunks")
    return None


def segment_words_with_llm(
    words: list[dict[str, Any]],
    language: str,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]] | None:
    diagnostics.setdefault("enabled", WHISPER_SEMANTIC_SEGMENT_ENABLED)
    diagnostics.setdefault("config", {
        "min_words": WHISPER_SEMANTIC_SEGMENT_MIN_WORDS,
        "editable_words": WHISPER_SEMANTIC_SEGMENT_EDITABLE_WORDS,
        "context_words": WHISPER_SEMANTIC_SEGMENT_CONTEXT_WORDS,
        "target_words": WHISPER_SEMANTIC_SEGMENT_TARGET_WORDS,
        "max_words": WHISPER_SEMANTIC_SEGMENT_MAX_WORDS,
        "max_attempts": MAX_ATTEMPTS,
    })

    if not WHISPER_SEMANTIC_SEGMENT_ENABLED:
        _append_summary(diagnostics, "disabled_runs")
        return None

    usable_words = [word for word in words if _word_text(word) and "start" in word and "end" in word]
    if len(usable_words) < WHISPER_SEMANTIC_SEGMENT_MIN_WORDS:
        _append_summary(diagnostics, "skipped_short_runs")
        return None

    settings = _api_settings()
    if settings is None:
        diagnostics["api_status"] = "missing_openai_api_key"
        _append_summary(diagnostics, "missing_api_key_runs")
        return None

    api_keys = _api_keys(settings)
    if not api_keys:
        diagnostics["api_status"] = "missing_openai_api_key"
        _append_summary(diagnostics, "missing_api_key_runs")
        return None

    diagnostics["api"] = {
        "base_url": settings["base_url"],
        "model": settings["model"],
        "api_key_count": len(api_keys),
        "timeout_seconds": settings.get("timeout_seconds"),
    }

    all_groups: list[list[dict[str, Any]]] = []
    chunk_index = 0
    for start in range(0, len(usable_words), WHISPER_SEMANTIC_SEGMENT_EDITABLE_WORDS):
        end = min(len(usable_words), start + WHISPER_SEMANTIC_SEGMENT_EDITABLE_WORDS)
        groups = _semantic_groups_for_chunk(
            chunk_index,
            usable_words,
            start,
            end,
            language,
            settings,
            api_keys,
            diagnostics,
        )
        if groups is None:
            groups = _fallback_groups_for_words(usable_words[start:end])
            diagnostics.setdefault("chunk_rule_fallbacks", []).append(
                {
                    "chunk_index": chunk_index,
                    "method": "semantic_caption_rule",
                    "word_count": end - start,
                    "segments": len(groups),
                }
            )
            _append_summary(diagnostics, "rule_fallback_chunks")
        all_groups.extend(groups)
        chunk_index += 1

    segments = [segment for group in all_groups if (segment := _segment_from_words(group)) is not None]
    if not segments:
        _append_summary(diagnostics, "empty_result_runs")
        return None

    _append_summary(diagnostics, "accepted_runs")
    diagnostics["result"] = {
        "input_words": len(usable_words),
        "segments": len(segments),
        "word_counts": [len(segment.get("words") or []) for segment in segments],
    }
    return segments
