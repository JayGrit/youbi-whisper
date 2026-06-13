# whisper Plan

## Responsibility

`whisper` converts vocals audio into ASR JSON with timestamps. It owns Whisper
model loading and ASR language selection.

## Input Table

`whisper`

Required fields:

- `task_id`
- `audio_vocals_path`
- `status = 'ready'`

## Outputs

- `asr_json_path`

It copies `asr_json_path` into `translator.asr_json_path`.

## Polling

Poll one ready row every `POLL_INTERVAL_SECONDS`.

## Processing

1. Mark row `running`.
2. Validate vocals file exists.
3. Run Whisper with word timestamps.
4. Write normalized ASR JSON.
5. Mark whisper `success`.
6. Mark translator `ready`.

## Failure Handling

Mark whisper and task as `failed`. Keep ASR output if it was partially written
only when valid JSON can be parsed.

## Later Work

- Add language detection from source platform.
- Add model-size configuration per worker.
- Add ASR confidence metrics.

