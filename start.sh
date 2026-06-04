#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

: "${DEVICE:=Macbook Air M4}"
export DEVICE
export YDBI_WHISPER_MODEL_ROOT="${YDBI_WHISPER_MODEL_ROOT:-$SCRIPT_DIR/models}"
export YDBI_WHISPER_DOWNLOAD_ROOT="${YDBI_WHISPER_DOWNLOAD_ROOT:-$YDBI_WHISPER_MODEL_ROOT/openai-whisper}"
export YDBI_WHISPERX_ALIGN_MODEL_DIR="${YDBI_WHISPERX_ALIGN_MODEL_DIR:-$YDBI_WHISPER_MODEL_ROOT/align}"
export YDBI_WHISPER_WORK_DIR="${YDBI_WHISPER_WORK_DIR:-$SCRIPT_DIR/work}"
export TORCH_HOME="${TORCH_HOME:-$YDBI_WHISPER_MODEL_ROOT/torch}"
export NLTK_DATA="${NLTK_DATA:-$YDBI_WHISPER_MODEL_ROOT/nltk}"
export HF_HOME="${HF_HOME:-$YDBI_WHISPER_MODEL_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$YDBI_WHISPER_MODEL_ROOT/cache}"

mkdir -p \
  "$YDBI_WHISPER_MODEL_ROOT" \
  "$YDBI_WHISPER_DOWNLOAD_ROOT" \
  "$YDBI_WHISPERX_ALIGN_MODEL_DIR" \
  "$YDBI_WHISPER_WORK_DIR" \
  "$TORCH_HOME" \
  "$NLTK_DATA" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME"

PYTHON_CMD=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
    PYTHON_CMD=$candidate
    break
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "Python 3.12+ was not found. Install Python, then rerun ./start.sh." >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating Python virtual environment..."
  "$PYTHON_CMD" -m venv .venv
fi

echo "Installing Python dependencies..."
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found in PATH. Install ffmpeg before processing audio." >&2
  exit 1
fi

echo "Starting ydbi-whisper..."
echo "Operator: $DEVICE"
echo "Model: large-v3-turbo"
echo "Runtime device: auto"
echo "Model root: $YDBI_WHISPER_MODEL_ROOT"
echo "Work dir: $YDBI_WHISPER_WORK_DIR"
echo

exec .venv/bin/python -m ydbi_whisper.main
