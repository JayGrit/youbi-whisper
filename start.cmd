@echo off
setlocal

cd /d "%~dp0"

set "DEVICE=MY_HP"
set "YDBI_WHISPER_MODEL=base"
set "YDBI_WHISPER_DEVICE=cuda"
set "YDBI_WHISPER_DOWNLOAD_ROOT=%~dp0models\whisper"
set "YDBI_WHISPER_WORK_DIR=%~dp0work"

if not exist "%YDBI_WHISPER_DOWNLOAD_ROOT%" mkdir "%YDBI_WHISPER_DOWNLOAD_ROOT%"
if not exist "%YDBI_WHISPER_WORK_DIR%" mkdir "%YDBI_WHISPER_WORK_DIR%"

echo Starting ydbi-whisper...
echo Python env: D:\Money\youbi-speaker\.venv
echo Model: %YDBI_WHISPER_MODEL%
echo Runtime device: %YDBI_WHISPER_DEVICE%
echo Model cache: %YDBI_WHISPER_DOWNLOAD_ROOT%
echo Work dir: %YDBI_WHISPER_WORK_DIR%
echo.

D:\Money\youbi-speaker\.venv\Scripts\ydbi-whisper.exe

endlocal
