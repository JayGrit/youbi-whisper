@echo off
setlocal

cd /d "%~dp0"

if not defined DEVICE set "DEVICE=MY_HP"
set "YDBI_WHISPER_MODEL_ROOT=%~dp0models"
set "YDBI_WHISPER_DOWNLOAD_ROOT=%~dp0models\openai-whisper"
set "YDBI_WHISPERX_ALIGN_MODEL_DIR=%~dp0models\align"
set "YDBI_WHISPER_WORK_DIR=%~dp0work"
set "TORCH_HOME=%~dp0models\torch"
set "NLTK_DATA=%~dp0models\nltk"
set "HF_HOME=%~dp0models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0models\huggingface\hub"
set "TRANSFORMERS_CACHE=%~dp0models\huggingface\transformers"
set "XDG_CACHE_HOME=%~dp0models\cache"

for %%D in (
  "%YDBI_WHISPER_MODEL_ROOT%"
  "%YDBI_WHISPER_DOWNLOAD_ROOT%"
  "%YDBI_WHISPERX_ALIGN_MODEL_DIR%"
  "%YDBI_WHISPER_WORK_DIR%"
  "%TORCH_HOME%"
  "%NLTK_DATA%"
  "%HF_HOME%"
  "%HUGGINGFACE_HUB_CACHE%"
  "%TRANSFORMERS_CACHE%"
  "%XDG_CACHE_HOME%"
) do if not exist "%%~D" mkdir "%%~D"

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.12"
)
if not defined PYTHON_CMD (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
  )
)
if not defined PYTHON_CMD (
  echo Python 3.12+ was not found. Install Python, then rerun start.cmd.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 exit /b 1
)

echo Installing Python dependencies...
".venv\Scripts\python.exe" -m pip install -U pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b 1

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found in PATH. Install ffmpeg before processing audio.
  exit /b 1
)

echo Starting ydbi-whisper...
echo Operator: %DEVICE%
echo Model: large-v3-turbo
echo Runtime device: auto
echo Model root: %YDBI_WHISPER_MODEL_ROOT%
echo Work dir: %YDBI_WHISPER_WORK_DIR%
echo Auto update: git pull every 60 seconds
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$Root = (Get-Location).Path;" ^
  "$Python = Join-Path $Root '.venv\Scripts\python.exe';" ^
  "function Invoke-DependencyInstall { Write-Host 'Installing Python dependencies after update...'; & $Python -m pip install -e .; if ($LASTEXITCODE -ne 0) { throw 'pip install -e . failed' } }" ^
  "function Start-Whisper { Write-Host 'Starting ydbi-whisper...'; return Start-Process -FilePath $Python -ArgumentList @('-m', 'ydbi_whisper.main') -WorkingDirectory $Root -PassThru -NoNewWindow }" ^
  "function Stop-Whisper($Process) { if ($Process -and -not $Process.HasExited) { Write-Host ('Stopping ydbi-whisper process ' + $Process.Id + '...'); Stop-Process -Id $Process.Id -Force; Wait-Process -Id $Process.Id -Timeout 30 -ErrorAction SilentlyContinue } }" ^
  "function Get-Head { try { return (& git -C $Root rev-parse HEAD 2>$null).Trim() } catch { return '' } }" ^
  "function Update-Repository { $Before = Get-Head; if (-not $Before) { Write-Warning 'Not a git repository or git is unavailable; skipping auto update.'; return $false }; Write-Host 'Checking for updates...'; & git -C $Root pull --ff-only; if ($LASTEXITCODE -ne 0) { Write-Warning 'git pull failed; keeping current process running.'; return $false }; $After = Get-Head; return ($After -and $After -ne $Before) }" ^
  "$Process = Start-Whisper;" ^
  "try { while ($true) { Start-Sleep -Seconds 60; if ($Process.HasExited) { Write-Warning ('ydbi-whisper exited with code ' + $Process.ExitCode + '; restarting.'); $Process = Start-Whisper; continue }; if (Update-Repository) { Write-Host 'Repository updated; restarting ydbi-whisper.'; Stop-Whisper $Process; Invoke-DependencyInstall; $Process = Start-Whisper } } } finally { Stop-Whisper $Process }"

endlocal
