@echo off
rem HeadVoice launcher for Windows: sets up on first run, then starts the app in the browser.
title HeadVoice
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, the tool that manages Python for HeadVoice...
  powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Preparing HeadVoice. The first start takes a few minutes.
uv sync --no-dev
if errorlevel 1 goto failed

if not exist "models\lbfmodel.yaml" (
  echo Downloading the face models...
  uv run python scripts\get_models.py
  if errorlevel 1 goto failed
)

echo Starting HeadVoice. Your browser will open in a moment. Keep this window open while you use it.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8765/"
uv run uvicorn headvoice.app.server:app --port 8765
goto :eof

:failed
echo.
echo Setup did not finish. Check your internet connection and try again.
pause
