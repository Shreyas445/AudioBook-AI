@echo off
setlocal EnableExtensions
title NexGen AI Audiobook CLI
cd /d "%~dp0"

chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

if "%~1"=="" (
    echo Usage:
    echo   Drag and drop a .txt, .pdf, or .json file onto run.bat
    echo   OR run: run.bat story.pdf --provider "NVIDIA NIM API" --model "nvidia/nemotron-3.5-bolt-e4b"
    echo   OR launch the full Studio GUI with: run_gui.bat
    pause
    exit /b 0
)

python main.py %*
if errorlevel 1 pause
endlocal
