@echo off
setlocal EnableExtensions
title NexGen AI Cinema Audiobook Studio
cd /d "%~dp0"

:: Force UTF-8 console & Python I/O so Unicode books and logs never fail on Windows
chcp 65001 >nul 2>&1
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo ==============================================================
echo   NexGen AI Cinema Audiobook Studio - Launching GUI...
echo ==============================================================

:: Activate local virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo [INFO] No .venv folder detected; using system Python.
)

:: Verify Python availability
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in .venv or system PATH!
    echo Please install Python 3.10+ or create .venv first.
    pause
    exit /b 1
)

:: Launch the Studio GUI (opens automatically at http://127.0.0.1:7860)
python app.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Application exited with code %EXIT_CODE%. Review the messages above.
    pause
)

endlocal
exit /b %EXIT_CODE%
