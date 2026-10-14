@echo off
setlocal enabledelayedexpansion
title Project-RAG_LLM Enterprise Studio

cd /d "%~dp0"

echo ====================================================================
echo   Project-RAG_LLM Enterprise Studio
echo   Titanium Obsidian ^& Emerald AI Knowledge Platform
echo ====================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please verify your Python environment before launching.
    echo.
    pause
    exit /b 1
)

:: Check if server is already running on port 8000
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Enterprise Studio backend is already running on http://127.0.0.1:8000
    echo [INFO] Opening Enterprise Studio in your default browser...
    start "" http://127.0.0.1:8000
    echo.
    echo [READY] Studio is open. You may close this launcher window.
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)

echo [INFO] Starting FastAPI backend on http://127.0.0.1:8000...
echo [INFO] Opening Enterprise Studio in your default browser in 2 seconds...
echo [INFO] Press Ctrl+C in this console window to gracefully stop the server.
echo.

:: Launch default browser after 2.5 seconds in background
start /b "" cmd /c "ping 127.0.0.1 -n 3 >nul && start """" http://127.0.0.1:8000"

:: Start Uvicorn in foreground
".venv\Scripts\python.exe" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Uvicorn server exited with error code %ERRORLEVEL%.
    pause
)
