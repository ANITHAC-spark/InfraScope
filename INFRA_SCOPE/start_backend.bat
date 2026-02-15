@echo off
REM InfraScope Quick Start Script (Windows)
REM This script sets up and runs the entire application

echo.
echo ================================================
echo   InfraScope - Quick Start Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python found: 
python --version

REM Navigate to backend
cd backend

REM Check if venv exists
if not exist "venv" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [2/5] Virtual environment already exists
)

REM Activate venv
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [4/5] Installing dependencies (this may take a few minutes)...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Start backend server
echo [5/5] Starting InfraScope Backend Server...
echo.
echo ================================================
echo   Server starting on http://localhost:5000
echo   Press Ctrl+C to stop
echo ================================================
echo.

python app.py

if errorlevel 1 (
    echo ERROR: Failed to start server
    pause
    exit /b 1
)

pause
