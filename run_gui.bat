@echo off
REM Launcher script for MG Classifier GUI (Windows)

echo Starting MG Classifier GUI...
echo ==========================================

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Using Python: 
python --version
echo ==========================================

REM Get the directory where this script is located
cd /d "%~dp0"

REM Run the GUI
python mg_classifier_gui.py

REM Check exit status
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo Error: Application exited with an error
    echo ==========================================
    pause
    exit /b 1
)

