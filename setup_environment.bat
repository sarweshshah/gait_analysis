@echo off
REM Gait Analysis System Environment Setup Script for Windows
REM ========================================================

echo 🚀 Setting up Gait Analysis System Environment...

REM Check if Python 3 is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.7+ first.
    pause
    exit /b 1
)

REM Remove existing virtual environment if it exists
if exist ".venv" (
    echo 🗑️  Removing existing virtual environment...
    rmdir /s /q .venv
)

REM Create new virtual environment
echo 📦 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir "data"
if not exist "results" mkdir "results"
if not exist "mediapipe_output" mkdir "mediapipe_output"

REM MediaPipe models are auto-downloaded on first use
echo 🤖 MediaPipe models will be auto-downloaded on first use

REM Test the installation
echo 🧪 Testing installation...
python test_system.py

echo.
echo 🎉 Environment setup completed successfully!
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate.bat
echo.
echo To run the gait analysis system:
echo   .venv\Scripts\activate.bat
echo   python main_gait_analysis.py --help
echo.
echo For more information, see README_TCN_Gait_Analysis.md
pause
