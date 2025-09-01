@echo off
REM Gait Analysis System Environment Setup Script (Windows)
REM ======================================================

echo 🚀 Setting up Gait Analysis System Environment...

REM Check if Python 3 is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3 is not installed. Please install Python 3.7+ first.
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Found Python %PYTHON_VERSION%

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

REM Install core dependencies first (without conflicts)
echo 📚 Installing core dependencies...
pip install numpy scipy pandas matplotlib seaborn pillow tqdm pathlib2 argparse

REM Install PyTorch
echo 🔥 Installing PyTorch...
pip install torch torchvision torchaudio

REM Install scikit-learn
echo 🤖 Installing scikit-learn...
pip install scikit-learn

REM Install OpenCV
echo 👁️  Installing OpenCV...
pip install opencv-python

REM Install TensorFlow with specific version to avoid JAX conflicts
echo 🧠 Installing TensorFlow (compatible version)...
pip install "tensorflow>=2.14.0,<2.15.0"

REM Install MediaPipe without JAX dependencies
echo 📱 Installing MediaPipe...
pip install mediapipe --no-deps

REM Install remaining MediaPipe dependencies manually (excluding JAX)
echo 📦 Installing MediaPipe dependencies...
pip install absl-py attrs flatbuffers protobuf sounddevice sentencepiece

REM Install development tools
echo 🛠️  Installing development tools...
pip install pytest black flake8

REM Install remaining utilities
echo 🔧 Installing utilities...
pip install imutils

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "videos" mkdir videos
if not exist "videos\raw" mkdir videos\raw
if not exist "videos\sneak" mkdir videos\sneak
if not exist "results" mkdir results
if not exist "mediapipe_output" mkdir mediapipe_output
if not exist "outputs" mkdir outputs
if not exist "outputs\gait_analysis" mkdir outputs\gait_analysis
if not exist "outputs\mediapipe" mkdir outputs\mediapipe
if not exist "outputs\test_results" mkdir outputs\test_results
if not exist "outputs\logs" mkdir outputs\logs
if not exist "outputs\models" mkdir outputs\models
if not exist "outputs\visualizations" mkdir outputs\visualizations

REM MediaPipe models are auto-downloaded on first use
echo 🤖 MediaPipe models will be auto-downloaded on first use

REM Test the installation
echo 🧪 Testing installation...
python usecases\testing\test_system.py

echo.
echo 🎉 Environment setup completed successfully!
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate.bat
echo.
echo To run the gait analysis system:
echo   .venv\Scripts\activate.bat
echo   python -m usecases.gait_analysis.main_gait_analysis --help
echo.
echo For more information, see README_TCN_Gait_Analysis.md

pause
