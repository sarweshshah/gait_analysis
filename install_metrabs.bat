@echo off
REM MeTRAbs Installation Script for Windows
REM ======================================

echo Installing MeTRAbs for Gait Analysis System
echo ===========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python and pip are available

REM Install PyTorch first (if not already installed)
echo 📦 Installing PyTorch...
pip install torch>=1.9.0

if errorlevel 1 (
    echo ❌ Failed to install PyTorch
    pause
    exit /b 1
) else (
    echo ✅ PyTorch installed successfully
)

REM Install MeTRAbs from GitHub
echo 📦 Installing MeTRAbs from GitHub...
pip install git+https://github.com/isarandi/metrabs.git

if errorlevel 1 (
    echo ❌ Failed to install MeTRAbs
    echo 💡 Try installing manually: pip install git+https://github.com/isarandi/metrabs.git
    pause
    exit /b 1
) else (
    echo ✅ MeTRAbs installed successfully
)

REM Test the installation
echo 🧪 Testing MeTRAbs installation...
python -c "import metrabs; print('✅ MeTRAbs imported successfully')"

if errorlevel 1 (
    echo ❌ MeTRAbs installation test failed
    pause
    exit /b 1
) else (
    echo.
    echo 🎉 MeTRAbs installation completed successfully!
    echo You can now use MeTRAbs in the gait analysis system.
    echo.
    echo Usage examples:
    echo   python scripts/pose_model_comparison.py --video data/video.mp4 --model metrabs
    echo   python usecases/gait_analysis/main_gait_analysis.py --videos data/video.mp4 --pose-model metrabs
    echo.
    pause
)
