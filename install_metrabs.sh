#!/bin/bash
# MeTRAbs Installation Script
# ===========================

echo "Installing MeTRAbs for Gait Analysis System"
echo "==========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python3 and pip3 are available"

# Install PyTorch first (if not already installed)
echo "📦 Installing PyTorch..."
pip3 install torch>=1.9.0

if [ $? -eq 0 ]; then
    echo "✅ PyTorch installed successfully"
else
    echo "❌ Failed to install PyTorch"
    exit 1
fi

# Install MeTRAbs from GitHub
echo "📦 Installing MeTRAbs from GitHub..."
pip3 install git+https://github.com/isarandi/metrabs.git

if [ $? -eq 0 ]; then
    echo "✅ MeTRAbs installed successfully"
else
    echo "❌ Failed to install MeTRAbs"
    echo "💡 Try installing manually: pip3 install git+https://github.com/isarandi/metrabs.git"
    exit 1
fi

# Test the installation
echo "🧪 Testing MeTRAbs installation..."
python3 -c "
try:
    import metrabs
    print('✅ MeTRAbs imported successfully')
    print(f'Version: {metrabs.__version__ if hasattr(metrabs, \"__version__\") else \"Unknown\"}')
except ImportError as e:
    print(f'❌ Failed to import MeTRAbs: {e}')
    exit(1)
except Exception as e:
    print(f'⚠ Warning: {e}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 MeTRAbs installation completed successfully!"
    echo "You can now use MeTRAbs in the gait analysis system."
    echo ""
    echo "Usage examples:"
    echo "  python3 scripts/pose_model_comparison.py --video data/video.mp4 --model metrabs"
    echo "  python3 usecases/gait_analysis/main_gait_analysis.py --videos data/video.mp4 --pose-model metrabs"
else
    echo "❌ MeTRAbs installation test failed"
    exit 1
fi
