#!/bin/bash

# Gait Analysis System Environment Setup Script
# ============================================

set -e  # Exit on any error

echo "🚀 Setting up Gait Analysis System Environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Found Python $PYTHON_VERSION"

# Check Python version compatibility
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    echo "❌ Python 3.7+ is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first (without conflicts)
echo "📚 Installing core dependencies..."
pip install numpy scipy pandas matplotlib seaborn pillow tqdm pathlib2 argparse

# Install PyTorch
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio

# Install scikit-learn
echo "🤖 Installing scikit-learn..."
pip install scikit-learn

# Install OpenCV
echo "👁️  Installing OpenCV..."
pip install opencv-python

# Install TensorFlow with specific version to avoid JAX conflicts
echo "🧠 Installing TensorFlow (compatible version)..."
pip install "tensorflow>=2.14.0,<2.15.0"

# Install MediaPipe without JAX dependencies
echo "📱 Installing MediaPipe..."
pip install mediapipe --no-deps

# Install remaining MediaPipe dependencies manually (excluding JAX)
echo "📦 Installing MediaPipe dependencies..."
pip install absl-py attrs flatbuffers protobuf sounddevice sentencepiece

# Install development tools
echo "🛠️  Installing development tools..."
pip install pytest black flake8

# Install remaining utilities
echo "🔧 Installing utilities..."
pip install imutils

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p videos
mkdir -p videos/raw
mkdir -p videos/sneak
mkdir -p results
mkdir -p mediapipe_output
mkdir -p outputs/gait_analysis
mkdir -p outputs/mediapipe
mkdir -p outputs/test_results
mkdir -p outputs/logs
mkdir -p outputs/models
mkdir -p outputs/visualizations

# MediaPipe models are auto-downloaded on first use
echo "🤖 MediaPipe models will be auto-downloaded on first use"

# Test the installation
echo "🧪 Testing installation..."
python usecases/testing/test_system.py

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the gait analysis system:"
echo "  source .venv/bin/activate"
echo "  python -m usecases.gait_analysis.main_gait_analysis --help"
echo ""
echo "For more information, see README_TCN_Gait_Analysis.md"
