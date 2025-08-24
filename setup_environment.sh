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

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p results
mkdir -p mediapipe_output

# MediaPipe models are auto-downloaded on first use
echo "🤖 MediaPipe models will be auto-downloaded on first use"

# Test the installation
echo "🧪 Testing installation..."
python test_system.py

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the gait analysis system:"
echo "  source .venv/bin/activate"
echo "  python main_gait_analysis.py --help"
echo ""
echo "For more information, see README_TCN_Gait_Analysis.md"
