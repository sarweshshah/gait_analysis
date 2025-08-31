# Installation Guide

This guide covers the installation of all dependencies for the Gait Analysis System, including both MediaPipe and MeTRAbs pose estimation models.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for MeTRAbs installation)
- CUDA-compatible GPU (recommended for MeTRAbs, optional for MediaPipe)

## Quick Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd gait_analysis
```

### 2. Install Core Dependencies

```bash
# Install all core dependencies
pip3 install -r requirements.txt
```

### 3. Install MeTRAbs (Optional)

MeTRAbs provides higher accuracy pose estimation but requires more computational resources.

#### Option A: Using Installation Scripts (Recommended)

**Linux/Mac:**
```bash
chmod +x install_metrabs.sh
./install_metrabs.sh
```

**Windows:**
```cmd
install_metrabs.bat
```

#### Option B: Manual Installation

```bash
# Install PyTorch first
pip3 install torch>=1.9.0

# Install MeTRAbs from GitHub
pip3 install git+https://github.com/isarandi/metrabs.git
```

## Detailed Installation

### Core Dependencies

The core dependencies are listed in `requirements.txt`:

- **Scientific Computing**: numpy, scipy, pandas
- **Machine Learning**: tensorflow, scikit-learn
- **Computer Vision**: mediapipe, opencv-python
- **Data Visualization**: matplotlib, seaborn
- **Image Processing**: Pillow, imutils
- **Utilities**: tqdm, pathlib2, argparse
- **Development**: pytest, black, flake8

### Pose Estimation Models

#### MediaPipe (Included in requirements.txt)

MediaPipe is automatically installed with the core dependencies and provides:
- Fast, real-time pose estimation
- Works well on CPU
- Lightweight and easy to use

#### MeTRAbs (Optional)

MeTRAbs provides higher accuracy but requires:
- PyTorch installation
- More computational resources
- GPU recommended for optimal performance

## Platform-Specific Instructions

### Linux/Mac

1. **Install Python dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Install MeTRAbs (optional):**
   ```bash
   ./install_metrabs.sh
   ```

3. **Verify installation:**
   ```bash
   python3 usecases/testing/test_pose_models.py --info
   ```

### Windows

1. **Install Python dependencies:**
   ```cmd
   pip3 install -r requirements.txt
   ```

2. **Install MeTRAbs (optional):**
   ```cmd
   install_metrabs.bat
   ```

3. **Verify installation:**
   ```cmd
   python3 usecases/testing/test_pose_models.py --info
   ```

### Docker (Alternative)

If you prefer using Docker:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install MeTRAbs
RUN pip3 install git+https://github.com/isarandi/metrabs.git

# Copy application code
COPY . .

# Set working directory
WORKDIR /app
```

## Verification

### Test Core Installation

```bash
# Test basic imports
python3 -c "
from core.pose_processor_manager import UnifiedPoseProcessor, PoseProcessorManager
print('✅ Core installation successful')
print('Available models:', list(PoseProcessorManager.get_available_models().keys()))
"
```

### Test Pose Models

```bash
# Test all available pose models
python3 usecases/testing/test_pose_models.py

# Test specific model
python3 usecases/testing/test_pose_models.py --model mediapipe
python3 usecases/testing/test_pose_models.py --model metrabs

# Compare models
python3 scripts/pose_model_comparison.py --info
```

### Test Complete System

```bash
# Run comprehensive system tests
python3 usecases/testing/test_system.py

# Test pose processor integration specifically
python3 usecases/testing/test_system.py --test-pose-processor
```

## Troubleshooting

### Common Issues

#### 1. MeTRAbs Installation Fails

**Error**: `No matching distribution found for metrabs>=0.1.0`

**Solution**: MeTRAbs is not available on PyPI. Use the installation scripts:
```bash
./install_metrabs.sh  # Linux/Mac
install_metrabs.bat   # Windows
```

#### 2. CUDA Out of Memory

**Error**: CUDA out of memory when using MeTRAbs

**Solutions**:
- Reduce batch size in configuration
- Use CPU instead of GPU
- Use smaller model (`metrabs_4x_512`)

#### 3. MediaPipe Import Error

**Error**: `ModuleNotFoundError: No module named 'mediapipe'`

**Solution**:
```bash
pip3 install mediapipe>=0.10.0
```

#### 4. PyTorch Installation Issues

**Error**: PyTorch installation fails

**Solution**: Install PyTorch from the official website:
```bash
# Visit: https://pytorch.org/get-started/locally/
# Choose your platform and CUDA version
```

### Performance Optimization

#### For Real-time Applications
- Use MediaPipe (faster, lighter)
- Use `model_complexity=0` for fastest processing
- Works well on CPU

#### For High Accuracy
- Use MeTRAbs with GPU
- Increase batch size for better GPU utilization
- Use test-time augmentation

#### For Research
- Compare both models using the comparison script
- Use MeTRAbs for offline processing
- Use MediaPipe for real-time applications

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv gait_analysis_env

# Activate virtual environment
source gait_analysis_env/bin/activate  # Linux/Mac
gait_analysis_env\Scripts\activate     # Windows

# Install dependencies
pip3 install -r requirements.txt
```

### Conda Environment

```bash
# Create conda environment
conda create -n gait_analysis python=3.9

# Activate environment
conda activate gait_analysis

# Install dependencies
pip3 install -r requirements.txt
```

## Next Steps

After successful installation:

1. **Test the system**: Run the test scripts to verify everything works
2. **Process videos**: Use the main analysis pipeline
3. **Compare models**: Use the comparison script to choose the best model for your use case
4. **Customize**: Modify configuration files for your specific needs

## Related Documentation

For more information about the project and its evolution:

- **Project Changelog**: [docs/README_Changelog.md](README_Changelog.md) - Complete project history and changes
- **MeTRAbs Integration**: [docs/README_MeTRAbs_Integration.md](README_MeTRAbs_Integration.md) - Detailed MeTRAbs guide
- **TCN System Documentation**: [docs/README_TCN_Gait_Analysis.md](README_TCN_Gait_Analysis.md) - Technical system documentation
- **Core Modules**: [core/README_CoreModules.md](../core/README_CoreModules.md) - Core system modules documentation

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the test output for specific error messages
3. Ensure all prerequisites are met
4. Try installing dependencies one by one to isolate issues
5. Check the GitHub repository for updated installation instructions
