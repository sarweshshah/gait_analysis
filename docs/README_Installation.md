# Installation Guide

This guide covers the installation of all dependencies for the Gait Analysis System, including MediaPipe pose estimation model.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for future model installations)
- CUDA-compatible GPU (optional, for future model support)

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

### 3. Verify Installation

```bash
# Test core imports
python3 -c "
from core.pose_processor_manager import UnifiedPoseProcessor, PoseProcessorManager
print('✅ Core installation successful')
print('Available models:', list(PoseProcessorManager.get_available_models().keys()))
"

# Test pose models
python3 usecases/testing/test_pose_models.py
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

## Platform-Specific Instructions

### Linux/Mac

1. **Install Python dependencies:**

   ```bash
   pip3 install -r requirements.txt
   ```

2. **Verify installation:**

   ```bash
   python3 usecases/testing/test_pose_models.py --info
   ```

### Windows

1. **Install Python dependencies:**

   ```cmd
   pip3 install -r requirements.txt
   ```

2. **Verify installation:**

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

#### 1. Model Installation Issues

**Error**: `No matching distribution found for some_model>=0.1.0`

**Solution**: Some models may not be available on PyPI. Check the model's documentation for installation instructions.

#### 2. CUDA Out of Memory

**Error**: CUDA out of memory when using GPU models

**Solutions**:

- Reduce batch size in configuration
- Use CPU instead of GPU
- Use smaller model variants

#### 3. MediaPipe Import Error

**Error**: `ModuleNotFoundError: No module named 'mediapipe'`

**Solution**:

```bash
pip3 install mediapipe>=0.10.0
```

#### 4. TensorFlow Import Error

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**: Install PyTorch from the official website:

```bash
# Visit: https://pytorch.org/get-started/locally/
# Or use pip:
pip3 install torch>=1.9.0
```

#### 5. OpenCV Import Error

**Error**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:

```bash
pip3 install opencv-python>=4.5.0
```

### Performance Optimization

#### For Real-time Applications

- Use MediaPipe (faster, lighter)
- Use `model_complexity=0` for fastest processing
- Works well on CPU

#### For High Accuracy

- Use GPU-accelerated models
- Increase batch size for better GPU utilization
- Use test-time augmentation

#### For Research

- Compare both models using the comparison script
- Use GPU-accelerated models for offline processing
- Use MediaPipe for real-time applications

## Configuration

### Default Configuration

The system uses configuration files in the `configs/` directory:

- **`configs/default.json`** - Default system parameters
- **`configs/gait_analysis.json`** - Configuration for pose models

### Environment Variables

You can override configuration using environment variables:

```bash
export GAIT_ANALYSIS_FPS=30.0
export GAIT_ANALYSIS_CONFIDENCE_THRESHOLD=0.3
```

## Development Setup

### Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip3 install -r requirements.txt
```

### Development Dependencies

```bash
# Install development dependencies
pip3 install -r requirements.txt
pip3 install black flake8 pytest
```

## Related Documentation

For more information about the project and its evolution:

- **Project Changelog**: [docs/README_Changelog.md](README_Changelog.md) - Complete project history and changes
- **TCN System Documentation**: [docs/README_TCN_Gait_Analysis.md](README_TCN_Gait_Analysis.md) - Technical system documentation
- **Core Modules**: [core/README_CoreModules.md](../core/README_CoreModules.md) - Core system modules documentation
- **Real-time Visualization**: [docs/README_RealTime_Visualization.md](README_RealTime_Visualization.md) - Real-time visualization guide

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the model comparison for your use case
3. Ensure proper hardware requirements
4. Consider using MediaPipe for real-time applications
5. Refer to the project changelog for recent changes and updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.
