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

#### MediaPipe Performance Tuning

**Model Complexity Settings** (Most Important):

```python
# Fastest processing (lower accuracy)
model_complexity=0  # ~2x faster, ~5% accuracy loss

# Balanced performance (recommended)
model_complexity=1  # Default setting, good balance

# Highest accuracy (slower)
model_complexity=2  # ~2x slower, ~5% accuracy gain
```

**Confidence Thresholds**:

```python
# Faster processing (fewer detections)
min_detection_confidence=0.3   # Lower threshold, more detections
min_tracking_confidence=0.3    # Lower threshold, more tracking

# Higher accuracy (fewer false positives)
min_detection_confidence=0.7   # Higher threshold, fewer detections
min_tracking_confidence=0.7    # Higher threshold, fewer tracking
```

**Frame Processing Optimization**:

```python
# Process every nth frame for speed
frame_skip = 2  # Process every 2nd frame (2x faster)
frame_skip = 3  # Process every 3rd frame (3x faster)

# For real-time: frame_skip = 1 (process all frames)
# For analysis: frame_skip = 2-5 (faster processing)
```

#### Resolution and Input Optimization

**Input Resolution**:

```python
# Fast processing (lower resolution)
input_resolution = (256, 256)    # ~4x faster than 640x640
input_resolution = (320, 320)    # ~2.5x faster than 640x640

# Standard processing
input_resolution = (640, 640)    # Default, balanced

# High accuracy (higher resolution)
input_resolution = (1280, 1280)  # ~4x slower, higher accuracy
```

**Video Quality Settings**:

```python
# Lower quality for speed
fps = 15.0        # Process at 15 FPS instead of 30
bitrate = "500k"  # Lower video bitrate

# Standard quality
fps = 30.0        # Default processing rate
bitrate = "2M"    # Standard bitrate

# High quality for accuracy
fps = 60.0        # Process at 60 FPS for smooth motion
bitrate = "5M"    # Higher bitrate for quality
```

#### For Real-time Applications

- **Use MediaPipe** (faster, lighter, CPU-optimized)
- **Set `model_complexity=0`** for fastest processing
- **Use `frame_skip=1`** for real-time responsiveness
- **Lower input resolution** (256x256 or 320x320)
- **Lower confidence thresholds** (0.3-0.5) for more detections
- **Process at 15-30 FPS** depending on hardware

#### For High Accuracy

- **Use `model_complexity=2`** for highest precision
- **Higher input resolution** (640x640 or 1280x1280)
- **Higher confidence thresholds** (0.7-0.9) for fewer false positives
- **Process at 30-60 FPS** for smooth motion capture
- **Use GPU acceleration** when available (TensorFlow backend)

#### For Research and Analysis

- **Compare models** using the comparison script
- **Use GPU-accelerated models** for offline processing
- **Use MediaPipe** for real-time applications
- **Experiment with different settings** for optimal results
- **Profile performance** using the built-in timing metrics

#### Hardware-Specific Optimization

**CPU Optimization**:

```python
# Optimize for CPU processing
model_complexity = 0 or 1
input_resolution = (320, 320) or (640, 640)
frame_skip = 2  # Reduce CPU load
```

**GPU Optimization**:

```python
# Optimize for GPU processing
model_complexity = 1 or 2
input_resolution = (640, 640) or (1280, 1280)
frame_skip = 1  # Maximize GPU utilization
batch_size = 4  # Increase for better GPU efficiency
```

**Memory Optimization**:

```python
# Reduce memory usage
input_resolution = (256, 256)
model_complexity = 0
frame_skip = 3
min_detection_confidence = 0.5  # Fewer detections = less memory
```

#### Performance Monitoring

**Built-in Metrics**:

```python
# The system provides these performance metrics:
- Processing time per frame
- Memory usage
- CPU/GPU utilization
- FPS achieved
- Accuracy metrics
```

**Performance Testing**:

```bash
# Test different configurations
python -m usecases.gait_analysis.main_gait_analysis \
    --videos videos/raw/sample.mp4 \
    --pose-detection-only \
    --config configs/performance_test.json

# Compare performance across models
python scripts/pose_model_comparison.py \
    --video videos/raw/sample.mp4 \
    --compare
```

#### Recommended Configurations

**Fast Real-time Processing**:

```json
{
  "model_complexity": 0,
  "min_detection_confidence": 0.3,
  "min_tracking_confidence": 0.3,
  "input_resolution": [256, 256],
  "frame_skip": 1,
  "fps": 30.0
}
```

**Balanced Performance**:

```json
{
  "model_complexity": 1,
  "min_detection_confidence": 0.5,
  "min_tracking_confidence": 0.5,
  "input_resolution": [640, 640],
  "frame_skip": 1,
  "fps": 30.0
}
```

#### For Research (High Accuracy Processing)

```json
{
  "model_complexity": 2,
  "min_detection_confidence": 0.7,
  "min_tracking_confidence": 0.7,
  "input_resolution": [1280, 1280],
  "frame_skip": 1,
  "fps": 60.0
}
```

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
