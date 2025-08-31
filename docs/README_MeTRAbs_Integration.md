# MeTRAbs Integration for Gait Analysis

This document describes the integration of MeTRAbs pose estimation model alongside the existing MediaPipe pipeline, providing users with multiple pose estimation options for gait analysis.

## Overview

The gait analysis system now supports two pose estimation models:

1. **MediaPipe Pose** - Lightweight, real-time pose estimation
2. **MeTRAbs** - High-accuracy pose estimation with test-time augmentation

Both models are integrated through a unified interface, allowing easy switching and comparison.

## Model Comparison

| Feature | MediaPipe | MeTRAbs |
|---------|-----------|---------|
| **Speed** | Fast (real-time) | Slower (batch processing) |
| **Accuracy** | Good | Higher |
| **Resource Usage** | Low | High |
| **GPU Required** | No | Recommended |
| **Landmarks** | 33 | 17 (COCO format) |
| **Output Format** | BODY_25 compatible | BODY_25 compatible |
| **Best For** | Real-time, mobile | High accuracy, research |

## Installation

### Prerequisites

1. Python 3.8+
2. CUDA-compatible GPU (recommended for MeTRAbs)
3. PyTorch

### Install Dependencies

```bash
# Install core dependencies
pip3 install -r requirements.txt

# Install MeTRAbs (choose one method):
# Method 1: Use the installation script
./install_metrabs.sh  # Linux/Mac
install_metrabs.bat   # Windows

# Method 2: Install manually
pip3 install torch>=1.9.0
pip3 install git+https://github.com/isarandi/metrabs.git
```

## Usage

### 1. Basic Usage with Model Selection

```python
from core.pose_processor_manager import UnifiedPoseProcessor

# Use MediaPipe (default)
processor = UnifiedPoseProcessor(model_type='mediapipe')
success = processor.process_video('path/to/video.mp4')

# Use MeTRAbs
processor = UnifiedPoseProcessor(model_type='metrabs')
success = processor.process_video('path/to/video.mp4')
```

### 2. Command Line Interface

```bash
# Use MediaPipe (default)
python3 usecases/gait_analysis/main_gait_analysis.py --videos data/video.mp4 --pose-model mediapipe

# Use MeTRAbs
python3 usecases/gait_analysis/main_gait_analysis.py --videos data/video.mp4 --pose-model metrabs

# Compare both models
python3 scripts/pose_model_comparison.py --video data/video.mp4 --compare
```

### 3. Model Switching

```python
from core.pose_processor_manager import UnifiedPoseProcessor

# Start with MediaPipe
processor = UnifiedPoseProcessor(model_type='mediapipe')
processor.process_video('video1.mp4')

# Switch to MeTRAbs
processor.switch_model('metrabs')
processor.process_video('video2.mp4')
```

### 4. Model Comparison

```python
from core.pose_processor_manager import UnifiedPoseProcessor

processor = UnifiedPoseProcessor()

# Compare all available models
results = processor.compare_models('path/to/video.mp4')

for model_type, result in results.items():
    print(f"{model_type}: {result['processing_time']:.2f}s")
```

## Configuration

### MediaPipe Configuration

```json
{
  "pose_model": "mediapipe",
  "mediapipe": {
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
  }
}
```

### MeTRAbs Configuration

```json
{
  "pose_model": "metrabs",
  "metrabs": {
    "model_name": "metrabs_4x_512",
    "device": "auto",
    "batch_size": 1,
    "num_aug": 1
  }
}
```

### Available MeTRAbs Models

- `metrabs_4x_512` - Faster, lower resolution
- `metrabs_4x_1024` - Slower, higher resolution

## Advanced Features

### 1. Test-Time Augmentation (MeTRAbs)

MeTRAbs supports test-time augmentation for improved accuracy:

```python
processor = UnifiedPoseProcessor(
    model_type='metrabs',
    num_aug=4  # Number of augmentations
)
```

### 2. Batch Processing (MeTRAbs)

Process multiple frames at once for better GPU utilization:

```python
processor = UnifiedPoseProcessor(
    model_type='metrabs',
    batch_size=4  # Process 4 frames at once
)
```

### 3. Device Selection

```python
# Auto-detect (recommended)
processor = UnifiedPoseProcessor(model_type='metrabs', device='auto')

# Force CPU
processor = UnifiedPoseProcessor(model_type='metrabs', device='cpu')

# Force GPU
processor = UnifiedPoseProcessor(model_type='metrabs', device='cuda')
```

## Performance Optimization

### For MediaPipe
- Use `model_complexity=0` for fastest processing
- Lower confidence thresholds for more detections
- Works well on CPU

### For MeTRAbs
- Use GPU for optimal performance
- Increase batch size for better GPU utilization
- Use test-time augmentation for better accuracy
- Consider using `metrabs_4x_512` for speed vs `metrabs_4x_1024` for accuracy

## Output Format

Both models output data in the same BODY_25 format for compatibility:

```json
{
  "frame_number": 0,
  "timestamp": 0.0,
  "people": [{
    "person_id": [0],
    "pose_keypoints_2d": [x1, y1, conf1, x2, y2, conf2, ...],
    "face_keypoints_2d": [],
    "hand_left_keypoints_2d": [],
    "hand_right_keypoints_2d": [],
    "pose_keypoints_3d": [],
    "face_keypoints_3d": [],
    "hand_left_keypoints_3d": [],
    "hand_right_keypoints_3d": []
  }]
}
```

## Troubleshooting

### Common Issues

1. **MeTRAbs Import Error**
   ```bash
   # Use the installation script
./install_metrabs.sh  # Linux/Mac
install_metrabs.bat   # Windows

# Or install manually
pip3 install torch>=1.9.0
pip3 install git+https://github.com/isarandi/metrabs.git
   ```

2. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU instead of GPU
   - Use smaller model (`metrabs_4x_512`)

3. **Slow Processing**
   - Use GPU for MeTRAbs
   - Increase batch size
   - Use MediaPipe for real-time applications

### Performance Tips

1. **For Real-time Applications**: Use MediaPipe
2. **For High Accuracy**: Use MeTRAbs with GPU
3. **For Research**: Compare both models
4. **For Mobile/Edge**: Use MediaPipe

## Examples

### Example 1: Basic Processing

```python
from core.pose_processor_manager import UnifiedPoseProcessor

# Process with MediaPipe
with UnifiedPoseProcessor(model_type='mediapipe') as processor:
    success = processor.process_video('walking.mp4')
    print(f"MediaPipe processing: {'Success' if success else 'Failed'}")

# Process with MeTRAbs
with UnifiedPoseProcessor(model_type='metrabs') as processor:
    success = processor.process_video('walking.mp4')
    print(f"MeTRAbs processing: {'Success' if success else 'Failed'}")
```

### Example 2: Model Comparison

```python
from core.pose_processor_manager import UnifiedPoseProcessor

processor = UnifiedPoseProcessor()

# Compare models
results = processor.compare_models('walking.mp4')

for model, result in results.items():
    if result['success']:
        print(f"{model}: {result['processing_time']:.2f}s")
    else:
        print(f"{model}: Failed - {result.get('error', 'Unknown error')}")
```

### Example 3: Configuration File

```json
{
  "pose_model": "metrabs",
  "pose_config": {
    "model_name": "metrabs_4x_1024",
    "device": "cuda",
    "batch_size": 4,
    "num_aug": 2
  },
  "fps": 30.0,
  "enable_gait_analysis": true
}
```

## Integration with Existing Pipeline

The MeTRAbs integration is fully compatible with the existing gait analysis pipeline:

1. **Data Preprocessing**: Works with existing `GaitDataPreprocessor`
2. **Model Training**: Compatible with TCN models
3. **Gait Events**: Supports rule-based event detection
4. **Visualization**: Works with existing visualization tools

## Future Enhancements

1. **Additional Models**: Support for more pose estimation models
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Adaptive Selection**: Automatically choose best model based on video characteristics
4. **Real-time Switching**: Switch models during processing based on performance

## Related Documentation

For more information about the project and its evolution:

- **Project Changelog**: [docs/README_Changelog.md](README_Changelog.md) - Complete project history and changes
- **Installation Guide**: [docs/README_Installation.md](README_Installation.md) - Comprehensive installation instructions
- **TCN System Documentation**: [docs/README_TCN_Gait_Analysis.md](README_TCN_Gait_Analysis.md) - Technical system documentation
- **Core Modules**: [core/README_CoreModules.md](../core/README_CoreModules.md) - Core system modules documentation

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the model comparison for your use case
3. Ensure proper hardware requirements for MeTRAbs
4. Consider using MediaPipe for real-time applications
5. Refer to the project changelog for recent changes and updates
