# Core System Modules

This document describes the core modules of the Gait Analysis System, providing a comprehensive overview of the system architecture and implementation details.

## Overview

The core modules provide the foundation for the gait analysis system, implementing:

- **Pose Estimation**: Unified pose processor manager supporting multiple pose estimation backends
- **Data Processing**: Advanced data preprocessing with gap-filling, filtering, and normalization
- **Machine Learning**: Temporal Convolutional Network (TCN) architecture for gait analysis
- **Training Pipeline**: Cross-validation training with comprehensive evaluation
- **Utilities**: Configuration management and logging systems

## Module Structure

### Pose Estimation

- **`pose_processor_manager.py`** - Unified pose processor manager supporting multiple pose estimation backends
- **`mediapipe_integration.py`** - MediaPipe pose estimation integration with BODY_25 keypoint mapping

### Data Processing

- **`gait_data_preprocessing.py`** - Advanced data preprocessing with gap-filling, filtering, and normalization

### Machine Learning

- **`tcn_gait_model.py`** - Temporal Convolutional Network architecture for gait analysis
- **`gait_training.py`** - Training pipeline with cross-validation and comprehensive evaluation

### Utilities

- **`utils/config.py`** - Configuration management system
- **`utils/logging_config.py`** - Centralized logging configuration
- **`utils/constants.py`** - Core constants and enums

## Detailed Module Documentation

### pose_processor_manager.py

**Purpose**: Provides a unified interface for different pose estimation models with easy switching between different pose estimation backends.

**Key Features**:

- **UnifiedPoseProcessor**: Single interface for all pose models
- **PoseProcessorManager**: Manager class for creating pose processors
- **PoseProcessor**: Abstract base class for pose processors
- Model switching capabilities
- Consistent API regardless of underlying model
- Support for multiple pose estimation backends

**Usage**:

```python
from core.pose_processor_manager import UnifiedPoseProcessor

# Use MediaPipe (default)
processor = UnifiedPoseProcessor(model_type='mediapipe')
pose_data = processor.process_video("video.mp4")

# Switch to another model (when available)
processor.switch_model('other_model')
pose_data = processor.process_video("video.mp4")
```

### mediapipe_integration.py

**Purpose**: Handles video processing with MediaPipe pose estimation and converts output to BODY_25 format.

**Key Features**:

- MediaPipe pose estimation with 33 landmarks
- BODY_25 keypoint mapping (25 keypoints including foot landmarks)
- Confidence-based filtering
- JSON output management
- Batch video processing

**Usage**:

```python
from core.mediapipe_integration import MediaPipeProcessor

processor = MediaPipeProcessor()
pose_data = processor.process_video("video.mp4")
```

### gait_data_preprocessing.py

**Purpose**: Advanced data preprocessing for gait analysis, including gap-filling, filtering, and feature extraction.

**Key Features**:

- Gap-filling using cubic spline interpolation
- Butterworth low-pass filtering (6 Hz cutoff)
- Keypoint normalization and standardization
- Feature extraction for gait analysis
- Confidence-based filtering
- BODY_25 keypoint format support

**Usage**:

```python
from core.gait_data_preprocessing import GaitDataPreprocessor

preprocessor = GaitDataPreprocessor(
    confidence_threshold=0.3,
    filter_cutoff=6.0,
    window_size=30
)

processed_data = preprocessor.preprocess_pose_data(pose_data)
```

### tcn_gait_model.py

**Purpose**: Implements Temporal Convolutional Network architecture optimized for gait analysis tasks.

**Key Features**:

- Dilated causal convolutions for temporal modeling
- Residual connections and batch normalization
- Support for both phase and event detection
- Configurable architecture parameters
- TensorFlow/Keras implementation

**Usage**:

```python
from core.tcn_gait_model import TCNGaitModel

model = TCNGaitModel(
    num_classes=4,
    num_filters=64,
    kernel_size=3,
    num_blocks=4
)
```

### gait_training.py

**Purpose**: Comprehensive training pipeline with cross-validation and evaluation for gait analysis models.

**Key Features**:

- Stratified k-fold cross-validation
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- Model checkpointing and result visualization
- Support for multiple task types

**Usage**:

```python
from core.gait_training import GaitTrainer

trainer = GaitTrainer(
    data_preprocessor=preprocessor,
    model_config=model_config,
    task_type='phase_detection'
)

results = trainer.train_and_evaluate(train_data, val_data)
```

### utils/config.py

**Purpose**: Configuration management system for the gait analysis pipeline.

**Key Features**:

- JSON-based configuration files
- Default parameter management
- Environment-specific configurations
- Validation and error handling
- Type-safe configuration access

**Usage**:

```python
from core.utils.config import ConfigManager

config = ConfigManager('configs/gait_analysis.json')
fps = config.get('fps', 30.0)
```

### utils/logging_config.py

**Purpose**: Centralized logging configuration for the entire system.

**Key Features**:

- Structured logging with timestamps
- Multiple output handlers (console, file)
- Configurable log levels
- Performance monitoring
- Error tracking and reporting

**Usage**:

```python
from core.utils.logging_config import setup_logging

logger = setup_logging('gait_analysis')
logger.info("Processing started")
```

## System Requirements

Core modules require:

- MediaPipe >= 0.10.0 (for MediaPipe pose estimation)
- PyTorch >= 1.9.0 (for future model support, optional)
- TensorFlow >= 2.8.0
- NumPy, SciPy, Pandas
- OpenCV >= 4.5.0
- Matplotlib, Seaborn (for visualization)

## Error Handling

Core modules implement robust error handling:

- Graceful degradation for missing pose data
- Validation of input parameters
- Comprehensive error messages
- Fallback mechanisms for failed operations
- Logging of errors for debugging

## Performance Considerations

- **Memory Management**: Efficient data structures and garbage collection
- **GPU Utilization**: Automatic GPU detection and utilization
- **Batch Processing**: Support for processing multiple videos
- **Caching**: Intelligent caching of processed data
- **Parallel Processing**: Multi-threading where appropriate

## Related Documentation

For more information about the project and its evolution:

- **Project Changelog**: [docs/README_Changelog.md](../docs/README_Changelog.md) - Complete project history and changes
- **Installation Guide**: [docs/README_Installation.md](../docs/README_Installation.md) - Comprehensive installation instructions
- **TCN System Documentation**: [docs/README_TCN_Gait_Analysis.md](../docs/README_TCN_Gait_Analysis.md) - Technical system documentation
- **Real-time Visualization**: [docs/README_RealTime_Visualization.md](../docs/README_RealTime_Visualization.md) - Real-time visualization guide

## Contributing

To extend the core modules:

1. **Follow the existing patterns** for new pose processors
2. **Implement abstract base classes** for consistency
3. **Add comprehensive tests** for new functionality
4. **Update documentation** for any new features
5. **Maintain backward compatibility** where possible

## License

This project is part of the Gait Analysis System. See the main LICENSE file for details.
