# Core System Modules

This directory contains the core modules that power the gait analysis system. These modules provide the fundamental functionality for pose estimation, data preprocessing, model training, and evaluation.

## Module Overview

### Data Processing
- **`gait_data_preprocessing.py`** - Advanced data preprocessing with gap-filling, filtering, and normalization
- **`mediapipe_integration.py`** - MediaPipe pose estimation integration with BODY_25 keypoint mapping

### Machine Learning
- **`tcn_gait_model.py`** - Temporal Convolutional Network architecture for gait analysis
- **`gait_training.py`** - Training pipeline with cross-validation and comprehensive evaluation

### Utilities
- **`utils/config.py`** - Configuration management system
- **`utils/logging_config.py`** - Centralized logging configuration

## Module Details

### gait_data_preprocessing.py

**Purpose**: Transforms raw MediaPipe pose data into clean, normalized features suitable for machine learning.

**Key Features**:
- Gap-filling using cubic spline interpolation
- Butterworth low-pass filtering (6 Hz cutoff)
- Coordinate normalization and scaling
- Feature engineering (joint angles, velocities)
- Fixed-length window creation for TCN input

**Usage**:
```python
from core.gait_data_preprocessing import GaitDataPreprocessor

preprocessor = GaitDataPreprocessor()
features, labels = preprocessor.process_pose_data(pose_data)
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

**Purpose**: Provides comprehensive training and evaluation pipeline with cross-validation.

**Key Features**:
- Stratified k-fold cross-validation
- Early stopping and learning rate scheduling
- Gait-specific evaluation metrics
- Model checkpointing and result visualization
- Comprehensive performance analysis

**Usage**:
```python
from core.gait_training import GaitTrainer

trainer = GaitTrainer(config)
results = trainer.train_with_cross_validation(features, labels)
```

### utils/config.py

**Purpose**: Centralized configuration management for all system components.

**Key Features**:
- JSON-based configuration files
- Default parameter management
- Environment-specific overrides
- Validation and type checking

**Usage**:
```python
from core.utils.config import ConfigManager

config = ConfigManager.load_config("configs/gait_analysis.json")
```

### utils/logging_config.py

**Purpose**: Standardized logging configuration across all modules.

**Key Features**:
- Structured logging with timestamps
- Multiple output handlers (console, file)
- Configurable log levels
- Module-specific loggers

**Usage**:
```python
from core.utils.logging_config import setup_logging

logger = setup_logging(__name__)
logger.info("Processing started")
```

## Integration with Use Cases

The core modules are designed to be imported and used by specific use case implementations in the `usecases/` directory:

```python
# Example usage in usecases/gait_analysis/main_gait_analysis.py
from core.mediapipe_integration import MediaPipeProcessor
from core.gait_data_preprocessing import GaitDataPreprocessor
from core.tcn_gait_model import TCNGaitModel
from core.gait_training import GaitTrainer
```

## Conventions

To keep the core APIs simple and readable, the system uses plain string literals for simple identifiers:

- Task types: 'phase_detection', 'event_detection'
- Event dictionary keys: 'heel_strikes', 'toe_offs', 'flat_foots', 'heel_offs'
- Sides and labels: 'left', 'right', etc.

Only core domain names like event names and sides are kept as centralized constants in `core/utils/constants.py` to avoid typos. Do not introduce constants for simple dictionary keys or task type strings.

## Core Constants

Shared domain names and labels are defined in `core/utils/constants.py`:

```python
from core.utils.constants import (
    SIDES, LEFT, RIGHT,
    EVENT_TYPES, HEEL_STRIKE, TOE_OFF,
    PHASE_LABELS_4, get_phase_labels, TASK_TYPES
)

print(SIDES)               # ['left', 'right']
print(EVENT_TYPES)         # ['heel_strike', 'toe_off'] (default basic set)
print(get_phase_labels(4)) # ['stance_left', 'swing_left', 'stance_right', 'swing_right']
print(TASK_TYPES)          # ['phase_detection', 'event_detection']
```

Note: Dictionary keys and task types should be used as string literals in code. Constants here are for event names, sides, and label sets.

## Configuration

Core modules use configuration files from the `configs/` directory:

- **`configs/default.json`** - Default system parameters
- **`configs/gait_analysis.json`** - Configuration for both models

## Dependencies

Core modules require:
- MediaPipe >= 0.10.0
- TensorFlow >= 2.8.0
- NumPy, SciPy, Pandas
- OpenCV for video processing
- Scikit-learn for evaluation metrics

## Development Guidelines

When extending core modules:

1. **Maintain backward compatibility** - Existing use cases should continue to work
2. **Follow configuration patterns** - Use ConfigManager for all parameters
3. **Add comprehensive logging** - Use the centralized logging system
4. **Include unit tests** - Test core functionality thoroughly
5. **Document public APIs** - Maintain clear docstrings and type hints

## Performance Considerations

- **GPU Acceleration**: Core modules support CUDA when available
- **Memory Management**: Efficient handling of large video datasets
- **Batch Processing**: Optimized for processing multiple videos
- **Caching**: Preprocessed data caching to avoid recomputation

## Error Handling

Core modules implement robust error handling:
- Graceful degradation for missing pose data
- Validation of input parameters
- Clear error messages with suggested solutions
- Automatic fallback to CPU processing if GPU fails

For detailed usage examples and advanced configuration, see the documentation in the `docs/` directory and the use case implementations in `usecases/`.
