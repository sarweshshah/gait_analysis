# Temporal Convolutional Network (TCN) for Markerless Gait Analysis

This document provides comprehensive technical documentation for the Temporal Convolutional Network (TCN) based gait analysis system, including architecture details, usage examples, and implementation guidelines.

## Overview

The TCN-based gait analysis system provides a complete pipeline for markerless gait analysis using computer vision and deep learning techniques. The system combines pose estimation with temporal sequence modeling to analyze gait patterns and detect gait events.

### Key Features

- **Unified Pose Estimation**: Support for multiple pose estimation backends with easy model switching
- **Advanced Data Preprocessing**: Gap-filling, filtering, and normalization
- **TCN Architecture**: Temporal sequence modeling for gait analysis
- **Cross-validation Training**: Robust evaluation pipeline
- **Comprehensive Evaluation**: Gait-specific metrics and visualization
- **Real-time Processing**: Support for real-time pose estimation and visualization

## System Architecture

### Core Components

The system consists of several key components:

1. **Pose Estimation Layer**

   - **MediaPipe Integration (`mediapipe_integration.py`)**: Fast, real-time pose estimation
   - Both models output BODY_25 compatible format
   - Automatic model selection and switching capabilities

2. **Data Preprocessing Layer**

   - **GaitDataPreprocessor (`gait_data_preprocessing.py`)**: Advanced data preprocessing
   - Gap-filling using cubic spline interpolation
   - Butterworth low-pass filtering (6 Hz cutoff)
   - Keypoint normalization and standardization

3. **Model Layer**

   - **TCNGaitModel (`tcn_gait_model.py`)**: Temporal Convolutional Network
   - Dilated causal convolutions for temporal modeling
   - Residual connections and batch normalization
   - Support for both phase and event detection

4. **Training Layer**

   - **GaitTrainer (`gait_training.py`)**: Training and evaluation pipeline
   - Stratified k-fold cross-validation
   - Early stopping and learning rate scheduling
   - Comprehensive evaluation metrics

5. **Management Layer**
   - **PoseProcessorManager (`pose_processor_manager.py`)**: Unified pose processor manager
   - **UnifiedPoseProcessor**: Single interface for all pose models
   - **PoseProcessor**: Abstract base class for pose processors
   - Supports multiple pose estimation backends

### Data Flow

```
Video Input → Pose Estimation → Data Preprocessing → TCN Model → Gait Analysis Results
     ↓              ↓                    ↓              ↓              ↓
  Raw Video    Keypoints (BODY_25)   Normalized    Predictions    Events/Phases
                              ↓              ↓              ↓
                        Confidence      Features      Metrics
                        Filtering      Extraction    & Reports
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- MediaPipe >= 0.10.0
- TensorFlow >= 2.8.0
- NumPy, SciPy, Pandas
- OpenCV >= 4.5.0

### Quick Setup

```bash
# Install dependencies
pip3 install -r requirements.txt

# Test installation
python3 usecases/testing/test_system.py
```

## Usage Examples

### 1. Basic Video Processing

```bash
# Process video with MediaPipe (default)
python3 usecases/gait_analysis/main_gait_analysis.py \
    --videos data/video.mp4 \
    --output outputs/mediapipe/

# Process video with other models (when available)
python3 usecases/gait_analysis/main_gait_analysis.py \
    --videos data/video.mp4 \
    --pose-model other_model \
    --output outputs/other_model/

# Compare available models on the same video
python3 scripts/pose_model_comparison.py \
    --video gait_video.mp4 \
    --compare \
    --output outputs/test_results/
```

### 2. Training and Evaluation

```bash
# Train TCN model with cross-validation
python3 usecases/gait_analysis/main_gait_analysis.py \
    --videos data/training_videos/ \
    --labels data/labels.csv \
    --task-type phase_detection \
    --output outputs/gait_analysis/
```

### 3. Custom Configuration

Create a configuration file `config.json`:

```json
{
  "task_type": "phase_detection",
  "num_classes": 4,
  "num_filters": 128,
  "kernel_size": 5,
  "num_blocks": 6,
  "dropout_rate": 0.3,
  "learning_rate": 0.0005,
  "window_size": 45,
  "n_folds": 3,
  "epochs": 150
}
```

Use the configuration:

```bash
# Activate virtual environment first
source .venv/bin/activate

python3 usecases/gait_analysis/main_gait_analysis.py \
    --videos video.mp4 \
    --config config.json \
    --output outputs/gait_analysis/
```

### 4. Programmatic Usage

```python
from usecases.gait_analysis.main_gait_analysis import GaitAnalysisPipeline, create_default_config
from core.pose_processor_manager import UnifiedPoseProcessor

# Create configuration
config = create_default_config()
config['task_type'] = 'phase_detection'
config['num_classes'] = 4
config['pose_model'] = 'mediapipe'  # or other supported models

# Initialize pipeline
pipeline = GaitAnalysisPipeline(config)

# Run complete analysis
video_paths = ['video1.mp4', 'video2.mp4']
labels = [0, 1]  # 0: normal gait, 1: abnormal gait

results = pipeline.run_complete_pipeline(video_paths, labels)

# Access results
print(f"Mean Accuracy: {results['overall_metrics']['mean_accuracy']:.4f}")

# Direct pose processor usage
processor = UnifiedPoseProcessor(model_type='mediapipe')
success = processor.process_video('video.mp4')

# Switch to different model
processor.switch_model('other_model')
success = processor.process_video('video.mp4')
```

## Conventions

To keep the API clear and simple, the system uses enums for task types and dictionary keys:

- Task types: use `core.constants.TaskType` and `TaskType.DEFAULT`
- Event dictionaries: use constants such as `core.constants.EventType` and `EventType.HEEL_STRIKE`
- Sides and labels in results: use constants such as `core.constants.Side` and `Side.LEFT`
- Pose models: use string literals 'mediapipe' or other supported models for model selection

Please note that these changes are now reflected in the repository.
For more details, see the [constants file](core/utils/constants.py) in the repository.

## Technical Details

### Receptive Field Calculation

The TCN's receptive field determines how many past time steps it can consider:

```
Receptive Field = 1 + Σ(kernel_size - 1) × dilation_rate
```

For the default configuration:

- `kernel_size = 3`
- `num_blocks = 4`
- `dilation_rate = 2`

This gives a receptive field of 1 + (3-1) × 2 + (3-1) × 4 + (3-1) × 8 + (3-1) × 16 = 1 + 2 + 4 + 8 + 16 = 31 time steps

### Model Architecture

The TCN architecture consists of:

1. **Input Layer**: Normalized pose keypoints (25 keypoints × 3 coordinates = 75 features)
2. **TCN Blocks**: Multiple dilated causal convolution blocks
3. **Residual Connections**: Skip connections for better gradient flow
4. **Batch Normalization**: Normalization for stable training
5. **Dropout**: Regularization to prevent overfitting
6. **Output Layer**: Softmax classification for gait phases/events

### Data Preprocessing Pipeline

1. **Gap Filling**: Cubic spline interpolation for missing keypoints
2. **Filtering**: Butterworth low-pass filter (6 Hz cutoff)
3. **Normalization**: Z-score normalization per keypoint
4. **Window Creation**: Fixed-length windows with overlap
5. **Feature Extraction**: Joint angles, velocities, accelerations

### Training Strategy

1. **Cross-Validation**: Stratified k-fold cross-validation
2. **Early Stopping**: Stop training when validation loss plateaus
3. **Learning Rate Scheduling**: Reduce learning rate on plateau
4. **Data Augmentation**: Temporal jittering and noise injection
5. **Model Checkpointing**: Save best model from each fold

## Output Structure

### Training Results

```
outputs/gait_analysis/
├── cv_metrics.json              # Cross-validation metrics
├── fold_scores.json             # Per-fold performance
├── training_histories.json      # Training curves data
├── classification_report.txt    # Detailed classification report
├── confusion_matrix.png         # Confusion matrix visualization
├── training_curves.png          # Training curves plot
├── model_fold_1.h5              # Best model from fold 1
├── model_fold_2.h5              # Best model from fold 2
└── detailed_results.json        # Complete results summary
```

### Visualization Examples

The system automatically generates:

1. **Training Curves**: Loss and accuracy over epochs for each fold
2. **Confusion Matrix**: Classification performance visualization
3. **Metrics Summary**: Bar charts of overall performance metrics
4. **Fold Comparison**: Performance comparison across folds

## Troubleshooting

### Common Issues

#### 1. MediaPipe Not Found

```bash
# Check MediaPipe installation
python3 -c "import mediapipe; print(mediapipe.__version__)"

# Install MediaPipe if needed
pip3 install mediapipe>=0.10.0
```

#### 2. CUDA Out of Memory

```python
# Reduce batch size
config['batch_size'] = 16

# Reduce model complexity
config['num_filters'] = 32
config['num_blocks'] = 3
```

#### 3. Poor Pose Detection

```python
# Adjust confidence threshold
config['confidence_threshold'] = 0.2

# Improve video quality
# - Ensure good lighting
# - Use side-view camera angle
# - Maintain consistent background
```

#### 4. Overfitting

```python
# Increase regularization
config['dropout_rate'] = 0.4

# Reduce model complexity
config['num_filters'] = 32
config['num_blocks'] = 3

# Use more training data
# - Collect more video sequences
# - Use data augmentation
```

### Performance Optimization

1. **GPU Acceleration**: Ensure CUDA is properly configured
2. **Batch Processing**: Process multiple videos in parallel
3. **Memory Management**: Use appropriate batch sizes
4. **Data Preprocessing**: Cache preprocessed data for faster training

## Related Documentation

For more information about the project and its evolution:

- **Project Changelog**: [docs/README_Changelog.md](README_Changelog.md) - Complete project history and changes
- **Installation Guide**: [docs/README_Installation.md](README_Installation.md) - Comprehensive installation instructions
- **Core Modules**: [core/README_CoreModules.md](../core/README_CoreModules.md) - Core system modules documentation

## References

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
2. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines.
3. Perry, J., & Burnfield, J. M. (2010). Gait analysis: normal and pathological function.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
