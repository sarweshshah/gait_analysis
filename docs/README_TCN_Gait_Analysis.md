# Temporal Convolutional Network (TCN) for Markerless Gait Analysis

A comprehensive system for markerless gait analysis using MediaPipe-extracted 2D human pose data and Temporal Convolutional Networks (TCNs). This system is designed to detect gait events (heel strike, toe-off) and phases (stance, swing) from video sequences without requiring expensive motion capture equipment.

## Key Features

- **MediaPipe Pose Integration**: Uses MediaPipe pose estimation with foot keypoints for comprehensive gait analysis
- **Advanced Data Preprocessing**: Gap-filling, low-pass filtering, and coordinate normalization
- **TCN Architecture**: Temporal Convolutional Networks with dilated convolutions for long-range dependencies
- **Cross-Validation**: Robust evaluation with k-fold cross-validation
- **Comprehensive Metrics**: Gait-specific evaluation metrics including time deviation analysis
- **Complete Pipeline**: End-to-end solution from video processing to model evaluation

## Table of Contents

1. [Installation](#installation)
2. [System Architecture](#system-architecture)
3. [Data Preprocessing](#data-preprocessing)
4. [TCN Model](#tcn-model)
5. [Training and Evaluation](#training-and-evaluation)
6. [Usage Examples](#usage-examples)
7. [Technical Details](#technical-details)
8. [Results and Analysis](#results-and-analysis)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7+
- MediaPipe (for pose estimation)
- CUDA-compatible GPU (recommended for faster processing)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd gait_analysis
```

### 2. Quick Setup (Recommended)

**On macOS/Linux:**
```bash
./setup_environment.sh
```

**On Windows:**
```cmd
setup_environment.bat
```

### 3. Manual Setup (Alternative)

**Create Virtual Environment:**
```bash
python3 -m venv .venv

source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

**Install Dependencies:**
```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 3. Install MediaPipe

Install MediaPipe using pip:
```bash
pip3 install mediapipe>=0.10.0
```

### 4. MediaPipe Models (Auto-downloaded)

MediaPipe models are automatically downloaded on first use. No manual download required.

### 5. Activate Virtual Environment (if not already activated)

```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 6. Verify Installation

```bash
# Test the complete system
python3 usecases/testing/test_system.py

# Test pose models specifically
python3 usecases/testing/test_pose_models.py

# Compare available models
python3 scripts/pose_model_comparison.py --info
```

These will run comprehensive tests of all system components including the unified pose processor.

## System Architecture

The system consists of five main components:

### 1. Unified Pose Processor (`pose_processor_manager.py`)
- **UnifiedPoseProcessor**: Single interface for all pose estimation models
- **PoseProcessorManager**: Manager class for creating pose processors
- **PoseProcessor**: Abstract base class for pose processors
- Supports MediaPipe and MeTRAbs with easy model switching
- Consistent API regardless of underlying model

### 2. Pose Estimation Models
- **MediaPipe Integration (`mediapipe_integration.py`)**: Fast, real-time pose estimation
- **MeTRAbs Integration (`metrabs_integration.py`)**: High-accuracy pose estimation
- Both models output BODY_25 compatible format
- Automatic model selection and switching capabilities

### 3. Data Preprocessing (`gait_data_preprocessing.py`)
- Cleans and interpolates missing keypoints
- Applies low-pass filtering to reduce noise
- Normalizes coordinates for viewpoint invariance
- Extracts gait-specific features
- Creates fixed-length windows for TCN input

### 4. TCN Model (`tcn_gait_model.py`)
- Implements dilated causal convolutions
- Supports both phase detection and event detection
- Includes residual connections and batch normalization
- Configurable architecture for different tasks

### 5. Training and Evaluation (`gait_training.py`)
- Cross-validation training pipeline
- Gait-specific evaluation metrics
- Comprehensive result visualization
- Model performance analysis

## Data Preprocessing

### BODY_25 Keypoints

The system uses MediaPipe's pose estimation which includes 33 landmarks mapped to 25 keypoints:

```
0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip,
10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe,
20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel
```

### Gait-Relevant Features

The system focuses on 11 gait-relevant keypoints:
- **Lower Body**: MidHip, RHip, RKnee, RAnkle, RBigToe, RHeel, LHip, LKnee, LAnkle, LBigToe, LHeel

### Preprocessing Pipeline

1. **Keypoint Extraction**: Extract 2D coordinates and confidence scores
2. **Data Cleaning**: Remove low-confidence detections (< 0.3)
3. **Gap Filling**: Cubic spline interpolation for missing keypoints
4. **Low-Pass Filtering**: Butterworth filter (6 Hz cutoff) to reduce noise
5. **Normalization**: Scale by shoulder-hip distance and center around hip midpoint
6. **Feature Engineering**: Extract joint angles, relative positions, and velocities
7. **Window Creation**: Create fixed-length time windows (30 frames ≈ 1 second)

## TCN Model

### Architecture Overview

The TCN uses dilated causal convolutions to capture long-range temporal dependencies:

```
Input (30 frames × 50 features)
    ↓
Initial Conv1D (64 filters)
    ↓
Temporal Block 1 (dilation=1)
    ↓
Temporal Block 2 (dilation=2)
    ↓
Temporal Block 3 (dilation=4)
    ↓
Temporal Block 4 (dilation=8)
    ↓
Global Average Pooling
    ↓
Dense Layers (128 → 64 → num_classes)
    ↓
Output (class probabilities)
```

### Temporal Block Structure

Each temporal block consists of:
- Two dilated causal convolutions
- Batch normalization
- ReLU activation
- Dropout regularization
- Residual connection

### Model Variants

1. **Phase Detection Model**: Multi-class classification for gait phases
   - Classes: Stance, Swing, Double Support, Other
   - Output: Softmax probabilities

2. **Event Detection Model**: Binary classification for gait events
   - Events: Heel Strike, Toe-off
   - Output: Sigmoid probabilities per frame

## Training and Evaluation

### Cross-Validation

The system uses stratified k-fold cross-validation (k=5) to ensure robust evaluation:

```python
# Example training configuration
cv_results = trainer.train_with_cross_validation(
    features=features,
    labels=labels,
    n_folds=5,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    early_stopping_patience=15
)
```

### Evaluation Metrics

#### For Phase Detection:
- **Overall Accuracy**: Percentage of correctly classified frames
- **F1 Score**: Harmonic mean of precision and recall
- **Phase Transition Accuracy**: Accuracy at phase change points
- **Per-Class Metrics**: Precision, recall, F1 for each phase

#### For Event Detection:
- **Time Deviation MAE**: Mean absolute error in milliseconds
- **Event Detection Accuracy**: Percentage of correctly detected events
- **Precision/Recall**: For each event type

### Callbacks and Regularization

- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Adaptive learning rate scheduling
- **Model Checkpointing**: Saves best models per fold
- **Dropout**: Regularization during training

## Usage Examples

### 1. Complete Pipeline Execution

```bash
# Activate virtual environment first
source .venv/bin/activate

# Process multiple videos for phase detection
python3 usecases/gait_analysis/main_gait_analysis.py \
    --videos video1.mp4 video2.mp4 video3.mp4 \
    --labels 0 1 0 \
    --task phase_detection \
    --output results/

# Process single video for event detection
python3 usecases/gait_analysis/main_gait_analysis.py \
    --videos gait_video.mp4 \
    --task event_detection \
    --output event_results/
```

### 2. Pose Model Processing

```bash
# Activate virtual environment first
source .venv/bin/activate

# Process video with MediaPipe (default)
python3 scripts/pose_model_comparison.py \
    --video gait_video.mp4 \
    --model mediapipe \
    --output pose_output/

# Process video with MeTRAbs
python3 scripts/pose_model_comparison.py \
    --video gait_video.mp4 \
    --model metrabs \
    --output pose_output/

# Compare both models
python3 scripts/pose_model_comparison.py \
    --video gait_video.mp4 \
    --compare \
    --output comparison_results/
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
    --output custom_results/
```

### 4. Programmatic Usage

```python
from usecases.gait_analysis.main_gait_analysis import GaitAnalysisPipeline, create_default_config
from core.pose_processor_manager import UnifiedPoseProcessor

# Create configuration
config = create_default_config()
config['task_type'] = 'phase_detection'
config['num_classes'] = 4
config['pose_model'] = 'metrabs'  # or 'mediapipe'

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
processor.switch_model('metrabs')
success = processor.process_video('video.mp4')
```

## Conventions

To keep the API clear and simple, the system uses enums for task types and dictionary keys:

- Task types: use `core.constants.TaskType` and `TaskType.DEFAULT`
- Event dictionaries: use constants such as `core.constants.EventType` and `EventType.HEEL_STRIKE`
- Sides and labels in results: use constants such as `core.constants.Side` and `Side.LEFT`
- Pose models: use string literals 'mediapipe' or 'metrabs' for model selection

Please note that these changes are now reflected in the repository.
For more details, see the [constants file](core/utils/constants.py) in the repository.

## Technical Details

### Receptive Field Calculation

The TCN's receptive field determines how many past time steps it can consider:

```
Receptive Field = 1 + Σ(kernel_size - 1) × dilation_rate
```

For default settings (kernel_size=3, num_blocks=4):
- Block 1: dilation=1, RF = 1 + 2×1 = 3
- Block 2: dilation=2, RF = 3 + 2×2 = 7
- Block 3: dilation=4, RF = 7 + 2×4 = 15
- Block 4: dilation=8, RF = 15 + 2×8 = 31

Total receptive field: 31 time steps (≈1 second at 30fps)

### Feature Engineering

The system extracts multiple feature types:

1. **Joint Positions**: (x, y) coordinates of gait-relevant keypoints
2. **Joint Angles**: Hip-knee-ankle angles for both legs
3. **Relative Positions**: Ankle positions relative to hip midpoint
4. **Velocities**: Frame-to-frame keypoint velocities

### Data Augmentation

For improved generalization, consider:
- Temporal jittering (small time shifts)
- Coordinate noise injection
- Rotation augmentation (for side-view videos)

## Results and Analysis

### Expected Performance

Based on similar studies, the system typically achieves:

- **Phase Detection**: 85-95% accuracy
- **Event Detection**: 80-90% accuracy
- **Time Deviation**: <50ms MAE for event timing

### Output Files

The system generates comprehensive results:

```
gait_analysis_results/
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
- **MeTRAbs Integration**: [docs/README_MeTRAbs_Integration.md](README_MeTRAbs_Integration.md) - Detailed MeTRAbs guide
- **Core Modules**: [core/README_CoreModules.md](../core/README_CoreModules.md) - Core system modules documentation

## References

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.
2. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines.
3. Perry, J., & Burnfield, J. M. (2010). Gait analysis: normal and pathological function.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for the pose estimation framework
- CMU Perceptual Computing Lab for the BODY_25 model
- TensorFlow/Keras community for the deep learning framework

---

For questions and support, please open an issue on the repository or contact the development team.
