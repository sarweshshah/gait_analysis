# Archived Files

This directory contains legacy Python scripts that were part of the initial development phase of the gait analysis system. These files have been superseded by the more comprehensive and modular approach implemented in the main project.

## Archived Files

### 1. poseTrailVideo.py
- **Purpose**: Basic pose detection with trail visualization
- **Features**: 
  - OpenPose integration using COCO model
  - Keypoint detection and visualization
  - Trail effect for keypoint tracking
- **Status**: Replaced by `openpose_integration.py` and `main_gait_analysis.py`

### 2. poseDetectVideo.py
- **Purpose**: Standalone pose detection script
- **Features**:
  - OpenPose pose estimation
  - Keypoint detection and validation
  - Pose visualization
- **Status**: Functionality integrated into `openpose_integration.py`

### 3. hipsTrailVideo.py
- **Purpose**: Specialized hips tracking with trail visualization
- **Features**:
  - Focus on hip keypoint tracking
  - Trail visualization for hip movement
- **Status**: Specialized functionality can be implemented using the main pipeline

### 4. poseAndObjectDetectVideo.py
- **Purpose**: Combined pose detection and object detection
- **Features**:
  - OpenPose for pose estimation
  - MobileNet SSD for object detection
  - Combined visualization
- **Status**: Object detection functionality can be added to the main pipeline if needed

## Migration Notes

These files were archived because:
1. The main project now uses a more modular architecture
2. Better error handling and logging in the new implementation
3. More comprehensive configuration management
4. Improved data preprocessing and feature extraction
5. Better integration with the TCN model for gait analysis

## Current Architecture

The current system is organized as follows:
- `main_gait_analysis.py` - Main pipeline orchestrator
- `openpose_integration.py` - OpenPose processing module
- `gait_data_preprocessing.py` - Data preprocessing and feature extraction
- `tcn_gait_model.py` - Temporal Convolutional Network model
- `gait_training.py` - Training and evaluation module
- `test_system.py` - System testing and validation

## Usage

If you need functionality from these archived files, consider:
1. Using the main pipeline with appropriate configuration
2. Extending the current modules to add specific features
3. Creating new modules that integrate with the existing architecture

The archived files can serve as reference for specific implementation details if needed.
