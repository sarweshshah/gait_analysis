# Archived Files - Legacy OpenPose-based Gait Analysis
This directory contains legacy scripts from the initial development phase of the gait analysis system. 
These scripts use OpenPose for pose estimation and provide basic visualization and detection capabilities.

## Archived Files

### 1. poseTrailVideo.py
- **Purpose**: Shows pose keypoints with trail effect over time
- **Features**: 
  - Real-time pose keypoint detection
  - Trail visualization showing keypoint movement
  - COCO format (18 keypoints)
- **Input**: Video file (default: `data/sarwesh.mp4`)
- **Output**: Real-time visualization window

### 2. poseDetectVideo.py
- **Purpose**: Shows pose detection with skeleton overlay
- **Features**:
  - Full skeleton visualization
  - Part Affinity Fields (PAF) for limb connections
  - Background subtraction capability
  - Multi-person detection
- **Input**: Video file (default: `data/sarwesh.mp4`)
- **Output**: Real-time skeleton overlay

### 3. hipsTrailVideo.py
- **Purpose**: Shows hip keypoints with trail effect
- **Features**:
  - Focused on hip keypoint tracking
  - Trail visualization for hip movement
  - Specialized for gait analysis
- **Input**: Video file (default: `data/hydrocephalus.mp4`)
- **Output**: Real-time hip trail visualization

### 4. poseAndObjectDetectVideo.py
- **Purpose**: Shows both pose and object detection
- **Features**:
  - Combined pose estimation and object detection
  - MobileNet SSD for object detection
  - Skeleton overlay with bounding boxes
- **Input**: Video file (default: `data/sarwesh.mp4`)
- **Output**: Real-time combined visualization

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- OpenCV with DNN support
- Internet connection for downloading model files

### Quick Setup

#### Option 1: Automated Setup (Recommended)

**For Linux/Mac:**
```bash
cd archive
chmod +x setup_environment.sh
./setup_environment.sh
```

**For Windows:**
```cmd
cd archive
setup_environment.bat
```

#### Option 2: Manual Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download model files:**
   - OpenPose COCO model (~200MB):
     - `pose_deploy_linevec.prototxt`
     - `pose_iter_440000.caffemodel`
   - MobileNet SSD model (~22MB):
     - `MobileNetSSD_deploy.prototxt`
     - `MobileNetSSD_deploy.caffemodel`

3. **Create directory structure:**
   ```
   archive/
   ├── dnn_models/
   │   ├── pose/coco/
   │   └── object_detection/
   ├── data/
   └── results/
   ```

## Usage

### Using the Script Runner (Recommended)

```bash
# List available scripts
python run_scripts.py --list

# Check dependencies and model files
python run_scripts.py --check

# Run a specific script
python run_scripts.py poseTrailVideo

# Run with custom video file
python run_scripts.py poseDetectVideo --video path/to/your/video.mp4

# Run with custom output directory
python run_scripts.py hipsTrailVideo --output custom_results
```

### Direct Script Execution

```bash
# Run individual scripts
python poseTrailVideo.py
python poseDetectVideo.py
python hipsTrailVideo.py
python poseAndObjectDetectVideo.py
```

### Configuration

Edit `config.py` to customize:
- Video file paths
- Detection thresholds
- Output settings
- Model parameters

## File Structure

```
archive/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── run_scripts.py              # Script runner utility
├── setup_environment.sh        # Linux/Mac setup script
├── setup_environment.bat       # Windows setup script
├── poseTrailVideo.py           # Pose trail visualization
├── poseDetectVideo.py          # Pose detection with skeleton
├── hipsTrailVideo.py           # Hip trail visualization
├── poseAndObjectDetectVideo.py # Combined pose and object detection
├── dnn_models/                 # Neural network models
│   ├── pose/coco/
│   │   ├── pose_deploy_linevec.prototxt
│   │   └── pose_iter_440000.caffemodel
│   └── object_detection/
│       ├── MobileNetSSD_deploy.prototxt
│       └── MobileNetSSD_deploy.caffemodel
├── data/                       # Input video files
└── results/                    # Output files
```

## Troubleshooting

### Common Issues

1. **"Model files not found"**
   - Run the setup script to download model files
   - Check that model files are in the correct directories

2. **"Video file not found"**
   - Place your video files in the `data/` directory
   - Update the video path in the script or use `--video` parameter

3. **"OpenCV DNN module not available"**
   - Install `opencv-contrib-python` instead of `opencv-python`
   - Ensure you have the full OpenCV installation

4. **"CUDA/GPU errors"**
   - These scripts work on CPU by default
   - For GPU support, install `opencv-python-gpu`

### Performance Tips

- **CPU Usage**: These scripts are CPU-intensive. Close other applications for better performance.
- **Video Resolution**: Lower resolution videos process faster.
- **Frame Rate**: Higher frame rates require more processing power.

## Migration Notes

These files were archived because:

1. **OpenPose Dependency**: Required heavy OpenPose installation
2. **Performance**: CPU-intensive processing
3. **Integration**: Difficult to integrate with modern pipeline
4. **Maintenance**: OpenPose models require regular updates

### If you need functionality from these archived files, consider:

1. **MediaPipe Integration**: Use the current system's MediaPipe-based pose estimation
2. **Modern Alternatives**: Consider MediaPipe, BlazePose, or other lightweight solutions
3. **Custom Implementation**: Adapt the visualization logic to work with current pipeline

The archived files can serve as reference for specific implementation details if needed.

## License

These scripts are part of the Gait Analysis System and follow the same license as the main project.
