# Gait Analysis on a webcam/video

Gait analysis is analysing an individual by their walking pattern and turns out to be a reliable indentification source.
As it turns out, it is as reliable and unique as one's fingerprint and retina scan.

## Original Basic Gait Analysis System

Following code explore the possibility to do the same by invovling following steps:

- Capturing image sequence
- Background modeling and Image subtraction
- Extracting binary silouette image
- Performing image correlation
- Applying discrete Fourier transformation
- Normalise results
- Perform primary component analysis
- Taking distance correction into account
- Indentification and verification

#### Pre-requisites:

- python-3.7
- mediapipe>=0.10.0
- opencv-contrib-python-3.4 (optional)
- imutils-0.5.1 (optional)

Pycharm IDE provides an easy interface to setup the environment for the same.
It automatically downloads the dependencies for the packages.

To check if you have successfully installed opencv, run the following command in the terminal:

```
>>> import mediapipe as mp
>>> print(mp.__version__)
```

If the results are printed out without any errors, congratulations !!!
You have installed MediaPipe successfully.

#### Step 1 | Capturing video frames:

The first step of the program captures image frames from available video file or webcam (by default) and presents it to the main thread for processing.
The image is turned to grayscale and fed for processing.

#### Performing background substraction (not required):

OpenCV provides various [background subtraction algorithms](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction) for video analysis.

The most common usage of these algorithms is to extract moving objects from a static background.
Hence, the `createBackgroundSubtractorMOG2()` method is used for performing the second step of gait analysis.
Here is the result:

![](https://github.com/sarweshshah/gait_analysis/blob/master/results/visualizations/background_subtraction.gif)

**NOTE:** The code performs the video processing and polling of video frames, both on the main thread. Since its an I/O bound process, the framerate of the output video becomes slow. (The CPU has to wait for the thread to get a new frame before it can apply background subtraction on it and vice-versa). Hence, we use threads.
[more...](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)

#### Step 2 | Pose Detection:

The [MediaPipe](https://mediapipe.dev/) library developed by Google provides real-time pose estimation capabilities with pre-trained models to detect the pose of any individual in a picture or video. These models accurately predict the joints inside the picture and even draw individual specific skeletons around each one of them.
This first layer of information can serve as data points for analysing a complete gait cycle of an individual.

**NOTE:** The background substraction step was removed later as the DNN code does not accept processed pictures as input. Hence that step is not necessary and can be ignored.

#### Step 3 | Trails of Recognised Joints:

If the posiiton of the joints of orthogonal view are traced on screen regularly at an interval of 1 sec, the corresponding snapshot of the movement can provide multiple information.
The distance between the adjoining points will give the instantenous speed of the joint (Time interval of 1 sec being constant).
The tangential angle gives the angle at the joint. This can give us useful info like the hip angle, calf angle, etc.
These 'keys' are similar to keys used in 3D motion capture animations to record position of individual joints which inturn becomes the basework of the entire animation.

The algorithm produced the following image for the side view of my walk:

![](https://github.com/sarweshshah/gait_analysis/blob/master/results/visualizations/pose_trail.gif)

---

## Advanced TCN-Based Gait Analysis System

This repository now includes a comprehensive **Temporal Convolutional Network (TCN)** system for markerless gait analysis with **unified pose estimation** supporting both MediaPipe and MeTRAbs.

**Key Features:**

- **Unified Pose Estimation**: Support for both MediaPipe (fast) and MeTRAbs (accurate) pose models
- **Model Switching**: Easy switching between pose estimation models
- **Advanced Data Preprocessing**: Gap-filling, filtering, and normalization
- **TCN Architecture**: Temporal sequence modeling for gait analysis
- **Cross-validation Training**: Robust evaluation pipeline
- **Comprehensive Evaluation**: Gait-specific metrics and visualization

**Project Structure:**

```
gait_analysis/
├── core/                                       # Core system modules
│   ├── utils/                                  # Utility modules
│   │   ├── constants.py                        # Core constants
│   │   ├── config.py                           # Configuration management
│   │   └── logging_config.py                   # Logging configuration
│   ├── pose_processor_manager.py               # Unified pose processor manager
│   ├── mediapipe_integration.py                # MediaPipe pose estimation
│   ├── metrabs_integration.py                  # MeTRAbs pose estimation
│   ├── gait_data_preprocessing.py              # Data preprocessing and feature extraction
│   ├── gait_training.py                        # Training and evaluation module
│   └── tcn_gait_model.py                       # Temporal Convolutional Network model
├── usecases/                                   # Use case implementations
│   ├── gait_analysis/                          # Main gait analysis use case
│   │   ├── features/                           # Feature-specific implementations
│   │   │   └── realtime_pose_visualization.py  # Real-time visualization
│   │   ├── utils.py                            # Utilities for quick analysis
│   │   └── main_gait_analysis.py               # Main pipeline orchestrator
│   └── testing/                                # Testing and validation
│       ├── test_pose_models.py                 # Pose model testing and comparison
│       └── test_system.py                      # System testing and validation
├── scripts/                                    # Utility scripts
│   ├── pose_model_comparison.py                # Pose model comparison tool
│   └── run_gait_analysis.py                    # Gait analysis runner
├── configs/                                    # Configuration files
│   ├── default.json                            # Default configuration
│   └── gait_analysis.json                      # Configuration for both models
├── docs/                                       # Documentation
│   ├── README_RealTime_Visualization.md        # Real-time visualization docs
│   ├── README_TCN_Gait_Analysis.md             # TCN system documentation
│   ├── README_Installation.md                  # Installation guide
│   ├── README_MeTRAbs_Integration.md           # MeTRAbs integration guide
│   └── README_Changelog.md                     # Project changelog and history
├── archive/                                    # Legacy scripts (see archive/README.md)
├── data/                                       # Input data directory
│   ├── models/                                 # Trained models
│   └── processed/                              # Processed data
├── videos/                                     # Video files directory
│   ├── raw/                                    # Raw video files
│   └── sneak/                                  # Sneak gait videos
└── results/                                    # Output results directory
    ├── gait_analysis/                          # Gait analysis results
    └── visualizations/                         # Generated visualizations
```

**Quick Start:**

```bash
# Setup environment
./setup_environment.sh  # macOS/Linux
# or
setup_environment.bat   # Windows

# Test pose models
source .venv/bin/activate
python3 usecases/testing/test_pose_models.py

# Run analysis with MediaPipe (default)
python3 usecases/gait_analysis/main_gait_analysis.py --help

# Compare pose models
python3 scripts/pose_model_comparison.py --help
```

## Pose Estimation Models

The system now supports two pose estimation models through a unified interface:

### MediaPipe (Default)
- **Speed**: Fast, real-time processing
- **Accuracy**: Good for most applications
- **Resource Usage**: Low, works on CPU
- **Best For**: Real-time applications, mobile/edge devices

### MeTRAbs (Optional)
- **Speed**: Slower, batch processing
- **Accuracy**: Higher precision
- **Resource Usage**: High, GPU recommended
- **Best For**: High-accuracy research, offline analysis

**Model Comparison:**
```bash
# Compare both models on the same video
python3 scripts/pose_model_comparison.py --video data/video.mp4 --compare

# Process with specific model
python3 usecases/gait_analysis/main_gait_analysis.py --videos video.mp4 --pose-model metrabs
```

## Real-time Pose Visualization

The system includes a real-time pose visualization tool that displays pose keypoints as colored dots with trail effects, similar to the original trail video approach but using the unified pose estimation system.

**Quick Demo:**

```bash
# Basic visualization with trail effect (MediaPipe)
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4

# Show confidence values
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --show-confidence

# Fast performance mode
python3 usecases/gait_analysis/features/realtime_pose_visualization.py videos/raw/sarwesh1.mp4 --model-complexity 0 --no-trail
```

**Interactive Controls:**

- **'q'**: Quit visualization
- **'t'**: Toggle trail effect
- **'c'**: Toggle connections
- **'r'**: Reset trail
- **SPACE**: Pause/resume
- **'1', '2', '3'**: Change model complexity

**For detailed documentation, see:** [docs/README_RealTime_Visualization.md](docs/README_RealTime_Visualization.md)

**For detailed TCN documentation, see:** [docs/README_TCN_Gait_Analysis.md](docs/README_TCN_Gait_Analysis.md)

**For installation guide, see:** [docs/README_Installation.md](docs/README_Installation.md)

**For MeTRAbs integration, see:** [docs/README_MeTRAbs_Integration.md](docs/README_MeTRAbs_Integration.md)

**For core modules documentation, see:** [core/README_CoreModules.md](core/README_CoreModules.md)

**For project changelog and history, see:** [docs/README_Changelog.md](docs/README_Changelog.md)

**Note:** Legacy scripts from the initial development phase have been moved to the `archive/` directory. See [archive/README.md](archive/README.md) for details about the archived files and migration notes. The basic gait analysis system documentation is now included above.
