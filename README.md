# Gait Analysis using opencv-python 3.4 on a webcam/video

Gait analysis is analysing an individual by their walking pattern and turns out to be a reliable indentification source.
As it turns out, it is as reliable and unique as one's fingerprint and retina scan.

## 🆕 Advanced TCN-Based Gait Analysis System

This repository now includes a comprehensive **Temporal Convolutional Network (TCN)** system for markerless gait analysis using OpenPose BODY_25 model. 

**Key Features:**
- OpenPose BODY_25 integration with foot keypoints
- Advanced data preprocessing with gap-filling and filtering
- TCN architecture for temporal sequence modeling
- Cross-validation training pipeline
- Comprehensive evaluation metrics

**Project Structure:**
```
gait_analysis/
├── main_gait_analysis.py          # Main pipeline orchestrator
├── openpose_integration.py        # OpenPose processing module
├── gait_data_preprocessing.py     # Data preprocessing and feature extraction
├── tcn_gait_model.py             # Temporal Convolutional Network model
├── gait_training.py              # Training and evaluation module
├── test_system.py                # System testing and validation
├── archive/                      # Legacy scripts (see archive/README.md)
├── data/                         # Input data directory
├── results/                      # Output results directory
└── dnn_models/                   # Pre-trained models
```

**Quick Start:**
```bash
# Setup environment
./setup_environment.sh  # macOS/Linux
# or
setup_environment.bat   # Windows

# Run analysis
source .venv/bin/activate
python main_gait_analysis.py --help
```

**For detailed documentation, see:** [README_TCN_Gait_Analysis.md](README_TCN_Gait_Analysis.md)

**Note:** Legacy scripts from the initial development phase have been moved to the `archive/` directory. See [archive/README.md](archive/README.md) for details about the archived files and migration notes.

---

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
- opencv-python-3.4
- opencv-contrib-python-3.4 (optional)
- imutils-0.5.1 (optional)

Pycharm IDE provides an easy interface to setup the environment for the same.
It automatically downloads the dependencies for the packages.

To check if you have successfully installed opencv, run the following command in the terminal:
```
>>> import cv2
>>> print cv2.__version__
```
If the results are printed out without any errors, congratulations !!! You have installed OpenCV-Python successfully.

<br>

#### Step 1 | Capturing video frames:
The first step of the program captures image frames from available video file or webcam (by default) and presents it to the main thread for processing.
The image is turned to grayscale and fed for processing.

#### Performing background substraction (not required):
OpenCV provides various [background subtraction algorithms](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html#background-subtraction) for video analysis. 

The most common usage of these algorithms is to extract moving objects from a static background.
Hence, the `createBackgroundSubtractorMOG2()` method is used for performing the second step of gait analysis.
Here is the result:

![](https://github.com/sarweshshah/gait_analysis/blob/master/results/background%20subtraction.gif)

**NOTE:** The code performs the video processing and polling of video frames, both on the main thread. Since its an I/O bound process, the framerate of the output video becomes slow. (The CPU has to wait for the thread to get a new frame before it can apply background subtraction on it and vice-versa). Hence, we use threads.
[more...](https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)

#### Step 2 | Pose Detection:
The [OpenPose deep learning library](https://github.com/CMU-Perceptual-Computing-Lab/openpose) developed by Perceptual Computing Lab at Carnegie Mellon University provides with specific weight files and model to detect the pose of any individual in a picture. These models accurately predict the joints inside the picture and even draw individual specific skeletons around each one of them.
This first layer of information can serve as data points for analysing a complete gait cycle of an individual.

**NOTE:** The background substraction step was removed later as the DNN code does not accept processed pictures as input. Hence that step is not necessary and can be ignored.

#### Step 3 | Trails of Recognised Joints:
If the posiiton of the joints of orthogonal view are traced on screen regularly at an interval of 1 sec, the corresponding snapshot of the movement can provide multiple information.
The distance between the adjoining points will give the instantenous speed of the joint (Time interval of 1 sec being constant).
The tangential angle gives the angle at the joint. This can give us useful info like the hip angle, calf angle, etc.
These 'keys' are similar to keys used in 3D motion capture animations to record position of individual joints which inturn becomes the basework of the entire animation.

The algorithm produced the following image for the side view of my walk:

![](https://github.com/sarweshshah/gait_analysis/blob/master/results/pose%20trail.gif)
