#!/bin/bash

# Setup script for Legacy OpenPose-based Gait Analysis Scripts
# ===========================================================

echo "Setting up environment for legacy gait analysis scripts..."

# Create necessary directories
echo "Creating directories..."
mkdir -p dnn_models/pose/coco
mkdir -p dnn_models/object_detection
mkdir -p data
mkdir -p results

# Download OpenPose COCO model files
echo "Downloading OpenPose COCO model files..."
cd dnn_models/pose/coco

# Download prototxt file
if [ ! -f "pose_deploy_linevec.prototxt" ]; then
    echo "Downloading pose_deploy_linevec.prototxt..."
    wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
else
    echo "pose_deploy_linevec.prototxt already exists"
fi

# Download caffemodel file
if [ ! -f "pose_iter_440000.caffemodel" ]; then
    echo "Downloading pose_iter_440000.caffemodel..."
    echo "Note: This file is large (~200MB). Downloading from CMU repository..."
    wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
else
    echo "pose_iter_440000.caffemodel already exists"
fi

cd ../../object_detection

# Download MobileNet SSD model files
echo "Downloading MobileNet SSD model files..."

# Download prototxt file
if [ ! -f "MobileNetSSD_deploy.prototxt" ]; then
    echo "Downloading MobileNetSSD_deploy.prototxt..."
    wget https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt -O MobileNetSSD_deploy.prototxt
else
    echo "MobileNetSSD_deploy.prototxt already exists"
fi

# Download caffemodel file
if [ ! -f "MobileNetSSD_deploy.caffemodel" ]; then
    echo "Downloading MobileNetSSD_deploy.caffemodel..."
    echo "Note: This file is large (~22MB). Downloading from Chuanqi305 repository..."
    wget https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc -O MobileNetSSD_deploy.caffemodel
else
    echo "MobileNetSSD_deploy.caffemodel already exists"
fi

cd ../..

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Place your video files in the 'data' directory"
echo "2. Update the video file paths in the scripts if needed"
echo "3. Run the scripts from the archive directory"
echo ""
echo "Available scripts:"
echo "- poseTrailVideo.py: Shows pose keypoints with trail effect"
echo "- poseDetectVideo.py: Shows pose detection with skeleton overlay"
echo "- hipsTrailVideo.py: Shows hip keypoints with trail effect"
echo "- poseAndObjectDetectVideo.py: Shows both pose and object detection"
