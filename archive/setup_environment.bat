@echo off
REM Setup script for Legacy OpenPose-based Gait Analysis Scripts
REM ===========================================================

echo Setting up environment for legacy gait analysis scripts...

REM Create necessary directories
echo Creating directories...
if not exist "dnn_models\pose\coco" mkdir "dnn_models\pose\coco"
if not exist "dnn_models\object_detection" mkdir "dnn_models\object_detection"
if not exist "data" mkdir "data"
if not exist "results" mkdir "results"

REM Download OpenPose COCO model files
echo Downloading OpenPose COCO model files...
cd dnn_models\pose\coco

REM Download prototxt file
if not exist "pose_deploy_linevec.prototxt" (
    echo Downloading pose_deploy_linevec.prototxt...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt' -OutFile 'pose_deploy_linevec.prototxt'"
) else (
    echo pose_deploy_linevec.prototxt already exists
)

REM Download caffemodel file
if not exist "pose_iter_440000.caffemodel" (
    echo Downloading pose_iter_440000.caffemodel...
    echo Note: This file is large (~200MB). Downloading from CMU repository...
    powershell -Command "Invoke-WebRequest -Uri 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel' -OutFile 'pose_iter_440000.caffemodel'"
) else (
    echo pose_iter_440000.caffemodel already exists
)

cd ..\..\object_detection

REM Download MobileNet SSD model files
echo Downloading MobileNet SSD model files...

REM Download prototxt file
if not exist "MobileNetSSD_deploy.prototxt" (
    echo Downloading MobileNetSSD_deploy.prototxt...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt' -OutFile 'MobileNetSSD_deploy.prototxt'"
) else (
    echo MobileNetSSD_deploy.prototxt already exists
)

REM Download caffemodel file
if not exist "MobileNetSSD_deploy.caffemodel" (
    echo Downloading MobileNetSSD_deploy.caffemodel...
    echo Note: This file is large (~22MB). Downloading from Chuanqi305 repository...
    powershell -Command "Invoke-WebRequest -Uri 'https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc' -OutFile 'MobileNetSSD_deploy.caffemodel'"
) else (
    echo MobileNetSSD_deploy.caffemodel already exists
)

cd ..\..

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo Setup completed!
echo.
echo Next steps:
echo 1. Place your video files in the 'data' directory
echo 2. Update the video file paths in the scripts if needed
echo 3. Run the scripts from the archive directory
echo.
echo Available scripts:
echo - poseTrailVideo.py: Shows pose keypoints with trail effect
echo - poseDetectVideo.py: Shows pose detection with skeleton overlay
echo - hipsTrailVideo.py: Shows hip keypoints with trail effect
echo - poseAndObjectDetectVideo.py: Shows both pose and object detection

pause
