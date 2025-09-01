"""
Configuration file for Legacy OpenPose-based Gait Analysis Scripts
================================================================

This file contains configuration settings for all legacy scripts.
Modify these settings to customize the behavior of the scripts.

Author: Gait Analysis System
"""

import os

# =============================================================================
# Model Configuration
# =============================================================================

# OpenPose COCO model paths
OPENPOSE_PROTO_FILE = "dnn_models/pose/coco/pose_deploy_linevec.prototxt"
OPENPOSE_WEIGHTS_FILE = "dnn_models/pose/coco/pose_iter_440000.caffemodel"

# MobileNet SSD model paths (for object detection)
OBJECT_DETECTION_PROTO_FILE = "dnn_models/object_detection/MobileNetSSD_deploy.prototxt"
OBJECT_DETECTION_WEIGHTS_FILE = "dnn_models/object_detection/MobileNetSSD_deploy.caffemodel"

# =============================================================================
# Video Processing Configuration
# =============================================================================

# Default video files for each script
DEFAULT_VIDEOS = {
    "poseTrailVideo": "videos/raw/sarwesh.mp4",
    "poseDetectVideo": "videos/raw/sarwesh.mp4", 
    "hipsTrailVideo": "videos/raw/hydrocephalus.mp4",
    "poseAndObjectDetectVideo": "videos/raw/sarwesh.mp4"
}

# Video processing settings
VIDEO_SETTINGS = {
    "queue_size": 1024,
    "resize_width": 1080,  # For poseTrailVideo and hipsTrailVideo
    "resize_width_detection": 960,  # For poseDetectVideo and poseAndObjectDetectVideo
    "frame_delay": 50,  # milliseconds between frames
    "startup_delay": 1.0  # seconds to wait before starting
}

# =============================================================================
# OpenPose Configuration
# =============================================================================

# COCO keypoint mapping (18 points)
KEYPOINT_MAPPING = [
    'Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr',
    'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank',
    'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear'
]

# Pose pairs for skeleton drawing
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 17], [5, 16]
]

# PAF (Part Affinity Fields) indices
MAP_IDX = [
    [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
    [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
    [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
    [37, 38], [45, 46]
]

# Colors for different keypoints
KEYPOINT_COLORS = [
    [0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
    [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
    [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]
]

# =============================================================================
# Detection Parameters
# =============================================================================

# Keypoint detection threshold
KEYPOINT_THRESHOLD = 0.1  # Default threshold for keypoint detection
KEYPOINT_THRESHOLD_TRAIL = 0.3  # Higher threshold for trail visualization

# PAF (Part Affinity Fields) parameters
PAF_SETTINGS = {
    "n_interp_samples": 10,  # Number of interpolation samples
    "paf_score_th": 0.1,     # PAF score threshold
    "conf_th": 0.7           # Confidence threshold
}

# =============================================================================
# Object Detection Configuration
# =============================================================================

# MobileNet SSD classes
OBJECT_DETECTION_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Object detection confidence threshold
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.6

# =============================================================================
# Background Subtraction Configuration
# =============================================================================

# Background subtraction settings (for poseDetectVideo.py)
BACKGROUND_SUBTRACTION = {
    "history": 15,
    "detect_shadows": True,
    "kernel_size": 7
}

# =============================================================================
# Output Configuration
# =============================================================================

# Output directories
OUTPUT_DIRS = {
    "results": "results",
    "videos": "outputs/videos",
    "images": "outputs/images",
    "logs": "outputs/logs"
}

# Create output directories
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# Utility Functions
# =============================================================================

def get_video_path(script_name):
    """Get the default video path for a script."""
    return DEFAULT_VIDEOS.get(script_name, "videos/raw/sample.mp4")

def get_resize_width(script_name):
    """Get the resize width for a script."""
    if script_name in ["poseTrailVideo", "hipsTrailVideo"]:
        return VIDEO_SETTINGS["resize_width"]
    else:
        return VIDEO_SETTINGS["resize_width_detection"]

def get_keypoint_threshold(script_name):
    """Get the keypoint threshold for a script."""
    if script_name in ["poseTrailVideo", "hipsTrailVideo"]:
        return KEYPOINT_THRESHOLD_TRAIL
    else:
        return KEYPOINT_THRESHOLD
