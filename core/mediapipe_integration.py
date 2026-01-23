"""
MediaPipe Integration Module for Gait Analysis
=============================================

This module provides MediaPipe-based pose estimation for gait analysis,
using the new MediaPipe Tasks API (0.10.x+).

Author: Gait Analysis System
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model URLs for automatic download
POSE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
POSE_LANDMARKER_MODEL_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_LANDMARKER_MODEL_FULL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


def get_model_path(model_complexity: int = 1) -> str:
    """
    Get the path to the pose landmarker model, downloading if necessary.
    
    Args:
        model_complexity: 0=lite, 1=full, 2=heavy
        
    Returns:
        Path to the model file
    """
    # Determine model URL and filename based on complexity
    if model_complexity == 0:
        model_url = POSE_LANDMARKER_MODEL_LITE_URL
        model_name = "pose_landmarker_lite.task"
    elif model_complexity == 2:
        model_url = POSE_LANDMARKER_MODEL_URL  # heavy
        model_name = "pose_landmarker_heavy.task"
    else:
        model_url = POSE_LANDMARKER_MODEL_FULL_URL
        model_name = "pose_landmarker_full.task"
    
    # Create models directory in the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, "models", "mediapipe")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_name)
    
    # Download if not exists
    if not os.path.exists(model_path):
        logger.info(f"Downloading MediaPipe Pose Landmarker model ({model_name})...")
        logger.info("This is a one-time download. Please wait...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            logger.info(f"Model downloaded successfully to: {model_path}")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            logger.error(f"Please download manually from: {model_url}")
            logger.error(f"And place it at: {model_path}")
            raise
    
    return model_path


class MediaPipeProcessor:
    """
    MediaPipe-based pose estimation processor for gait analysis.

    This class handles:
    - MediaPipe Pose Landmarker initialization and configuration (Tasks API)
    - Real-time pose estimation from video
    - Keypoint extraction and formatting
    - JSON output generation compatible with existing pipeline
    """

    def __init__(
        self,
        output_dir: str = "outputs/mediapipe",
        fps: float = 30.0,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the MediaPipe pose processor.

        Args:
            output_dir: Directory to save pose estimation results
            fps: Video frame rate
            static_image_mode: Whether to process static images or video
            model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
            smooth_landmarks: Whether to smooth landmarks (not used in Tasks API)
            enable_segmentation: Whether to enable segmentation
            smooth_segmentation: Whether to smooth segmentation (not used in Tasks API)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.output_dir = output_dir
        self.fps = fps
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Get model path (downloads if necessary)
        model_path = get_model_path(model_complexity)
        
        # Initialize MediaPipe Pose Landmarker with Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Choose running mode based on static_image_mode
        if static_image_mode:
            running_mode = vision.RunningMode.IMAGE
        else:
            running_mode = vision.RunningMode.VIDEO
        
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=enable_segmentation
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        
        # Track timestamp for video mode
        self.frame_timestamp_ms = 0

        # MediaPipe Pose landmarks mapping (33 landmarks)
        self.mediapipe_landmarks = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

        # Map MediaPipe landmarks to BODY_25 format for compatibility
        self.body_25_keypoints = [
            "Nose",
            "Neck",
            "RShoulder",
            "RElbow",
            "RWrist",
            "LShoulder",
            "LElbow",
            "LWrist",
            "MidHip",
            "RHip",
            "RKnee",
            "RAnkle",
            "LHip",
            "LKnee",
            "LAnkle",
            "REye",
            "LEye",
            "REar",
            "LEar",
            "LBigToe",
            "LSmallToe",
            "LHeel",
            "RBigToe",
            "RSmallToe",
            "RHeel",
        ]

        # Gait-relevant keypoints (prioritized for analysis)
        self.gait_keypoints = [
            "MidHip",
            "RHip",
            "RKnee",
            "RAnkle",
            "RBigToe",
            "RHeel",
            "LHip",
            "LKnee",
            "LAnkle",
            "LBigToe",
            "LHeel",
        ]

        # Mapping from MediaPipe landmarks to BODY_25 format
        self.landmark_mapping = self._create_landmark_mapping()

        logger.info(f"Initialized MediaPipeProcessor with {len(self.mediapipe_landmarks)} landmarks (Tasks API)")

    def _create_landmark_mapping(self) -> Dict[int, int]:
        """
        Create mapping from MediaPipe landmarks to BODY_25 format.

        Returns:
            Dictionary mapping MediaPipe landmark indices to BODY_25 indices
        """
        mapping = {}

        # Direct mappings
        direct_mappings = {
            "nose": 0,  # Nose
            "left_shoulder": 6,  # LShoulder
            "right_shoulder": 2,  # RShoulder
            "left_elbow": 7,  # LElbow
            "right_elbow": 3,  # RElbow
            "left_wrist": 8,  # LWrist
            "right_wrist": 4,  # RWrist
            "left_hip": 12,  # LHip
            "right_hip": 9,  # RHip
            "left_knee": 13,  # LKnee
            "right_knee": 10,  # RKnee
            "left_ankle": 14,  # LAnkle
            "right_ankle": 11,  # RAnkle
            "left_eye": 16,  # LEye
            "right_eye": 15,  # REye
            "left_ear": 18,  # LEar
            "right_ear": 17,  # REar
        }

        # Create mapping for direct matches
        for mp_name, body25_idx in direct_mappings.items():
            if mp_name in self.mediapipe_landmarks:
                mp_idx = self.mediapipe_landmarks.index(mp_name)
                mapping[mp_idx] = body25_idx

        # Calculate MidHip (average of left and right hip)
        left_hip_idx = self.mediapipe_landmarks.index("left_hip")
        right_hip_idx = self.mediapipe_landmarks.index("right_hip")
        mapping[left_hip_idx] = 8  # MidHip (will be calculated)
        mapping[right_hip_idx] = 8  # MidHip (will be calculated)

        # Map foot landmarks (approximate mapping)
        # MediaPipe has heel and foot_index, we'll map to heel and big toe
        if "left_heel" in self.mediapipe_landmarks:
            left_heel_idx = self.mediapipe_landmarks.index("left_heel")
            mapping[left_heel_idx] = 22  # LHeel

        if "right_heel" in self.mediapipe_landmarks:
            right_heel_idx = self.mediapipe_landmarks.index("right_heel")
            mapping[right_heel_idx] = 24  # RHeel

        if "left_foot_index" in self.mediapipe_landmarks:
            left_foot_idx = self.mediapipe_landmarks.index("left_foot_index")
            mapping[left_foot_idx] = 20  # LBigToe (approximate)

        if "right_foot_index" in self.mediapipe_landmarks:
            right_foot_idx = self.mediapipe_landmarks.index("right_foot_index")
            mapping[right_foot_idx] = 23  # RBigToe (approximate)

        return mapping

    def process_video(self, video_path: str) -> bool:
        """
        Process video file and extract pose landmarks.

        Args:
            video_path: Path to input video file

        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing video: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return False

        # Get video FPS for timestamp calculation
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = self.fps
        frame_duration_ms = int(1000 / video_fps)

        frame_count = 0
        frame_data = []
        self.frame_timestamp_ms = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                frame_landmarks = self._process_frame(frame, frame_count)
                if frame_landmarks:
                    frame_data.append(frame_landmarks)

                frame_count += 1
                self.frame_timestamp_ms += frame_duration_ms

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

            # Save results
            self._save_results(frame_data, video_path)
            logger.info(f"Successfully processed {len(frame_data)} frames")
            return True

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
        finally:
            cap.release()

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Optional[Dict]:
        """
        Process a single frame and extract pose landmarks using Tasks API.

        Args:
            frame: Input frame as numpy array
            frame_number: Frame number for identification

        Returns:
            Dictionary containing pose data for the frame
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process with MediaPipe Tasks API
        if self.static_image_mode:
            results = self.pose_landmarker.detect(mp_image)
        else:
            results = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Extract landmarks from first detected pose
        landmarks = results.pose_landmarks[0]

        # Convert to BODY_25 format
        body_25_data = self._convert_to_body_25_format(landmarks, frame.shape)

        # Create frame data structure
        frame_data = {
            "frame_number": frame_number,
            "timestamp": frame_number / self.fps,
            "people": [
                {
                    "person_id": [0],
                    "pose_keypoints_2d": body_25_data,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": [],
                }
            ],
        }

        return frame_data

    def _convert_to_body_25_format(self, landmarks: List, frame_shape: Tuple[int, int, int]) -> List[float]:
        """
        Convert MediaPipe landmarks to BODY_25 format.

        Args:
            landmarks: List of MediaPipe landmarks (NormalizedLandmark objects)
            frame_shape: Original frame shape (height, width, channels)

        Returns:
            List of keypoints in BODY_25 format [x1, y1, conf1, x2, y2, conf2, ...]
        """
        height, width = frame_shape[:2]
        body_25_keypoints = [0.0] * (25 * 3)  # 25 keypoints * 3 values (x, y, confidence)

        # Calculate MidHip from left and right hip
        left_hip_idx = self.mediapipe_landmarks.index("left_hip")
        right_hip_idx = self.mediapipe_landmarks.index("right_hip")
        left_hip = landmarks[left_hip_idx]
        right_hip = landmarks[right_hip_idx]
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        left_hip_vis = left_hip.visibility if hasattr(left_hip, 'visibility') else left_hip.presence
        right_hip_vis = right_hip.visibility if hasattr(right_hip, 'visibility') else right_hip.presence
        mid_hip_conf = (left_hip_vis + right_hip_vis) / 2

        # Calculate Neck from left and right shoulder
        left_shoulder_idx = self.mediapipe_landmarks.index("left_shoulder")
        right_shoulder_idx = self.mediapipe_landmarks.index("right_shoulder")
        left_shoulder = landmarks[left_shoulder_idx]
        right_shoulder = landmarks[right_shoulder_idx]
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2
        left_shoulder_vis = left_shoulder.visibility if hasattr(left_shoulder, 'visibility') else left_shoulder.presence
        right_shoulder_vis = right_shoulder.visibility if hasattr(right_shoulder, 'visibility') else right_shoulder.presence
        neck_conf = (left_shoulder_vis + right_shoulder_vis) / 2

        # Map landmarks to BODY_25 format
        for mp_idx, landmark in enumerate(landmarks):
            if mp_idx in self.landmark_mapping:
                body25_idx = self.landmark_mapping[mp_idx]

                # Convert normalized coordinates to pixel coordinates
                x = landmark.x * width
                y = landmark.y * height
                conf = landmark.visibility if hasattr(landmark, 'visibility') else landmark.presence

                # Store in BODY_25 format
                body_25_keypoints[body25_idx * 3] = x
                body_25_keypoints[body25_idx * 3 + 1] = y
                body_25_keypoints[body25_idx * 3 + 2] = conf

        # Set MidHip and Neck explicitly
        body_25_keypoints[8 * 3] = mid_hip_x * width  # MidHip
        body_25_keypoints[8 * 3 + 1] = mid_hip_y * height
        body_25_keypoints[8 * 3 + 2] = mid_hip_conf

        body_25_keypoints[1 * 3] = neck_x * width  # Neck
        body_25_keypoints[1 * 3 + 1] = neck_y * height
        body_25_keypoints[1 * 3 + 2] = neck_conf

        return body_25_keypoints

    def _save_results(self, frame_data: List[Dict], video_path: str):
        """
        Save pose estimation results to JSON files.

        Args:
            frame_data: List of frame data dictionaries
            video_path: Original video path for naming
        """
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual frame files
        for frame_data_item in frame_data:
            frame_num = frame_data_item["frame_number"]
            filename = f"{video_name}_{timestamp}_{frame_num:06d}.json"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(frame_data_item, f, indent=2)

        # Save summary file
        summary = {
            "video_path": video_path,
            "total_frames": len(frame_data),
            "fps": self.fps,
            "timestamp": timestamp,
            "landmark_mapping": {str(k): v for k, v in self.landmark_mapping.items()},
        }

        summary_path = os.path.join(self.output_dir, f"{video_name}_{timestamp}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved {len(frame_data)} frame files to {self.output_dir}")

    def process_webcam(self, duration: float = 10.0) -> bool:
        """
        Process webcam feed for real-time pose estimation.

        Args:
            duration: Duration to record in seconds

        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Starting webcam processing for {duration} seconds")

        # For webcam, we need to create a new landmarker with IMAGE mode
        # since we process frames independently
        model_path = get_model_path(self.model_complexity)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False
        )
        webcam_landmarker = vision.PoseLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return False

        frame_count = 0
        frame_data = []
        start_time = datetime.now()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check duration
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration:
                    break

                # Process frame with webcam landmarker
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = webcam_landmarker.detect(mp_image)

                frame_landmarks = None
                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    landmarks = results.pose_landmarks[0]
                    body_25_data = self._convert_to_body_25_format(landmarks, frame.shape)
                    frame_landmarks = {
                        "frame_number": frame_count,
                        "timestamp": frame_count / self.fps,
                        "people": [
                            {
                                "person_id": [0],
                                "pose_keypoints_2d": body_25_data,
                                "face_keypoints_2d": [],
                                "hand_left_keypoints_2d": [],
                                "hand_right_keypoints_2d": [],
                                "pose_keypoints_3d": [],
                                "face_keypoints_3d": [],
                                "hand_left_keypoints_3d": [],
                                "hand_right_keypoints_3d": [],
                            }
                        ],
                    }
                    frame_data.append(frame_landmarks)

                # Display frame with landmarks
                self._draw_landmarks_on_frame(frame, results)
                cv2.imshow("MediaPipe Pose", frame)

                if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                    break

                frame_count += 1

            # Save results
            self._save_results(frame_data, "webcam_feed")
            logger.info(f"Successfully processed {len(frame_data)} frames from webcam")
            return True

        except Exception as e:
            logger.error(f"Error processing webcam: {e}")
            return False
        finally:
            cap.release()
            cv2.destroyAllWindows()
            webcam_landmarker.close()

    def _draw_landmarks_on_frame(self, frame: np.ndarray, results):
        """
        Draw pose landmarks on frame for visualization.

        Args:
            frame: Input frame
            results: PoseLandmarkerResult from Tasks API
        """
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return

        landmarks = results.pose_landmarks[0]
        height, width = frame.shape[:2]

        # Draw connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (27, 31), (28, 30), (28, 32)
        ]

        for connection in connections:
            idx1, idx2 = connection
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]
                vis1 = lm1.visibility if hasattr(lm1, 'visibility') else lm1.presence
                vis2 = lm2.visibility if hasattr(lm2, 'visibility') else lm2.presence
                if vis1 > 0.5 and vis2 > 0.5:
                    pt1 = (int(lm1.x * width), int(lm1.y * height))
                    pt2 = (int(lm2.x * width), int(lm2.y * height))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            visibility = landmark.visibility if hasattr(landmark, 'visibility') else landmark.presence
            if visibility > 0.5:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the MediaPipe model.

        Returns:
            Dictionary with model information
        """
        complexity_names = {0: "lite", 1: "full", 2: "heavy"}
        return {
            "name": "MediaPipe Pose Landmarker (Tasks API)",
            "description": "Lightweight, real-time pose estimation by Google using Tasks API",
            "api_version": "Tasks API (0.10.x+)",
            "model_complexity": complexity_names.get(self.model_complexity, "full"),
            "landmarks": 33,
            "keypoints": 25,  # After conversion to BODY_25
            "advantages": [
                "Fast and lightweight",
                "Good for real-time applications",
                "Easy to use and integrate",
                "Works well on CPU",
                "Better cross-platform compatibility with Tasks API",
            ],
            "disadvantages": [
                "Lower accuracy compared to deep learning models",
                "Limited to 2D pose estimation"
            ],
            "best_for": ["Real-time applications", "Mobile/edge devices", "Quick prototyping"],
        }

    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, "pose_landmarker"):
            self.pose_landmarker.close()
        logger.info("MediaPipe resources cleaned up")
