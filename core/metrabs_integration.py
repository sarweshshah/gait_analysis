"""
MeTRAbs Integration Module for Gait Analysis
==========================================

This module provides MeTRAbs-based pose estimation for gait analysis,
offering an alternative to MediaPipe with potentially better accuracy.

Author: Gait Analysis System
"""

import os
import json
import cv2
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. MeTRAbs integration will not work.")
    print("Install PyTorch: pip install torch torchvision torchaudio")
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime
import urllib.request
import zipfile
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeTRAbsProcessor:
    """
    MeTRAbs-based pose estimation processor for gait analysis.
    
    This class handles:
    - MeTRAbs model initialization and configuration
    - Real-time pose estimation from video
    - Keypoint extraction and formatting
    - JSON output generation compatible with existing pipeline
    """
    
    def __init__(self, 
                 output_dir: str = 'metrabs_output',
                 fps: float = 30.0,
                 model_name: str = 'metrabs_4x_512',
                 device: str = 'auto',
                 batch_size: int = 1,
                 num_aug: int = 1):
        """
        Initialize the MeTRAbs pose processor.
        
        Args:
            output_dir: Directory to save pose estimation results
            fps: Video frame rate
            model_name: MeTRAbs model name (e.g., 'metrabs_4x_512', 'metrabs_4x_1024')
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            batch_size: Batch size for inference
            num_aug: Number of augmentations for test-time augmentation
        """
        self.output_dir = output_dir
        self.fps = fps
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_aug = num_aug
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize MeTRAbs model
        self.model = None
        self._setup_metrabs()
        
        # MeTRAbs landmarks mapping (17 COCO keypoints)
        self.metrabs_landmarks = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Map MeTRAbs landmarks to BODY_25 format for compatibility
        self.body_25_keypoints = [
            'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
            'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
            'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel',
            'RBigToe', 'RSmallToe', 'RHeel'
        ]
        
        # Gait-relevant keypoints (prioritized for analysis)
        self.gait_keypoints = [
            'MidHip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RHeel',
            'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LHeel'
        ]
        
        # Mapping from MeTRAbs landmarks to BODY_25 format
        self.landmark_mapping = self._create_landmark_mapping()
        
        logger.info(f"Initialized MeTRAbsProcessor with {len(self.metrabs_landmarks)} landmarks")
        logger.info(f"Using device: {self.device}")
    
    def _setup_metrabs(self):
        """Setup MeTRAbs model and dependencies."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for MeTRAbs integration")
            logger.error("Please install PyTorch: pip install torch torchvision torchaudio")
            raise ImportError("PyTorch not available")
            
        try:
            # Install MeTRAbs if not available
            self._install_metrabs_if_needed()
            
            # Import MeTRAbs
            import metrabs
            from metrabs import models
            
            # Load model
            self.model = models.load_model(self.model_name, device=self.device)
            logger.info(f"Loaded MeTRAbs model: {self.model_name}")
            
        except ImportError as e:
            logger.error(f"Failed to import MeTRAbs: {e}")
            logger.error("Please install MeTRAbs: pip install metrabs")
            raise
        except Exception as e:
            logger.error(f"Failed to setup MeTRAbs: {e}")
            raise
    
    def _install_metrabs_if_needed(self):
        """Install MeTRAbs if not already installed."""
        try:
            import metrabs
            logger.info("MeTRAbs already installed")
        except ImportError:
            logger.info("MeTRAbs not found. Installing from GitHub...")
            import subprocess
            import sys
            
            try:
                # Install from GitHub repository
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "git+https://github.com/isarandi/metrabs.git"
                ])
                logger.info("MeTRAbs installed successfully from GitHub")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install MeTRAbs: {e}")
                logger.error("Please install manually using:")
                logger.error("pip install git+https://github.com/isarandi/metrabs.git")
                logger.error("Or run: ./install_metrabs.sh (Linux/Mac) or install_metrabs.bat (Windows)")
                raise
    
    def _create_landmark_mapping(self) -> Dict[int, int]:
        """
        Create mapping from MeTRAbs landmarks to BODY_25 format.
        
        Returns:
            Dictionary mapping MeTRAbs landmark indices to BODY_25 indices
        """
        mapping = {}
        
        # Direct mappings
        direct_mappings = {
            'nose': 0,  # Nose
            'left_shoulder': 6,  # LShoulder
            'right_shoulder': 2,  # RShoulder
            'left_elbow': 7,  # LElbow
            'right_elbow': 3,  # RElbow
            'left_wrist': 8,  # LWrist
            'right_wrist': 4,  # RWrist
            'left_hip': 12,  # LHip
            'right_hip': 9,  # RHip
            'left_knee': 13,  # LKnee
            'right_knee': 10,  # RKnee
            'left_ankle': 14,  # LAnkle
            'right_ankle': 11,  # RAnkle
            'left_eye': 16,  # LEye
            'right_eye': 15,  # REye
            'left_ear': 18,  # LEar
            'right_ear': 17,  # REar
        }
        
        # Create mapping for direct matches
        for metrabs_name, body25_idx in direct_mappings.items():
            if metrabs_name in self.metrabs_landmarks:
                metrabs_idx = self.metrabs_landmarks.index(metrabs_name)
                mapping[metrabs_idx] = body25_idx
        
        # Calculate MidHip (average of left and right hip)
        left_hip_idx = self.metrabs_landmarks.index('left_hip')
        right_hip_idx = self.metrabs_landmarks.index('right_hip')
        mapping[left_hip_idx] = 8  # MidHip (will be calculated)
        mapping[right_hip_idx] = 8  # MidHip (will be calculated)
        
        return mapping
    
    def process_video(self, video_path: str) -> bool:
        """
        Process video file and extract pose landmarks.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing video with MeTRAbs: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return False
        
        frame_count = 0
        frame_data = []
        frames_batch = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to batch
                frames_batch.append(frame)
                
                # Process batch when full
                if len(frames_batch) >= self.batch_size:
                    batch_results = self._process_batch(frames_batch, frame_count)
                    frame_data.extend(batch_results)
                    frames_batch = []
                    frame_count += len(batch_results)
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
            
            # Process remaining frames
            if frames_batch:
                batch_results = self._process_batch(frames_batch, frame_count)
                frame_data.extend(batch_results)
            
            # Save results
            self._save_results(frame_data, video_path)
            logger.info(f"Successfully processed {len(frame_data)} frames with MeTRAbs")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
        finally:
            cap.release()
    
    def _process_batch(self, frames: List[np.ndarray], start_frame: int) -> List[Dict]:
        """
        Process a batch of frames with MeTRAbs.
        
        Args:
            frames: List of input frames
            start_frame: Starting frame number
            
        Returns:
            List of frame data dictionaries
        """
        if not self.model:
            return []
        
        try:
            # Convert frames to RGB and normalize
            rgb_frames = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb_frame)
            
            # Stack frames into batch
            batch = np.stack(rgb_frames)
            
            # Run inference
            with torch.no_grad():
                pred = self.model.predict_single_batch(
                    batch, 
                    num_aug=self.num_aug,
                    max_person_instances=1
                )
            
            # Extract keypoints
            keypoints_2d = pred['poses2d']  # Shape: (batch, num_persons, num_keypoints, 3)
            
            frame_data = []
            for i, frame in enumerate(frames):
                frame_num = start_frame + i
                
                if i < keypoints_2d.shape[0] and keypoints_2d.shape[1] > 0:
                    # Get first person's keypoints
                    person_keypoints = keypoints_2d[i, 0]  # Shape: (num_keypoints, 3)
                    
                    # Convert to BODY_25 format
                    body_25_data = self._convert_to_body_25_format(person_keypoints, frame.shape)
                    
                    # Create frame data structure
                    frame_data_item = {
                        "frame_number": frame_num,
                        "timestamp": frame_num / self.fps,
                        "people": [{
                            "person_id": [0],
                            "pose_keypoints_2d": body_25_data,
                            "face_keypoints_2d": [],
                            "hand_left_keypoints_2d": [],
                            "hand_right_keypoints_2d": [],
                            "pose_keypoints_3d": [],
                            "face_keypoints_3d": [],
                            "hand_left_keypoints_3d": [],
                            "hand_right_keypoints_3d": []
                        }]
                    }
                else:
                    # No person detected
                    frame_data_item = {
                        "frame_number": frame_num,
                        "timestamp": frame_num / self.fps,
                        "people": []
                    }
                
                frame_data.append(frame_data_item)
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []
    
    def _convert_to_body_25_format(self, keypoints: np.ndarray, frame_shape: Tuple[int, int, int]) -> List[float]:
        """
        Convert MeTRAbs keypoints to BODY_25 format.
        
        Args:
            keypoints: MeTRAbs keypoints array (num_keypoints, 3)
            frame_shape: Original frame shape (height, width, channels)
            
        Returns:
            List of keypoints in BODY_25 format [x1, y1, conf1, x2, y2, conf2, ...]
        """
        height, width = frame_shape[:2]
        body_25_keypoints = [0.0] * (25 * 3)  # 25 keypoints * 3 values (x, y, confidence)
        
        # Calculate MidHip from left and right hip
        left_hip_idx = self.metrabs_landmarks.index('left_hip')
        right_hip_idx = self.metrabs_landmarks.index('right_hip')
        
        if left_hip_idx < keypoints.shape[0] and right_hip_idx < keypoints.shape[0]:
            left_hip = keypoints[left_hip_idx]
            right_hip = keypoints[right_hip_idx]
            mid_hip_x = (left_hip[0] + right_hip[0]) / 2
            mid_hip_y = (left_hip[1] + right_hip[1]) / 2
            mid_hip_conf = (left_hip[2] + right_hip[2]) / 2
        else:
            mid_hip_x, mid_hip_y, mid_hip_conf = 0, 0, 0
        
        # Calculate Neck from left and right shoulder
        left_shoulder_idx = self.metrabs_landmarks.index('left_shoulder')
        right_shoulder_idx = self.metrabs_landmarks.index('right_shoulder')
        
        if left_shoulder_idx < keypoints.shape[0] and right_shoulder_idx < keypoints.shape[0]:
            left_shoulder = keypoints[left_shoulder_idx]
            right_shoulder = keypoints[right_shoulder_idx]
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
            neck_conf = (left_shoulder[2] + right_shoulder[2]) / 2
        else:
            neck_x, neck_y, neck_conf = 0, 0, 0
        
        # Map landmarks to BODY_25 format
        for metrabs_idx, keypoint in enumerate(keypoints):
            if metrabs_idx in self.landmark_mapping:
                body25_idx = self.landmark_mapping[metrabs_idx]
                
                # MeTRAbs keypoints are already in pixel coordinates
                x = keypoint[0]
                y = keypoint[1]
                conf = keypoint[2]
                
                # Store in BODY_25 format
                body_25_keypoints[body25_idx * 3] = x
                body_25_keypoints[body25_idx * 3 + 1] = y
                body_25_keypoints[body25_idx * 3 + 2] = conf
        
        # Set MidHip and Neck explicitly
        body_25_keypoints[8 * 3] = mid_hip_x  # MidHip
        body_25_keypoints[8 * 3 + 1] = mid_hip_y
        body_25_keypoints[8 * 3 + 2] = mid_hip_conf
        
        body_25_keypoints[1 * 3] = neck_x  # Neck
        body_25_keypoints[1 * 3 + 1] = neck_y
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
            
            with open(filepath, 'w') as f:
                json.dump(frame_data_item, f, indent=2)
        
        # Save summary file
        summary = {
            "video_path": video_path,
            "total_frames": len(frame_data),
            "fps": self.fps,
            "timestamp": timestamp,
            "model_name": self.model_name,
            "device": self.device,
            "landmark_mapping": self.landmark_mapping
        }
        
        summary_path = os.path.join(self.output_dir, f"{video_name}_{timestamp}_summary.json")
        with open(summary_path, 'w') as f:
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
        logger.info(f"Starting webcam processing with MeTRAbs for {duration} seconds")
        
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
                
                # Process single frame
                batch_results = self._process_batch([frame], frame_count)
                if batch_results:
                    frame_data.append(batch_results[0])
                
                # Display frame with landmarks
                self._draw_landmarks(frame, batch_results[0] if batch_results else None)
                cv2.imshow('MeTRAbs Pose', frame)
                
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
    
    def _draw_landmarks(self, frame: np.ndarray, frame_data: Optional[Dict]):
        """
        Draw pose landmarks on frame for visualization.
        
        Args:
            frame: Input frame
            frame_data: Frame data with landmarks
        """
        if not frame_data or not frame_data.get("people"):
            return
        
        # Extract keypoints from frame data
        keypoints = frame_data["people"][0]["pose_keypoints_2d"]
        
        # Draw keypoints
        for i in range(0, len(keypoints), 3):
            x, y, conf = keypoints[i:i+3]
            if conf > 0.5:  # Only draw confident landmarks
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Draw connections (basic skeleton)
        connections = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),  # Upper body
            (1, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),  # Lower body
            (0, 1), (0, 15), (0, 16), (15, 17), (16, 18)  # Head
        ]
        
        for connection in connections:
            if (connection[0] * 3 + 2 < len(keypoints) and 
                connection[1] * 3 + 2 < len(keypoints)):
                x1, y1, conf1 = keypoints[connection[0] * 3:connection[0] * 3 + 3]
                x2, y2, conf2 = keypoints[connection[1] * 3:connection[1] * 3 + 3]
                
                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    def cleanup(self):
        """Clean up MeTRAbs resources."""
        if hasattr(self, 'model') and self.model:
            del self.model
        logger.info("MeTRAbs resources cleaned up")
