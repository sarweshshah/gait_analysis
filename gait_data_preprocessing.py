"""
Gait Analysis Data Preprocessing Module
=======================================

This module handles the complete data preprocessing pipeline for markerless gait analysis
using MediaPipe pose estimation with foot keypoints.

Author: Gait Analysis System
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import butter, filtfilt
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GaitDataPreprocessor:
    """
    Comprehensive data preprocessor for MediaPipe gait analysis.
    
    This class handles:
    - MediaPipe pose estimation integration
    - Data cleaning and gap-filling
    - Low-pass filtering
    - Coordinate normalization
    - Feature engineering for TCN input
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.3,
                 filter_cutoff: float = 6.0,
                 filter_order: int = 4,
                 window_size: int = 30,  # ~1 second at 30fps
                 overlap: float = 0.5):
        """
        Initialize the gait data preprocessor.
        
        Args:
            confidence_threshold: Minimum confidence for keypoint detection
            filter_cutoff: Low-pass filter cutoff frequency (Hz)
            filter_order: Butterworth filter order
            window_size: Number of frames for TCN input window
            overlap: Overlap ratio between consecutive windows
        """
        self.confidence_threshold = confidence_threshold
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.window_size = window_size
        self.overlap = overlap
        
        # MediaPipe keypoint mapping (33 landmarks, mapped to BODY_25 format)
        self.keypoint_mapping = [
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
        
        # Keypoint indices for gait analysis
        self.gait_indices = [self.keypoint_mapping.index(kp) for kp in self.gait_keypoints]
        
        logger.info(f"Initialized GaitDataPreprocessor with {len(self.gait_keypoints)} gait keypoints")
    
    def load_mediapipe_data(self, json_directory: str) -> List[Dict]:
        """
        Load MediaPipe JSON files from directory.
        
        Args:
            json_directory: Path to directory containing MediaPipe JSON files
            
        Returns:
            List of frame data dictionaries
        """
        json_files = sorted([f for f in os.listdir(json_directory) if f.endswith('.json')])
        frame_data = []
        
        for json_file in json_files:
            file_path = os.path.join(json_directory, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    frame_data.append(data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(frame_data)} frames from {json_directory}")
        return frame_data
    
    def extract_keypoints_from_frame(self, frame_data: Dict) -> np.ndarray:
        """
        Extract keypoints from a single frame's MediaPipe data.
        
        Args:
            frame_data: MediaPipe JSON data for one frame
            
        Returns:
            Array of shape (25, 3) with (x, y, confidence) for each keypoint
        """
        if 'people' not in frame_data or len(frame_data['people']) == 0:
            # No person detected, return zeros
            return np.zeros((25, 3))
        
        # Get the first person's keypoints (assuming single person)
        person = frame_data['people'][0]
        keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)
        
        # Ensure we have exactly 25 keypoints (BODY_25 format)
        if keypoints.shape[0] < 25:
            # Pad with zeros if fewer keypoints
            padding = np.zeros((25 - keypoints.shape[0], 3))
            keypoints = np.vstack([keypoints, padding])
        elif keypoints.shape[0] > 25:
            # Truncate if more keypoints
            keypoints = keypoints[:25]
        
        return keypoints
    
    def clean_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Clean keypoints by removing low-confidence detections.
        
        Args:
            keypoints: Array of shape (25, 3) with (x, y, confidence)
            
        Returns:
            Cleaned keypoints with low-confidence points set to NaN
        """
        cleaned = keypoints.copy()
        # Set low-confidence keypoints to NaN
        low_confidence_mask = keypoints[:, 2] < self.confidence_threshold
        cleaned[low_confidence_mask, :2] = np.nan
        return cleaned
    
    def interpolate_missing_keypoints(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        Fill missing keypoints using cubic spline interpolation.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Interpolated keypoints sequence
        """
        n_frames, n_keypoints, n_coords = keypoints_sequence.shape
        interpolated = keypoints_sequence.copy()
        
        for kp_idx in range(n_keypoints):
            for coord_idx in range(2):  # Only interpolate x, y coordinates
                # Get valid time points and values
                valid_mask = ~np.isnan(keypoints_sequence[:, kp_idx, coord_idx])
                if np.sum(valid_mask) < 3:  # Need at least 3 points for cubic spline
                    continue
                
                valid_times = np.where(valid_mask)[0]
                valid_values = keypoints_sequence[valid_mask, kp_idx, coord_idx]
                
                if len(valid_times) > 3:
                    # Use cubic spline interpolation
                    spline = interpolate.interp1d(valid_times, valid_values, 
                                                kind='cubic', bounds_error=False, 
                                                fill_value='extrapolate')
                else:
                    # Use linear interpolation for few points
                    spline = interpolate.interp1d(valid_times, valid_values, 
                                                kind='linear', bounds_error=False, 
                                                fill_value='extrapolate')
                
                # Interpolate missing values
                all_times = np.arange(n_frames)
                interpolated_values = spline(all_times)
                
                # Replace only NaN values
                nan_mask = np.isnan(keypoints_sequence[:, kp_idx, coord_idx])
                interpolated[nan_mask, kp_idx, coord_idx] = interpolated_values[nan_mask]
        
        return interpolated
    
    def apply_low_pass_filter(self, keypoints_sequence: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        Apply low-pass filter to reduce noise in keypoint coordinates.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            fps: Frames per second for filter design
            
        Returns:
            Filtered keypoints sequence
        """
        # Design Butterworth low-pass filter
        nyquist = fps / 2.0
        normalized_cutoff = self.filter_cutoff / nyquist
        b, a = butter(self.filter_order, normalized_cutoff, btype='low')
        
        filtered = keypoints_sequence.copy()
        n_frames, n_keypoints, n_coords = keypoints_sequence.shape
        
        for kp_idx in range(n_keypoints):
            for coord_idx in range(2):  # Only filter x, y coordinates
                signal = keypoints_sequence[:, kp_idx, coord_idx]
                if not np.any(np.isnan(signal)):
                    filtered[:, kp_idx, coord_idx] = filtfilt(b, a, signal)
        
        return filtered
    
    def normalize_coordinates(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates using shoulder-hip distance as reference.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Normalized keypoints sequence
        """
        normalized = keypoints_sequence.copy()
        n_frames = keypoints_sequence.shape[0]
        
        for frame_idx in range(n_frames):
            frame_keypoints = keypoints_sequence[frame_idx]
            
            # Calculate shoulder-hip distance as normalization factor
            # Use right shoulder (2) and right hip (9) for BODY_25
            shoulder_pos = frame_keypoints[2, :2]  # RShoulder
            hip_pos = frame_keypoints[9, :2]       # RHip
            
            if not (np.any(np.isnan(shoulder_pos)) or np.any(np.isnan(hip_pos))):
                scale_factor = np.linalg.norm(shoulder_pos - hip_pos)
                
                if scale_factor > 0:
                    # Normalize all coordinates
                    normalized[frame_idx, :, :2] = frame_keypoints[:, :2] / scale_factor
                    
                    # Center coordinates around hip midpoint
                    hip_midpoint = (frame_keypoints[9, :2] + frame_keypoints[12, :2]) / 2  # RHip + LHip
                    if not np.any(np.isnan(hip_midpoint)):
                        normalized[frame_idx, :, :2] -= hip_midpoint
        
        return normalized
    
    def extract_gait_features(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        Extract gait-specific features from keypoints.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Feature array of shape (n_frames, n_features)
        """
        n_frames = keypoints_sequence.shape[0]
        features = []
        
        for frame_idx in range(n_frames):
            frame_keypoints = keypoints_sequence[frame_idx]
            frame_features = []
            
            # Extract gait-relevant keypoints
            gait_kps = frame_keypoints[self.gait_indices, :2]
            
            # 1. Joint positions (x, y coordinates)
            for kp in gait_kps:
                if not np.any(np.isnan(kp)):
                    frame_features.extend(kp)
                else:
                    frame_features.extend([0, 0])
            
            # 2. Joint angles (if enough keypoints are available)
            angles = self._calculate_joint_angles(frame_keypoints)
            frame_features.extend(angles)
            
            # 3. Relative positions (ankle relative to hip)
            relative_positions = self._calculate_relative_positions(frame_keypoints)
            frame_features.extend(relative_positions)
            
            # 4. Velocity features (if not first frame)
            if frame_idx > 0:
                prev_keypoints = keypoints_sequence[frame_idx - 1]
                velocities = self._calculate_velocities(frame_keypoints, prev_keypoints)
                frame_features.extend(velocities)
            else:
                # Zero velocities for first frame
                frame_features.extend([0] * len(self.gait_indices) * 2)
            
            features.append(frame_features)
        
        return np.array(features)
    
    def _calculate_joint_angles(self, keypoints: np.ndarray) -> List[float]:
        """Calculate joint angles for gait analysis."""
        angles = []
        
        # Hip-Knee-Ankle angles
        for side in ['right', 'left']:
            if side == 'right':
                hip_idx, knee_idx, ankle_idx = 9, 10, 11  # RHip, RKnee, RAnkle
            else:
                hip_idx, knee_idx, ankle_idx = 12, 13, 14  # LHip, LKnee, LAnkle
            
            hip = keypoints[hip_idx, :2]
            knee = keypoints[knee_idx, :2]
            ankle = keypoints[ankle_idx, :2]
            
            if not (np.any(np.isnan(hip)) or np.any(np.isnan(knee)) or np.any(np.isnan(ankle))):
                angle = self._calculate_angle(hip, knee, ankle)
                angles.append(angle)
            else:
                angles.append(0)
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p2 is the vertex)."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_relative_positions(self, keypoints: np.ndarray) -> List[float]:
        """Calculate relative positions for gait analysis."""
        relative_positions = []
        
        # Ankle positions relative to hip midpoint
        hip_midpoint = (keypoints[9, :2] + keypoints[12, :2]) / 2  # RHip + LHip
        
        if not np.any(np.isnan(hip_midpoint)):
            for ankle_idx in [11, 14]:  # RAnkle, LAnkle
                ankle = keypoints[ankle_idx, :2]
                if not np.any(np.isnan(ankle)):
                    relative = ankle - hip_midpoint
                    relative_positions.extend(relative)
                else:
                    relative_positions.extend([0, 0])
        else:
            relative_positions.extend([0, 0, 0, 0])
        
        return relative_positions
    
    def _calculate_velocities(self, current_kps: np.ndarray, prev_kps: np.ndarray) -> List[float]:
        """Calculate velocities of keypoints."""
        velocities = []
        
        for idx in self.gait_indices:
            current = current_kps[idx, :2]
            prev = prev_kps[idx, :2]
            
            if not (np.any(np.isnan(current)) or np.any(np.isnan(prev))):
                velocity = current - prev
                velocities.extend(velocity)
            else:
                velocities.extend([0, 0])
        
        return velocities
    
    def create_tcn_windows(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create fixed-length windows for TCN input.
        
        Args:
            features: Feature array of shape (n_frames, n_features)
            
        Returns:
            Tuple of (windows, labels) where labels are placeholder zeros
        """
        n_frames, n_features = features.shape
        step_size = int(self.window_size * (1 - self.overlap))
        
        windows = []
        labels = []
        
        for start_idx in range(0, n_frames - self.window_size + 1, step_size):
            end_idx = start_idx + self.window_size
            window = features[start_idx:end_idx]
            
            # Only include complete windows
            if window.shape[0] == self.window_size:
                windows.append(window)
                labels.append(0)  # Placeholder label
        
        return np.array(windows), np.array(labels)
    
    def process_video_sequence(self, json_directory: str, fps: float = 30.0) -> Dict:
        """
        Complete preprocessing pipeline for a video sequence.
        
        Args:
            json_directory: Path to MediaPipe JSON files
            fps: Frames per second of the video
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Processing video sequence from {json_directory}")
        
        # 1. Load MediaPipe data
        frame_data = self.load_mediapipe_data(json_directory)
        
        if len(frame_data) == 0:
            raise ValueError("No valid MediaPipe data found")
        
        # 2. Extract keypoints
        keypoints_sequence = []
        for frame_data_item in frame_data:
            keypoints = self.extract_keypoints_from_frame(frame_data_item)
            keypoints_sequence.append(keypoints)
        
        keypoints_sequence = np.array(keypoints_sequence)
        logger.info(f"Extracted keypoints: {keypoints_sequence.shape}")
        
        # 3. Clean keypoints
        cleaned_keypoints = self.clean_keypoints(keypoints_sequence)
        logger.info("Cleaned keypoints")
        
        # 4. Interpolate missing keypoints
        interpolated_keypoints = self.interpolate_missing_keypoints(cleaned_keypoints)
        logger.info("Interpolated missing keypoints")
        
        # 5. Apply low-pass filter
        filtered_keypoints = self.apply_low_pass_filter(interpolated_keypoints, fps)
        logger.info("Applied low-pass filter")
        
        # 6. Normalize coordinates
        normalized_keypoints = self.normalize_coordinates(filtered_keypoints)
        logger.info("Normalized coordinates")
        
        # 7. Extract features
        features = self.extract_gait_features(normalized_keypoints)
        logger.info(f"Extracted features: {features.shape}")
        
        # 8. Create TCN windows
        windows, labels = self.create_tcn_windows(features)
        logger.info(f"Created TCN windows: {windows.shape}")
        
        return {
            'keypoints_sequence': normalized_keypoints,
            'features': features,
            'windows': windows,
            'labels': labels,
            'metadata': {
                'n_frames': len(frame_data),
                'fps': fps,
                'window_size': self.window_size,
                'n_features': features.shape[1],
                'n_windows': len(windows)
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    preprocessor = GaitDataPreprocessor(
        confidence_threshold=0.3,
        filter_cutoff=6.0,
        window_size=30
    )
    
    # Process a video sequence (replace with actual path)
    # result = preprocessor.process_video_sequence("path/to/mediapipe/json/files")
    # print(f"Processed {result['metadata']['n_windows']} windows")
