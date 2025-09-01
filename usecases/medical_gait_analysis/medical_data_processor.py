"""
Medical Gait Data Processor Module
==================================

Specialized data preprocessing for medical gait analysis.
Extends the base GaitDataPreprocessor with medical-specific features and processing.

Author: Medical Gait Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
from pathlib import Path

# Import base preprocessor
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from core.gait_data_preprocessing import GaitDataPreprocessor
from .medical_gait_events import HydrocephalusGaitEvents, MedicalGaitEvents

logger = logging.getLogger(__name__)

class MedicalGaitDataProcessor(GaitDataPreprocessor):
    """
    Enhanced data processor for medical gait analysis.
    
    Extends the base processor with:
    - Medical condition-specific feature extraction
    - Clinical gait parameter calculation
    - Pathological gait pattern detection
    - Medical dataset handling
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.3,
                 filter_cutoff: float = 6.0,
                 filter_order: int = 4,
                 window_size: int = 30,
                 overlap: float = 0.5,
                 medical_condition: str = "hydrocephalus"):
        """
        Initialize medical gait data processor.
        
        Args:
            confidence_threshold: Minimum confidence for keypoint detection
            filter_cutoff: Low-pass filter cutoff frequency (Hz)
            filter_order: Butterworth filter order
            window_size: Number of frames for analysis window
            overlap: Overlap ratio between consecutive windows
            medical_condition: Target medical condition for specialized processing
        """
        super().__init__(confidence_threshold, filter_cutoff, filter_order, window_size, overlap)
        
        self.medical_condition = medical_condition.lower()
        
        # Initialize medical-specific event detector
        if self.medical_condition == "hydrocephalus":
            self.event_detector = HydrocephalusGaitEvents()
        else:
            # Default to base medical events
            self.event_detector = MedicalGaitEvents()
        
        # Medical-specific keypoints of interest
        self.medical_keypoints = {
            'hydrocephalus': [
                'MidHip', 'RHip', 'LHip', 'RKnee', 'LKnee', 
                'RAnkle', 'LAnkle', 'RBigToe', 'LBigToe', 'RHeel', 'LHeel'
            ],
            'parkinsons': [
                'Neck', 'MidHip', 'RHip', 'LHip', 'RShoulder', 'LShoulder',
                'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RKnee', 'LKnee'
            ],
            'stroke': [
                'MidHip', 'RHip', 'LHip', 'RKnee', 'LKnee', 
                'RAnkle', 'LAnkle', 'RShoulder', 'LShoulder'
            ]
        }
        
        logger.info(f"Initialized MedicalGaitDataProcessor for {medical_condition}")
    
    def process_medical_video_sequence(self, json_directory: str, 
                                     patient_metadata: Dict[str, Any] = None,
                                     fps: float = 30.0) -> Dict[str, Any]:
        """
        Complete medical gait analysis pipeline for a video sequence.
        
        Args:
            json_directory: Path to MediaPipe JSON files
            patient_metadata: Optional patient information (age, diagnosis, etc.)
            fps: Frames per second of the video
            
        Returns:
            Dictionary containing processed data, events, and medical metrics
        """
        logger.info(f"Processing medical gait sequence from {json_directory}")
        
        # Run base preprocessing
        base_result = self.process_video_sequence(json_directory, fps)
        
        # Extract medical-specific events
        events = self.event_detector.detect_events(base_result['keypoints_sequence'])
        
        # Calculate medical metrics
        if isinstance(self.event_detector, HydrocephalusGaitEvents):
            medical_metrics = self.event_detector.calculate_hydrocephalus_metrics(
                events, base_result['keypoints_sequence']
            )
        else:
            medical_metrics = self.event_detector.calculate_temporal_parameters(events)
        
        # Extract clinical features
        clinical_features = self.extract_clinical_features(
            base_result['keypoints_sequence'], events, medical_metrics
        )
        
        # Calculate asymmetry indices
        asymmetry_metrics = self.calculate_asymmetry_indices(base_result['keypoints_sequence'])
        
        # Combine all metrics
        all_metrics = {**medical_metrics, **asymmetry_metrics}
        
        # Add patient metadata if provided
        if patient_metadata:
            all_metrics.update(patient_metadata)
        
        return {
            **base_result,
            'events': events,
            'medical_metrics': all_metrics,
            'clinical_features': clinical_features,
            'medical_condition': self.medical_condition,
            'patient_metadata': patient_metadata or {}
        }
    
    def extract_clinical_features(self, keypoints_sequence: np.ndarray, 
                                events: Dict[str, List], 
                                metrics: Dict[str, float]) -> np.ndarray:
        """
        Extract clinical-grade features for medical analysis.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            events: Dictionary of detected gait events
            metrics: Dictionary of calculated metrics
            
        Returns:
            Clinical feature vector
        """
        features = []
        
        # 1. Temporal-spatial parameters
        temporal_features = [
            metrics.get('stride_time_mean', 0),
            metrics.get('stride_time_std', 0),
            metrics.get('stride_time_cv', 0),
            metrics.get('step_time_mean', 0),
            metrics.get('step_time_std', 0),
            metrics.get('step_time_asymmetry', 0),
            metrics.get('stance_time_mean', 0),
            metrics.get('swing_time_mean', 0)
        ]
        features.extend(temporal_features)
        
        # 2. Pathological indicators
        if self.medical_condition == "hydrocephalus":
            pathological_features = [
                metrics.get('shuffling_percentage', 0),
                metrics.get('magnetic_gait_percentage', 0),
                metrics.get('freezing_episode_count', 0),
                metrics.get('wide_base_percentage', 0),
                metrics.get('step_height_cv', 0)
            ]
            features.extend(pathological_features)
        
        # 3. Kinematic features
        kinematic_features = self._extract_kinematic_features(keypoints_sequence)
        features.extend(kinematic_features)
        
        # 4. Stability measures
        stability_features = self._extract_stability_features(keypoints_sequence)
        features.extend(stability_features)
        
        return np.array(features)
    
    def _extract_kinematic_features(self, keypoints_sequence: np.ndarray) -> List[float]:
        """Extract kinematic features from joint trajectories."""
        features = []
        
        # Joint angle ranges and velocities
        joint_pairs = [
            (9, 10, 11),   # RHip-RKnee-RAnkle
            (12, 13, 14),  # LHip-LKnee-LAnkle
        ]
        
        for hip_idx, knee_idx, ankle_idx in joint_pairs:
            # Calculate joint angles over time
            angles = []
            for frame in keypoints_sequence:
                hip = frame[hip_idx, :2]
                knee = frame[knee_idx, :2]
                ankle = frame[ankle_idx, :2]
                
                if not (np.any(np.isnan(hip)) or np.any(np.isnan(knee)) or np.any(np.isnan(ankle))):
                    angle = self._calculate_angle(hip, knee, ankle)
                    angles.append(angle)
            
            if angles:
                features.extend([
                    np.mean(angles),      # Mean angle
                    np.std(angles),       # Angle variability
                    np.max(angles) - np.min(angles),  # Range of motion
                ])
            else:
                features.extend([0, 0, 0])
        
        return features
    
    def _extract_stability_features(self, keypoints_sequence: np.ndarray) -> List[float]:
        """Extract gait stability features."""
        features = []
        
        # Center of mass approximation (hip midpoint)
        com_trajectory = []
        for frame in keypoints_sequence:
            right_hip = frame[9, :2]  # RHip
            left_hip = frame[12, :2]  # LHip
            
            if not (np.any(np.isnan(right_hip)) or np.any(np.isnan(left_hip))):
                com = (right_hip + left_hip) / 2
                com_trajectory.append(com)
            else:
                com_trajectory.append([np.nan, np.nan])
        
        com_trajectory = np.array(com_trajectory)
        
        # COM velocity and acceleration
        if len(com_trajectory) > 2:
            com_velocity = np.gradient(com_trajectory, axis=0)
            com_acceleration = np.gradient(com_velocity, axis=0)
            
            # Stability metrics
            features.extend([
                np.nanstd(com_trajectory[:, 0]),      # Lateral COM variability
                np.nanstd(com_trajectory[:, 1]),      # Vertical COM variability
                np.nanmean(np.linalg.norm(com_velocity, axis=1)),  # Mean COM velocity
                np.nanstd(np.linalg.norm(com_velocity, axis=1)),   # COM velocity variability
                np.nanmean(np.linalg.norm(com_acceleration, axis=1))  # Mean COM acceleration
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def calculate_asymmetry_indices(self, keypoints_sequence: np.ndarray) -> Dict[str, float]:
        """
        Calculate bilateral asymmetry indices for medical assessment.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Dictionary of asymmetry metrics
        """
        asymmetry_metrics = {}
        
        # Joint pairs for asymmetry analysis
        joint_pairs = {
            'hip': (9, 12),      # RHip, LHip
            'knee': (10, 13),    # RKnee, LKnee
            'ankle': (11, 14),   # RAnkle, LAnkle
        }
        
        for joint_name, (right_idx, left_idx) in joint_pairs.items():
            right_trajectory = keypoints_sequence[:, right_idx, :2]
            left_trajectory = keypoints_sequence[:, left_idx, :2]
            
            # Calculate range of motion for each side
            right_rom = self._calculate_range_of_motion(right_trajectory)
            left_rom = self._calculate_range_of_motion(left_trajectory)
            
            # Asymmetry index: |R - L| / ((R + L) / 2) * 100
            if (right_rom + left_rom) > 0:
                asymmetry = abs(right_rom - left_rom) / ((right_rom + left_rom) / 2) * 100
                asymmetry_metrics[f'{joint_name}_asymmetry'] = asymmetry
            else:
                asymmetry_metrics[f'{joint_name}_asymmetry'] = 0
        
        # Step length asymmetry
        step_lengths_right, step_lengths_left = self._calculate_step_lengths(keypoints_sequence)
        
        if step_lengths_right and step_lengths_left:
            right_mean = np.mean(step_lengths_right)
            left_mean = np.mean(step_lengths_left)
            
            if (right_mean + left_mean) > 0:
                step_asymmetry = abs(right_mean - left_mean) / ((right_mean + left_mean) / 2) * 100
                asymmetry_metrics['step_length_asymmetry'] = step_asymmetry
            else:
                asymmetry_metrics['step_length_asymmetry'] = 0
        
        return asymmetry_metrics
    
    def _calculate_range_of_motion(self, trajectory: np.ndarray) -> float:
        """Calculate range of motion for a joint trajectory."""
        if len(trajectory) == 0:
            return 0
        
        valid_points = trajectory[~np.isnan(trajectory).any(axis=1)]
        if len(valid_points) < 2:
            return 0
        
        # Calculate range in both x and y directions
        x_range = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
        y_range = np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])
        
        # Return combined range of motion
        return np.sqrt(x_range**2 + y_range**2)
    
    def _calculate_step_lengths(self, keypoints_sequence: np.ndarray) -> Tuple[List[float], List[float]]:
        """Calculate step lengths for both legs."""
        right_ankle = keypoints_sequence[:, 11, :2]  # RAnkle
        left_ankle = keypoints_sequence[:, 14, :2]   # LAnkle
        
        right_steps = []
        left_steps = []
        
        # Simple step detection based on ankle forward movement
        for i in range(1, len(keypoints_sequence)):
            # Right step
            if not (np.any(np.isnan(right_ankle[i])) or np.any(np.isnan(right_ankle[i-1]))):
                step_vector = right_ankle[i] - right_ankle[i-1]
                step_length = np.linalg.norm(step_vector)
                if step_length > 0.01:  # Minimum step threshold
                    right_steps.append(step_length)
            
            # Left step
            if not (np.any(np.isnan(left_ankle[i])) or np.any(np.isnan(left_ankle[i-1]))):
                step_vector = left_ankle[i] - left_ankle[i-1]
                step_length = np.linalg.norm(step_vector)
                if step_length > 0.01:  # Minimum step threshold
                    left_steps.append(step_length)
        
        return right_steps, left_steps
    
    def create_medical_dataset(self, data_directory: str, 
                             labels_file: str = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create a medical dataset from multiple patient videos.
        
        Args:
            data_directory: Directory containing patient video subdirectories
            labels_file: Optional CSV file with patient labels and metadata
            
        Returns:
            Tuple of (features, labels, metadata_list)
        """
        logger.info(f"Creating medical dataset from {data_directory}")
        
        # Load labels if provided
        labels_df = None
        if labels_file and os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        # Process each patient directory
        for patient_dir in os.listdir(data_directory):
            patient_path = os.path.join(data_directory, patient_dir)
            
            if not os.path.isdir(patient_path):
                continue
            
            logger.info(f"Processing patient: {patient_dir}")
            
            try:
                # Get patient metadata from labels file
                patient_metadata = {}
                patient_label = 0  # Default label
                
                if labels_df is not None:
                    patient_row = labels_df[labels_df['patient_id'] == patient_dir]
                    if not patient_row.empty:
                        patient_metadata = patient_row.iloc[0].to_dict()
                        patient_label = patient_metadata.get('label', 0)
                
                # Process patient video
                result = self.process_medical_video_sequence(
                    patient_path, patient_metadata
                )
                
                # Extract features for ML
                features = result['clinical_features']
                
                all_features.append(features)
                all_labels.append(patient_label)
                all_metadata.append({
                    'patient_id': patient_dir,
                    'metadata': patient_metadata,
                    'medical_metrics': result['medical_metrics']
                })
                
            except Exception as e:
                logger.warning(f"Failed to process patient {patient_dir}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid patient data found")
        
        # Convert to numpy arrays
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logger.info(f"Created dataset with {len(all_features)} patients, "
                   f"{features_array.shape[1]} features")
        
        return features_array, labels_array, all_metadata
    
    def save_medical_analysis_report(self, results: Dict[str, Any], 
                                   output_path: str):
        """
        Save a comprehensive medical analysis report.
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the report
        """
        report = {
            'patient_info': results.get('patient_metadata', {}),
            'analysis_summary': {
                'medical_condition': self.medical_condition,
                'total_frames': results['metadata']['n_frames'],
                'analysis_duration': results['metadata']['n_frames'] / results['metadata']['fps'],
                'fps': results['metadata']['fps']
            },
            'gait_events': results.get('events', {}),
            'medical_metrics': results.get('medical_metrics', {}),
            'clinical_interpretation': self._generate_clinical_interpretation(
                results.get('medical_metrics', {})
            ),
            'recommendations': self._generate_recommendations(
                results.get('medical_metrics', {})
            )
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Medical analysis report saved to {output_path}")
    
    def _generate_clinical_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate clinical interpretation of gait metrics."""
        interpretation = {}
        
        if self.medical_condition == "hydrocephalus":
            # Hydrocephalus-specific interpretations
            shuffling_pct = metrics.get('shuffling_percentage', 0)
            if shuffling_pct > 50:
                interpretation['shuffling'] = "Severe shuffling gait pattern observed"
            elif shuffling_pct > 20:
                interpretation['shuffling'] = "Moderate shuffling gait pattern"
            else:
                interpretation['shuffling'] = "Minimal shuffling observed"
            
            magnetic_pct = metrics.get('magnetic_gait_percentage', 0)
            if magnetic_pct > 40:
                interpretation['magnetic_gait'] = "Significant magnetic gait pattern"
            elif magnetic_pct > 15:
                interpretation['magnetic_gait'] = "Mild magnetic gait features"
            else:
                interpretation['magnetic_gait'] = "No significant magnetic gait"
            
            freezing_count = metrics.get('freezing_episode_count', 0)
            if freezing_count > 5:
                interpretation['freezing'] = "Frequent freezing episodes"
            elif freezing_count > 0:
                interpretation['freezing'] = "Occasional freezing episodes"
            else:
                interpretation['freezing'] = "No freezing episodes detected"
        
        return interpretation
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate clinical recommendations based on metrics."""
        recommendations = []
        
        if self.medical_condition == "hydrocephalus":
            shuffling_pct = metrics.get('shuffling_percentage', 0)
            magnetic_pct = metrics.get('magnetic_gait_percentage', 0)
            freezing_count = metrics.get('freezing_episode_count', 0)
            
            if shuffling_pct > 30 or magnetic_pct > 25:
                recommendations.append("Consider evaluation for NPH (Normal Pressure Hydrocephalus)")
                recommendations.append("Physical therapy consultation recommended")
            
            if freezing_count > 3:
                recommendations.append("Gait training and cueing strategies")
                recommendations.append("Fall prevention assessment")
            
            if metrics.get('step_height_cv', 0) > 0.5:
                recommendations.append("Balance training exercises")
                recommendations.append("Consider assistive devices")
        
        if not recommendations:
            recommendations.append("Continue regular monitoring")
            recommendations.append("Maintain current treatment plan")
        
        return recommendations
