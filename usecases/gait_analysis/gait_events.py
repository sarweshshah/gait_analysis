"""
Basic Gait Events Detection Module
=================================

This module provides rule-based detection of fundamental biomechanical gait events
using MediaPipe pose keypoints. Unlike the medical gait events module, this focuses
on basic gait cycle events that are common to all walking patterns.

Author: Gait Analysis System
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import euclidean
from core.gait_data_preprocessing import GaitDataPreprocessor
from core.utils.constants import (
    LEFT, RIGHT,
    HEEL_STRIKE, TOE_OFF,
)

logger = logging.getLogger(__name__)

class BasicGaitEvents:
    """
    Detects fundamental gait events from MediaPipe pose keypoints.
    
    Events detected:
    - Heel Strike (Initial Contact): When foot first touches ground
    - Toe Off (Terminal Contact): When foot leaves ground
    - Mid Stance: Peak of stance phase
    - Mid Swing: Peak of swing phase
    """
    
    # Default indices (MediaPipe) will be set in __init__ based on format
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    
    def __init__(self, fps: float = 30.0, confidence_threshold: float = 0.3, keypoint_format: str = 'mediapipe'):
        """
        Initialize gait events detector.
        
        Args:
            fps: Video frame rate for timing calculations
            confidence_threshold: Minimum confidence for keypoint detection
            keypoint_format: 'mediapipe' (33 landmarks) or 'body25' (OpenPose BODY_25 mapping)
        """
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.frame_time = 1.0 / fps
        self.keypoint_format = keypoint_format

        # Configure landmark indices based on the keypoint format
        if keypoint_format == 'mediapipe':
            # MediaPipe pose landmark indices (33 landmarks)
            self.LEFT_ANKLE = 27
            self.RIGHT_ANKLE = 28
            self.LEFT_HEEL = 29
            self.RIGHT_HEEL = 30
            self.LEFT_FOOT_INDEX = 31
            self.RIGHT_FOOT_INDEX = 32
            self.LEFT_HIP = 23
            self.RIGHT_HIP = 24
            self.LEFT_KNEE = 25
            self.RIGHT_KNEE = 26
        elif keypoint_format == 'body25':
            # BODY_25 indices as produced by our preprocessing (`core/gait_data_preprocessing.py`)
            # RHip=9, RKnee=10, RAnkle=11, LHip=12, LKnee=13, LAnkle=14
            # LBigToe=19, LSmallToe=20, LHeel=21, RBigToe=22, RSmallToe=23, RHeel=24
            self.LEFT_ANKLE = 14
            self.RIGHT_ANKLE = 11
            self.LEFT_HEEL = 21
            self.RIGHT_HEEL = 24
            self.LEFT_FOOT_INDEX = 19  # LBigToe
            self.RIGHT_FOOT_INDEX = 22  # RBigToe
            self.LEFT_HIP = 12
            self.RIGHT_HIP = 9
            self.LEFT_KNEE = 13
            self.RIGHT_KNEE = 10
        else:
            raise ValueError(f"Unsupported keypoint_format: {keypoint_format}")
        
    def detect_events(self, keypoints_sequence: np.ndarray, 
                     confidences: Optional[np.ndarray] = None) -> Dict[str, List]:
        """
        Detect gait events from pose keypoint sequence.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, n_keypoints, 2) with x,y coordinates
            confidences: Optional confidence scores for keypoints
            
        Returns:
            Dictionary containing detected events with timestamps
        """
        if keypoints_sequence.shape[0] < 10:
            logger.warning("Sequence too short for reliable event detection")
            return self._empty_events()
        
        # Extract foot and leg keypoints
        left_foot_data = self._extract_foot_data(keypoints_sequence, 'left')
        right_foot_data = self._extract_foot_data(keypoints_sequence, 'right')
        
        # Detect events for each foot
        left_events = self._detect_foot_events(left_foot_data, 'left')
        right_events = self._detect_foot_events(right_foot_data, 'right')
        
        # Combine and validate events
        all_events = self._combine_bilateral_events(left_events, right_events)
        
        # Add timing information
        all_events = self._add_timing_info(all_events)

        logger.info(
            f"Detected {len(all_events['heel_strikes'])} heel strikes, "
            f"{len(all_events['toe_offs'])} toe offs"
        )
        
        return all_events
    
    def _extract_foot_data(self, keypoints: np.ndarray, side: str) -> Dict[str, np.ndarray]:
        """Extract relevant keypoints for one foot."""
        if side == LEFT:
            ankle_idx = self.LEFT_ANKLE
            heel_idx = self.LEFT_HEEL
            toe_idx = self.LEFT_FOOT_INDEX
            hip_idx = self.LEFT_HIP
            knee_idx = self.LEFT_KNEE
        else:
            ankle_idx = self.RIGHT_ANKLE
            heel_idx = self.RIGHT_HEEL
            toe_idx = self.RIGHT_FOOT_INDEX
            hip_idx = self.RIGHT_HIP
            knee_idx = self.RIGHT_KNEE
        
        return {
            'ankle': keypoints[:, ankle_idx, :],
            'heel': keypoints[:, heel_idx, :],
            'toe': keypoints[:, toe_idx, :],
            'hip': keypoints[:, hip_idx, :],
            'knee': keypoints[:, knee_idx, :]
        }
    
    def _detect_foot_events(self, foot_data: Dict[str, np.ndarray], side: str) -> Dict[str, List]:
        """Detect gait events for one foot."""
        ankle_y = foot_data['ankle'][:, 1]  # Y-coordinate (vertical)
        heel_y = foot_data['heel'][:, 1]
        
        # Smooth the signals
        if len(ankle_y) > 5:
            ankle_y_smooth = savgol_filter(ankle_y, window_length=5, polyorder=2)
            heel_y_smooth = savgol_filter(heel_y, window_length=5, polyorder=2)
        else:
            ankle_y_smooth = ankle_y
            heel_y_smooth = heel_y
        
        # Calculate foot velocity (vertical)
        foot_velocity = np.gradient(ankle_y_smooth)
        
        # Detect heel strikes (local minima in ankle height + low velocity)
        heel_strikes = self._detect_heel_strikes(ankle_y_smooth, foot_velocity)
        
        # Detect toe offs (local minima in heel height + increasing velocity)
        toe_offs = self._detect_toe_offs(heel_y_smooth, foot_velocity)
        
        # Detect stance and swing phases
        stance_phases, swing_phases = self._detect_phases(heel_strikes, toe_offs, len(ankle_y))
        
        return {
            'heel_strikes': heel_strikes,
            'toe_offs': toe_offs,
            'stance_phases': stance_phases,
            'swing_phases': swing_phases,
            'side': side
        }
    
    def _detect_heel_strikes(self, ankle_y: np.ndarray, velocity: np.ndarray) -> List[int]:
        """Detect heel strike events."""
        # Find local minima in ankle height
        minima, _ = find_peaks(-ankle_y, height=-np.percentile(ankle_y, 25), distance=int(0.3 * self.fps))
        
        # Filter by velocity (should be near zero or negative at heel strike)
        heel_strikes = []
        for minimum in minima:
            if minimum < len(velocity) and velocity[minimum] <= np.percentile(velocity, 30):
                heel_strikes.append(minimum)
        
        return heel_strikes
    
    def _detect_toe_offs(self, heel_y: np.ndarray, velocity: np.ndarray) -> List[int]:
        """Detect toe off events."""
        # Find points where heel starts rising (positive velocity)
        rising_points = np.where(np.diff(heel_y) > 0)[0]
        
        # Find local minima in heel height that precede rising motion
        minima, _ = find_peaks(-heel_y, height=-np.percentile(heel_y, 30), distance=int(0.3 * self.fps))
        
        toe_offs = []
        for minimum in minima:
            # Check if this minimum is followed by rising motion
            next_rising = rising_points[rising_points > minimum]
            if len(next_rising) > 0 and next_rising[0] - minimum < int(0.1 * self.fps):
                toe_offs.append(minimum)
        
        return toe_offs
    
    def _detect_phases(self, heel_strikes: List[int], toe_offs: List[int], 
                      sequence_length: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Detect stance and swing phases."""
        stance_phases = []
        swing_phases = []
        
        # Sort all events
        all_events = [(hs, HEEL_STRIKE) for hs in heel_strikes] + [(to, TOE_OFF) for to in toe_offs]
        all_events.sort(key=lambda x: x[0])
        
        # Identify phases
        current_phase = None
        phase_start = 0
        
        for frame, event_type in all_events:
            if event_type == HEEL_STRIKE:
                if current_phase == 'swing':
                    swing_phases.append((phase_start, frame))
                current_phase = 'stance'
                phase_start = frame
            elif event_type == TOE_OFF:
                if current_phase == 'stance':
                    stance_phases.append((phase_start, frame))
                current_phase = 'swing'
                phase_start = frame
        
        # Handle final phase
        if current_phase == 'stance':
            stance_phases.append((phase_start, sequence_length - 1))
        elif current_phase == 'swing':
            swing_phases.append((phase_start, sequence_length - 1))
        
        return stance_phases, swing_phases
    
    def _combine_bilateral_events(self, left_events: Dict, right_events: Dict) -> Dict[str, List]:
        """Combine events from both feet."""
        combined = {
            'heel_strikes': [],
            'toe_offs': [],
            'stance_phases': [],
            'swing_phases': [],
            'double_support_phases': [],
            'single_support_phases': []
        }
        
        # Add left foot events
        for hs in left_events['heel_strikes']:
            combined['heel_strikes'].append({'frame': hs, 'side': LEFT})
        for to in left_events['toe_offs']:
            combined['toe_offs'].append({'frame': to, 'side': LEFT})
        for stance in left_events['stance_phases']:
            combined['stance_phases'].append({'start': stance[0], 'end': stance[1], 'side': LEFT})
        for swing in left_events['swing_phases']:
            combined['swing_phases'].append({'start': swing[0], 'end': swing[1], 'side': LEFT})
        
        # Add right foot events
        for hs in right_events['heel_strikes']:
            combined['heel_strikes'].append({'frame': hs, 'side': RIGHT})
        for to in right_events['toe_offs']:
            combined['toe_offs'].append({'frame': to, 'side': RIGHT})
        for stance in right_events['stance_phases']:
            combined['stance_phases'].append({'start': stance[0], 'end': stance[1], 'side': RIGHT})
        for swing in right_events['swing_phases']:
            combined['swing_phases'].append({'start': swing[0], 'end': swing[1], 'side': RIGHT})
        
        # Sort events by frame
        combined['heel_strikes'].sort(key=lambda x: x['frame'])
        combined['toe_offs'].sort(key=lambda x: x['frame'])
        
        # Detect double support and single support phases
        combined = self._detect_support_phases(combined)
        
        return combined
    
    def _detect_support_phases(self, events: Dict) -> Dict:
        """Detect double and single support phases."""
        # This is a simplified implementation
        # In reality, you'd need more sophisticated logic to determine
        # when both feet are on ground (double support) vs one foot (single support)
        
        double_support = []
        single_support = []
        
        # Basic heuristic: double support occurs briefly after heel strike
        # and before toe off of the opposite foot
        for hs in events['heel_strikes']:
            # Look for opposite foot toe off within reasonable time window
            opposite_side = RIGHT if hs['side'] == LEFT else LEFT
            nearby_toe_offs = [to for to in events['toe_offs'] 
                              if to['side'] == opposite_side and 
                              abs(to['frame'] - hs['frame']) < int(0.2 * self.fps)]
            
            if nearby_toe_offs:
                # Double support phase
                start_frame = min(hs['frame'], nearby_toe_offs[0]['frame'])
                end_frame = max(hs['frame'], nearby_toe_offs[0]['frame'])
                double_support.append({'start': start_frame, 'end': end_frame})
        
        events['double_support_phases'] = double_support
        events['single_support_phases'] = single_support  # Would need more complex logic
        
        return events
    
    def _add_timing_info(self, events: Dict) -> Dict:
        """Add timing information to events."""
        for event_list in ['heel_strikes', 'toe_offs']:
            for event in events[event_list]:
                event['time'] = event['frame'] * self.frame_time
        
        for phase_list in ['stance_phases', 'swing_phases', 'double_support_phases']:
            for phase in events[phase_list]:
                phase['start_time'] = phase['start'] * self.frame_time
                phase['end_time'] = phase['end'] * self.frame_time
                phase['duration'] = phase['end_time'] - phase['start_time']
        
        return events
    
    def _empty_events(self) -> Dict[str, List]:
        """Return empty events structure."""
        return {
            'heel_strikes': [],
            'toe_offs': [],
            'stance_phases': [],
            'swing_phases': [],
            'double_support_phases': [],
            'single_support_phases': []
        }
    
    def calculate_gait_metrics(self, events: Dict) -> Dict[str, float]:
        """Calculate basic gait metrics from detected events."""
        metrics = {}
        
        # Calculate stride times (heel strike to heel strike, same foot)
        left_heel_strikes = [e for e in events['heel_strikes'] if e['side'] == LEFT]
        right_heel_strikes = [e for e in events['heel_strikes'] if e['side'] == RIGHT]
        
        if len(left_heel_strikes) > 1:
            left_stride_times = [left_heel_strikes[i+1]['time'] - left_heel_strikes[i]['time'] 
                               for i in range(len(left_heel_strikes)-1)]
            metrics['left_stride_time_mean'] = np.mean(left_stride_times)
            metrics['left_stride_time_std'] = np.std(left_stride_times)
        
        if len(right_heel_strikes) > 1:
            right_stride_times = [right_heel_strikes[i+1]['time'] - right_heel_strikes[i]['time'] 
                                for i in range(len(right_heel_strikes)-1)]
            metrics['right_stride_time_mean'] = np.mean(right_stride_times)
            metrics['right_stride_time_std'] = np.std(right_stride_times)
        
        # Calculate stance and swing times
        left_stance = [p for p in events['stance_phases'] if p['side'] == LEFT]
        right_stance = [p for p in events['stance_phases'] if p['side'] == RIGHT]
        
        if left_stance:
            metrics['left_stance_time_mean'] = np.mean([p['duration'] for p in left_stance])
        if right_stance:
            metrics['right_stance_time_mean'] = np.mean([p['duration'] for p in right_stance])
        
        # Calculate cadence (steps per minute)
        total_time = max([e['time'] for e in events['heel_strikes']], default=0)
        if total_time > 0:
            total_steps = len(events['heel_strikes'])
            metrics['cadence'] = (total_steps / total_time) * 60
        
        return metrics
