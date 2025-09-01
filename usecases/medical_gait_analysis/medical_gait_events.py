"""
Medical Gait Events Module
==========================

Defines custom gait events and patterns specific to medical conditions.
Includes specialized event detection for Hydrocephalus and other neurological disorders.

Author: Medical Gait Analysis System
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class GaitPhase(Enum):
    """Standard gait phases for medical analysis."""
    HEEL_STRIKE = "heel_strike"
    LOADING_RESPONSE = "loading_response"
    MID_STANCE = "mid_stance"
    TERMINAL_STANCE = "terminal_stance"
    PRE_SWING = "pre_swing"
    INITIAL_SWING = "initial_swing"
    MID_SWING = "mid_swing"
    TERMINAL_SWING = "terminal_swing"

class MedicalGaitEvent(Enum):
    """Medical-specific gait events."""
    FOOT_FLAT = "foot_flat"
    TOE_OFF = "toe_off"
    HEEL_CONTACT = "heel_contact"
    DOUBLE_SUPPORT = "double_support"
    SINGLE_SUPPORT = "single_support"
    SWING_PHASE = "swing_phase"
    STANCE_PHASE = "stance_phase"
    
    # Pathological events
    SHUFFLING = "shuffling"
    FREEZING = "freezing"
    FESTINATION = "festination"
    ATAXIC_GAIT = "ataxic_gait"
    SPASTIC_GAIT = "spastic_gait"
    MAGNETIC_GAIT = "magnetic_gait"  # Hydrocephalus-specific

class MedicalGaitEvents(ABC):
    """
    Abstract base class for medical gait event detection.
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Initialize medical gait events detector.
        
        Args:
            fps: Frames per second of the video data
        """
        self.fps = fps
        self.frame_duration = 1.0 / fps
        
    @abstractmethod
    def detect_events(self, keypoints_sequence: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect gait events from keypoints sequence.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Dictionary mapping event names to frame indices
        """
        pass
    
    def calculate_temporal_parameters(self, events: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Calculate temporal gait parameters from detected events.
        
        Args:
            events: Dictionary of detected events
            
        Returns:
            Dictionary of temporal parameters
        """
        parameters = {}
        
        # Calculate stride time
        if 'heel_strike_right' in events and len(events['heel_strike_right']) > 1:
            heel_strikes = events['heel_strike_right']
            stride_times = [(heel_strikes[i+1] - heel_strikes[i]) * self.frame_duration 
                          for i in range(len(heel_strikes)-1)]
            parameters['stride_time_mean'] = np.mean(stride_times)
            parameters['stride_time_std'] = np.std(stride_times)
            parameters['stride_time_cv'] = parameters['stride_time_std'] / parameters['stride_time_mean']
        
        # Calculate step time
        if 'heel_strike_right' in events and 'heel_strike_left' in events:
            right_strikes = events['heel_strike_right']
            left_strikes = events['heel_strike_left']
            
            # Calculate step times (right to left and left to right)
            step_times = []
            for r_strike in right_strikes:
                # Find next left heel strike
                next_left = [l for l in left_strikes if l > r_strike]
                if next_left:
                    step_times.append((next_left[0] - r_strike) * self.frame_duration)
            
            for l_strike in left_strikes:
                # Find next right heel strike
                next_right = [r for r in right_strikes if r > l_strike]
                if next_right:
                    step_times.append((next_right[0] - l_strike) * self.frame_duration)
            
            if step_times:
                parameters['step_time_mean'] = np.mean(step_times)
                parameters['step_time_std'] = np.std(step_times)
                parameters['step_time_asymmetry'] = self._calculate_asymmetry(step_times)
        
        # Calculate stance and swing phase durations
        if 'toe_off_right' in events and 'heel_strike_right' in events:
            stance_durations = self._calculate_phase_durations(
                events['heel_strike_right'], events['toe_off_right']
            )
            swing_durations = self._calculate_phase_durations(
                events['toe_off_right'], events['heel_strike_right'], next_cycle=True
            )
            
            if stance_durations:
                parameters['stance_time_mean'] = np.mean(stance_durations) * self.frame_duration
                parameters['stance_time_std'] = np.std(stance_durations) * self.frame_duration
            
            if swing_durations:
                parameters['swing_time_mean'] = np.mean(swing_durations) * self.frame_duration
                parameters['swing_time_std'] = np.std(swing_durations) * self.frame_duration
        
        return parameters
    
    def _calculate_asymmetry(self, values: List[float]) -> float:
        """Calculate asymmetry index for bilateral measurements."""
        if len(values) < 2:
            return 0.0
        
        # Split into right and left values (assuming alternating pattern)
        right_values = values[::2]
        left_values = values[1::2]
        
        if not right_values or not left_values:
            return 0.0
        
        right_mean = np.mean(right_values)
        left_mean = np.mean(left_values)
        
        asymmetry = abs(right_mean - left_mean) / ((right_mean + left_mean) / 2) * 100
        return asymmetry
    
    def _calculate_phase_durations(self, start_events: List[int], end_events: List[int], 
                                 next_cycle: bool = False) -> List[int]:
        """Calculate durations between paired events."""
        durations = []
        
        for start in start_events:
            if next_cycle:
                # Find next event after start
                next_events = [e for e in end_events if e > start]
                if next_events:
                    durations.append(next_events[0] - start)
            else:
                # Find closest event before next start
                valid_ends = [e for e in end_events if e > start]
                if valid_ends:
                    durations.append(valid_ends[0] - start)
        
        return durations

class HydrocephalusGaitEvents(MedicalGaitEvents):
    """
    Specialized gait event detection for Hydrocephalus patients.
    
    Hydrocephalus gait characteristics:
    - Magnetic gait (feet appear stuck to ground)
    - Shuffling steps
    - Reduced step height
    - Wide-based gait
    - Difficulty initiating gait
    """
    
    def __init__(self, fps: float = 30.0):
        super().__init__(fps)
        
        # Hydrocephalus-specific thresholds
        self.min_step_height = 0.02  # Normalized units
        self.max_step_velocity = 0.1  # Normalized units per frame
        self.shuffling_threshold = 0.01  # Minimum foot clearance
        self.magnetic_gait_threshold = 0.005  # Very low foot lift
        
    def detect_events(self, keypoints_sequence: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect gait events specific to Hydrocephalus patterns.
        
        Args:
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Dictionary mapping event names to frame indices
        """
        events = {}
        n_frames = keypoints_sequence.shape[0]
        
        # Extract ankle and toe positions
        right_ankle = keypoints_sequence[:, 11, :2]  # RAnkle
        left_ankle = keypoints_sequence[:, 14, :2]   # LAnkle
        right_toe = keypoints_sequence[:, 22, :2]    # RBigToe
        left_toe = keypoints_sequence[:, 21, :2]     # LBigToe
        
        # Detect heel strikes and toe offs
        events['heel_strike_right'] = self._detect_heel_strikes(right_ankle)
        events['heel_strike_left'] = self._detect_heel_strikes(left_ankle)
        events['toe_off_right'] = self._detect_toe_offs(right_toe, right_ankle)
        events['toe_off_left'] = self._detect_toe_offs(left_toe, left_ankle)
        
        # Detect Hydrocephalus-specific events
        events['shuffling_episodes'] = self._detect_shuffling(right_ankle, left_ankle)
        events['magnetic_gait_episodes'] = self._detect_magnetic_gait(right_ankle, left_ankle)
        events['freezing_episodes'] = self._detect_freezing_episodes(right_ankle, left_ankle)
        events['wide_base_episodes'] = self._detect_wide_base_gait(right_ankle, left_ankle)
        
        return events
    
    def _detect_heel_strikes(self, ankle_positions: np.ndarray) -> List[int]:
        """Detect heel strike events from ankle trajectory."""
        heel_strikes = []
        
        if ankle_positions.shape[0] < 3:
            return heel_strikes
        
        # Calculate vertical velocity
        y_positions = ankle_positions[:, 1]
        y_velocity = np.gradient(y_positions)
        
        # Find local minima in vertical position with negative velocity
        for i in range(1, len(y_positions) - 1):
            if (y_positions[i] < y_positions[i-1] and 
                y_positions[i] < y_positions[i+1] and
                y_velocity[i] < 0):
                heel_strikes.append(i)
        
        return heel_strikes
    
    def _detect_toe_offs(self, toe_positions: np.ndarray, ankle_positions: np.ndarray) -> List[int]:
        """Detect toe-off events from toe and ankle trajectories."""
        toe_offs = []
        
        if toe_positions.shape[0] < 3 or ankle_positions.shape[0] < 3:
            return toe_offs
        
        # Calculate vertical velocity of toe
        toe_y = toe_positions[:, 1]
        toe_velocity = np.gradient(toe_y)
        
        # Find rapid upward movement of toe
        for i in range(1, len(toe_velocity) - 1):
            if (toe_velocity[i] > 0.01 and  # Significant upward velocity
                toe_velocity[i] > toe_velocity[i-1] and
                toe_velocity[i] > toe_velocity[i+1]):
                toe_offs.append(i)
        
        return toe_offs
    
    def _detect_shuffling(self, right_ankle: np.ndarray, left_ankle: np.ndarray) -> List[Tuple[int, int]]:
        """Detect shuffling episodes (low foot clearance)."""
        shuffling_episodes = []
        
        # Calculate foot clearance (minimum height during swing)
        for ankle in [right_ankle, left_ankle]:
            y_positions = ankle[:, 1]
            
            # Find periods of consistently low foot clearance
            low_clearance_frames = []
            for i in range(len(y_positions)):
                if y_positions[i] < self.shuffling_threshold:
                    low_clearance_frames.append(i)
            
            # Group consecutive frames into episodes
            if low_clearance_frames:
                episodes = self._group_consecutive_frames(low_clearance_frames, min_duration=5)
                shuffling_episodes.extend(episodes)
        
        return shuffling_episodes
    
    def _detect_magnetic_gait(self, right_ankle: np.ndarray, left_ankle: np.ndarray) -> List[Tuple[int, int]]:
        """Detect magnetic gait episodes (extremely low foot lift)."""
        magnetic_episodes = []
        
        for ankle in [right_ankle, left_ankle]:
            y_positions = ankle[:, 1]
            
            # Find periods where foot barely lifts off ground
            magnetic_frames = []
            for i in range(len(y_positions)):
                if y_positions[i] < self.magnetic_gait_threshold:
                    magnetic_frames.append(i)
            
            # Group consecutive frames into episodes
            if magnetic_frames:
                episodes = self._group_consecutive_frames(magnetic_frames, min_duration=10)
                magnetic_episodes.extend(episodes)
        
        return magnetic_episodes
    
    def _detect_freezing_episodes(self, right_ankle: np.ndarray, left_ankle: np.ndarray) -> List[Tuple[int, int]]:
        """Detect freezing episodes (sudden stops in gait)."""
        freezing_episodes = []
        
        # Calculate combined ankle velocity
        right_velocity = np.linalg.norm(np.gradient(right_ankle, axis=0), axis=1)
        left_velocity = np.linalg.norm(np.gradient(left_ankle, axis=0), axis=1)
        combined_velocity = (right_velocity + left_velocity) / 2
        
        # Find periods of very low velocity
        freezing_threshold = 0.001
        freezing_frames = []
        
        for i in range(len(combined_velocity)):
            if combined_velocity[i] < freezing_threshold:
                freezing_frames.append(i)
        
        # Group consecutive frames into episodes
        if freezing_frames:
            episodes = self._group_consecutive_frames(freezing_frames, min_duration=15)
            freezing_episodes.extend(episodes)
        
        return freezing_episodes
    
    def _detect_wide_base_gait(self, right_ankle: np.ndarray, left_ankle: np.ndarray) -> List[Tuple[int, int]]:
        """Detect wide-base gait episodes."""
        wide_base_episodes = []
        
        # Calculate step width (distance between ankles)
        step_widths = []
        for i in range(len(right_ankle)):
            if not (np.any(np.isnan(right_ankle[i])) or np.any(np.isnan(left_ankle[i]))):
                width = abs(right_ankle[i, 0] - left_ankle[i, 0])
                step_widths.append(width)
            else:
                step_widths.append(0)
        
        step_widths = np.array(step_widths)
        
        # Define wide base threshold (e.g., 1.5 times normal step width)
        normal_width = np.median(step_widths[step_widths > 0])
        wide_threshold = normal_width * 1.5
        
        # Find frames with wide base
        wide_frames = []
        for i in range(len(step_widths)):
            if step_widths[i] > wide_threshold:
                wide_frames.append(i)
        
        # Group consecutive frames into episodes
        if wide_frames:
            episodes = self._group_consecutive_frames(wide_frames, min_duration=8)
            wide_base_episodes.extend(episodes)
        
        return wide_base_episodes
    
    def _group_consecutive_frames(self, frames: List[int], min_duration: int = 5) -> List[Tuple[int, int]]:
        """Group consecutive frame indices into episodes."""
        if not frames:
            return []
        
        episodes = []
        start_frame = frames[0]
        end_frame = frames[0]
        
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1:
                # Consecutive frame
                end_frame = frames[i]
            else:
                # Gap found, save episode if long enough
                if end_frame - start_frame + 1 >= min_duration:
                    episodes.append((start_frame, end_frame))
                start_frame = frames[i]
                end_frame = frames[i]
        
        # Don't forget the last episode
        if end_frame - start_frame + 1 >= min_duration:
            episodes.append((start_frame, end_frame))
        
        return episodes
    
    def calculate_hydrocephalus_metrics(self, events: Dict[str, List[int]], 
                                      keypoints_sequence: np.ndarray) -> Dict[str, float]:
        """
        Calculate Hydrocephalus-specific gait metrics.
        
        Args:
            events: Dictionary of detected events
            keypoints_sequence: Array of shape (n_frames, 25, 3)
            
        Returns:
            Dictionary of Hydrocephalus-specific metrics
        """
        metrics = {}
        
        # Calculate basic temporal parameters
        temporal_params = self.calculate_temporal_parameters(events)
        metrics.update(temporal_params)
        
        # Shuffling severity
        if 'shuffling_episodes' in events:
            total_frames = keypoints_sequence.shape[0]
            shuffling_frames = sum([end - start + 1 for start, end in events['shuffling_episodes']])
            metrics['shuffling_percentage'] = (shuffling_frames / total_frames) * 100
            metrics['shuffling_episode_count'] = len(events['shuffling_episodes'])
        
        # Magnetic gait severity
        if 'magnetic_gait_episodes' in events:
            total_frames = keypoints_sequence.shape[0]
            magnetic_frames = sum([end - start + 1 for start, end in events['magnetic_gait_episodes']])
            metrics['magnetic_gait_percentage'] = (magnetic_frames / total_frames) * 100
            metrics['magnetic_gait_episode_count'] = len(events['magnetic_gait_episodes'])
        
        # Freezing episodes
        if 'freezing_episodes' in events:
            metrics['freezing_episode_count'] = len(events['freezing_episodes'])
            if events['freezing_episodes']:
                freezing_durations = [(end - start + 1) * self.frame_duration 
                                    for start, end in events['freezing_episodes']]
                metrics['mean_freezing_duration'] = np.mean(freezing_durations)
                metrics['max_freezing_duration'] = np.max(freezing_durations)
        
        # Wide base gait
        if 'wide_base_episodes' in events:
            total_frames = keypoints_sequence.shape[0]
            wide_base_frames = sum([end - start + 1 for start, end in events['wide_base_episodes']])
            metrics['wide_base_percentage'] = (wide_base_frames / total_frames) * 100
        
        # Calculate step height variability
        right_ankle = keypoints_sequence[:, 11, 1]  # RAnkle Y
        left_ankle = keypoints_sequence[:, 14, 1]   # LAnkle Y
        
        # Find swing phases and calculate step heights
        step_heights = []
        for ankle_y in [right_ankle, left_ankle]:
            # Simple step height calculation (max - min during swing)
            if len(ankle_y) > 10:
                for i in range(5, len(ankle_y) - 5):
                    local_min = np.min(ankle_y[i-5:i+5])
                    local_max = np.max(ankle_y[i-5:i+5])
                    step_heights.append(local_max - local_min)
        
        if step_heights:
            metrics['step_height_mean'] = np.mean(step_heights)
            metrics['step_height_std'] = np.std(step_heights)
            metrics['step_height_cv'] = metrics['step_height_std'] / metrics['step_height_mean']
        
        return metrics
