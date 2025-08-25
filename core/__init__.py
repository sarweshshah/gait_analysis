"""
Core Gait Analysis Package
=========================

This package contains the core logic and shared components for gait analysis.
"""

from .mediapipe_integration import MediaPipeProcessor
from .gait_data_preprocessing import GaitDataPreprocessor
from .tcn_gait_model import create_gait_tcn_model, compile_gait_model
from .gait_training import GaitTrainer, GaitMetrics

__all__ = [
    'MediaPipeProcessor',
    'GaitDataPreprocessor',
    'create_gait_tcn_model',
    'compile_gait_model',
    'GaitTrainer',
    'GaitMetrics'
]

__version__ = "1.0.0"
