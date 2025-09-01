"""
Medical Gait Analysis Module
============================

This module provides specialized gait analysis for medical conditions detection,
including Hydrocephalus, Parkinson's disease, and other neurological disorders.

Features:
- Custom medical gait event definitions
- Specialized classifiers for medical conditions
- Training pipeline for medical datasets
- Clinical-grade analysis and reporting

Author: Medical Gait Analysis System
"""

from .medical_gait_events import MedicalGaitEvents, HydrocephalusGaitEvents
from .medical_gait_classifier import MedicalGaitClassifier, HydrocephalusClassifier
from .medical_data_processor import MedicalGaitDataProcessor
from .medical_training_pipeline import MedicalGaitTrainingPipeline

__version__ = "1.0.0"
__all__ = [
    "MedicalGaitEvents",
    "HydrocephalusGaitEvents", 
    "MedicalGaitClassifier",
    "HydrocephalusClassifier",
    "MedicalGaitDataProcessor",
    "MedicalGaitTrainingPipeline"
]
