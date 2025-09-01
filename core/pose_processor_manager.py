"""
Pose Processor Manager
====================

This module provides a unified interface for different pose estimation models,
allowing easy switching between different pose estimation backends.

Author: Gait Analysis System
"""

import os
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseProcessor(ABC):
    """
    Abstract base class for pose processors.
    All pose processors must implement this interface.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the pose processor."""
        pass
    
    @abstractmethod
    def process_video(self, video_path: str) -> bool:
        """Process video file and extract pose landmarks."""
        pass
    
    @abstractmethod
    def process_webcam(self, duration: float = 10.0) -> bool:
        """Process webcam feed for real-time pose estimation."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass

class PoseProcessorManager:
    """
    Manager class for creating pose processors.
    Supports multiple pose estimation backends with unified interface.
    """
    
    AVAILABLE_MODELS = {
        'mediapipe': 'MediaPipe Pose'
    }
    
    @staticmethod
    def create_processor(model_type: str = 'mediapipe', **kwargs) -> PoseProcessor:
        """
        Create a pose processor instance.
        
        Args:
            model_type: Type of pose model ('mediapipe' or other supported models)
            **kwargs: Additional arguments for the processor
            
        Returns:
            PoseProcessor instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_type = model_type.lower()
        
        if model_type == 'mediapipe':
            from .mediapipe_integration import MediaPipeProcessor
            return MediaPipeProcessor(**kwargs)
        
        else:
            available = ', '.join(PoseProcessorManager.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available: {available}")
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get list of available pose models.
        
        Returns:
            Dictionary mapping model keys to display names
        """
        return PoseProcessorManager.AVAILABLE_MODELS.copy()
    
    @staticmethod
    def validate_model_type(model_type: str) -> bool:
        """
        Validate if a model type is supported.
        
        Args:
            model_type: Model type to validate
            
        Returns:
            True if supported, False otherwise
        """
        return model_type.lower() in PoseProcessorManager.AVAILABLE_MODELS
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_type: Type of pose model
            
        Returns:
            Dictionary with model information
        """
        model_type = model_type.lower()
        
        if model_type == 'mediapipe':
            return {
                'name': 'MediaPipe Pose',
                'description': 'Lightweight, real-time pose estimation by Google',
                'landmarks': 33,
                'keypoints': 25,  # After conversion to BODY_25
                'advantages': [
                    'Fast and lightweight',
                    'Good for real-time applications',
                    'Easy to use and integrate',
                    'Works well on CPU'
                ],
                'disadvantages': [
                    'Lower accuracy compared to deep learning models',
                    'Limited to 2D pose estimation'
                ],
                'best_for': [
                    'Real-time applications',
                    'Mobile/edge devices',
                    'Quick prototyping'
                ]
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class UnifiedPoseProcessor:
    """
    Unified pose processor that can switch between different models.
    Provides a consistent interface regardless of the underlying model.
    """
    
    def __init__(self, 
                 model_type: str = 'mediapipe',
                 output_dir: str = 'outputs/mediapipe',
                 **kwargs):
        """
        Initialize the unified pose processor.
        
        Args:
            model_type: Type of pose model ('mediapipe' or other supported models)
            output_dir: Base output directory
            **kwargs: Additional arguments for the specific processor
        """
        self.model_type = model_type.lower()
        self.output_dir = output_dir
        
        # Validate model type
        if not PoseProcessorManager.validate_model_type(self.model_type):
            available = ', '.join(PoseProcessorManager.get_available_models().keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available: {available}")
        
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, self.model_type)
        
        # Create processor
        self.processor = PoseProcessorManager.create_processor(
            model_type=self.model_type,
            output_dir=model_output_dir,
            **kwargs
        )
        
        logger.info(f"Initialized {self.model_type} pose processor")
        logger.info(f"Output directory: {model_output_dir}")
    
    def process_video(self, video_path: str) -> bool:
        """
        Process video file and extract pose landmarks.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing video with {self.model_type}: {video_path}")
        return self.processor.process_video(video_path)
    
    def process_webcam(self, duration: float = 10.0) -> bool:
        """
        Process webcam feed for real-time pose estimation.
        
        Args:
            duration: Duration to record in seconds
            
        Returns:
            True if processing successful, False otherwise
        """
        logger.info(f"Processing webcam with {self.model_type} for {duration} seconds")
        return self.processor.process_webcam(duration)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return PoseProcessorManager.get_model_info(self.model_type)
    
    def switch_model(self, new_model_type: str, **kwargs):
        """
        Switch to a different pose model.
        
        Args:
            new_model_type: New model type to switch to
            **kwargs: Additional arguments for the new processor
        """
        logger.info(f"Switching from {self.model_type} to {new_model_type}")
        
        # Cleanup current processor
        if hasattr(self, 'processor'):
            self.processor.cleanup()
        
        # Update model type
        self.model_type = new_model_type.lower()
        
        # Create new model-specific output directory
        model_output_dir = os.path.join(self.output_dir, self.model_type)
        
        # Create new processor
        self.processor = PoseProcessorManager.create_processor(
            model_type=self.model_type,
            output_dir=model_output_dir,
            **kwargs
        )
        
        logger.info(f"Successfully switched to {self.model_type} pose processor")
    
    def compare_models(self, video_path: str, models: list = None) -> Dict[str, Any]:
        """
        Compare different pose models on the same video.
        
        Args:
            video_path: Path to video file for comparison
            models: List of model types to compare (default: all available)
            
        Returns:
            Dictionary with comparison results
        """
        if models is None:
            models = list(PoseProcessorManager.get_available_models().keys())
        
        results = {}
        
        for model_type in models:
            if not PoseProcessorManager.validate_model_type(model_type):
                logger.warning(f"Skipping invalid model type: {model_type}")
                continue
            
            logger.info(f"Testing {model_type} on {video_path}")
            
            try:
                # Create temporary processor for comparison
                temp_processor = PoseProcessorManager.create_processor(
                    model_type=model_type,
                    output_dir=os.path.join(self.output_dir, f"comparison_{model_type}")
                )
                
                # Process video and measure time
                import time
                start_time = time.time()
                success = temp_processor.process_video(video_path)
                end_time = time.time()
                
                results[model_type] = {
                    'success': success,
                    'processing_time': end_time - start_time,
                    'model_info': PoseProcessorManager.get_model_info(model_type)
                }
                
                # Cleanup
                temp_processor.cleanup()
                
            except Exception as e:
                logger.error(f"Error testing {model_type}: {e}")
                results[model_type] = {
                    'success': False,
                    'error': str(e),
                    'model_info': PoseProcessorManager.get_model_info(model_type)
                }
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()
        logger.info(f"Cleaned up {self.model_type} pose processor")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
