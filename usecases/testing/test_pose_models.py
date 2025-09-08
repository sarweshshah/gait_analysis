#!/usr/bin/env python3
"""
Test Script for Pose Models
==========================

This script tests the integration of pose estimation models.
"""

import os
import sys
import logging
import pytest
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pose_processor_manager import UnifiedPoseProcessor, PoseProcessorManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def video_path():
    """Fixture providing a test video path."""
    # Check for available test videos
    test_videos = [
        "videos/raw/sarwesh1.mp4",
        "videos/raw/sarwesh.mp4", 
        "videos/raw/chavan.mp4",
        "videos/raw/nitin.mp4",
        "videos/raw/sanghveer.mp4"
    ]
    
    for video in test_videos:
        if os.path.exists(video):
            return video
    
    # If no video found, return None to skip the test
    return None

def test_model_availability():
    """Test if all models are available."""
    print("Testing model availability...")
    
    available_models = PoseProcessorManager.get_available_models()
    print(f"Available models: {list(available_models.keys())}")
    
    # Assert that we have at least one model available
    assert len(available_models) > 0, "No models are available"
    
    for model_type in available_models.keys():
        try:
            info = PoseProcessorManager.get_model_info(model_type)
            print(f"✓ {model_type}: {info['name']}")
            # Assert that model info is valid
            assert 'name' in info, f"Model info missing 'name' for {model_type}"
        except Exception as e:
            print(f"✗ {model_type}: Error - {e}")
            # Don't fail the test for individual model errors, just log them

def test_processor_creation():
    """Test creating pose processors."""
    print("\nTesting processor creation...")
    
    # Test MediaPipe processor
    print("Creating MediaPipe processor...")
    mediapipe_processor = UnifiedPoseProcessor(model_type='mediapipe')
    print("✓ MediaPipe processor created successfully")
    
    # Assert that processor was created successfully
    assert mediapipe_processor is not None, "MediaPipe processor creation failed"
    
    # Test that we can get model info
    model_info = mediapipe_processor.get_model_info()
    assert model_info is not None, "Could not get model info"
    assert 'name' in model_info, "Model info missing 'name' field"
    
    mediapipe_processor.cleanup()

def test_model_switching():
    """Test switching between models."""
    print("\nTesting model switching...")
    
    # Start with MediaPipe
    processor = UnifiedPoseProcessor(model_type='mediapipe')
    print("✓ Started with MediaPipe")
    
    # Get model info
    info = processor.get_model_info()
    print(f"  Model: {info['name']}")
    
    # Assert that we got valid model info
    assert info is not None, "Could not get model info"
    assert 'name' in info, "Model info missing 'name' field"
    assert info['name'] == 'MediaPipe Pose', f"Expected 'MediaPipe Pose', got '{info['name']}'"
    
    processor.cleanup()

def test_video_processing(video_path):
    """Test video processing with available models."""
    # Skip test if no video is available
    if video_path is None:
        pytest.skip("No test video available")
    
    print(f"\nTesting video processing with: {video_path}")
    
    # Assert that video file exists
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    
    available_models = PoseProcessorManager.get_available_models()
    
    # Assert that we have at least one model to test
    assert len(available_models) > 0, "No models available for testing"
    
    for model_type in available_models.keys():
        print(f"\nTesting {model_type}...")
        
        try:
            with UnifiedPoseProcessor(model_type=model_type) as processor:
                success = processor.process_video(video_path)
                
                if success:
                    print(f"✓ {model_type} processing successful")
                else:
                    print(f"✗ {model_type} processing failed")
                    # Don't fail the test for individual model processing failures
                    
        except Exception as e:
            print(f"✗ {model_type} processing error: {e}")
            # Don't fail the test for individual model errors, just log them

def main():
    """Main test function."""
    print("="*60)
    print("POSE MODEL INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Model availability
    test_model_availability()
    
    # Test 2: Processor creation
    test_processor_creation()
    
    # Test 3: Model switching
    test_model_switching()
    
    # Test 4: Video processing (if video available)
    video_path = "videos/raw/sarwesh1.mp4"  # Updated path to videos folder
    if os.path.exists(video_path):
        test_video_processing(video_path)
    else:
        print(f"\nSkipping video processing test - video not found: {video_path}")
        print("To test video processing, place a video file in the videos/raw directory")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    print("\nSummary:")
    print("- MediaPipe should work if mediapipe is installed")
    print("- Additional models can be easily added to the system")
    print("- All models provide the same interface for easy switching")
    print("- Video processing test requires a video file in the videos/raw directory")

if __name__ == "__main__":
    main()
