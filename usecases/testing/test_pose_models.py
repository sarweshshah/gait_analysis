#!/usr/bin/env python3
"""
Test Script for Pose Models
==========================

This script tests the integration of both MediaPipe and MeTRAbs pose estimation models.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pose_processor_manager import UnifiedPoseProcessor, PoseProcessorManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_availability():
    """Test if all models are available."""
    print("Testing model availability...")
    
    available_models = PoseProcessorManager.get_available_models()
    print(f"Available models: {list(available_models.keys())}")
    
    for model_type in available_models.keys():
        try:
            info = PoseProcessorManager.get_model_info(model_type)
            print(f"✓ {model_type}: {info['name']}")
        except Exception as e:
            print(f"✗ {model_type}: Error - {e}")
    
    return True

def test_processor_creation():
    """Test creating pose processors."""
    print("\nTesting processor creation...")
    
    try:
        # Test MediaPipe processor
        print("Creating MediaPipe processor...")
        mediapipe_processor = UnifiedPoseProcessor(model_type='mediapipe')
        print("✓ MediaPipe processor created successfully")
        mediapipe_processor.cleanup()
        
        # Test MeTRAbs processor (may fail if not installed)
        print("Creating MeTRAbs processor...")
        try:
            metrabs_processor = UnifiedPoseProcessor(model_type='metrabs')
            print("✓ MeTRAbs processor created successfully")
            metrabs_processor.cleanup()
        except Exception as e:
            print(f"⚠ MeTRAbs processor creation failed (expected if not installed): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Processor creation failed: {e}")
        return False

def test_model_switching():
    """Test switching between models."""
    print("\nTesting model switching...")
    
    try:
        # Start with MediaPipe
        processor = UnifiedPoseProcessor(model_type='mediapipe')
        print("✓ Started with MediaPipe")
        
        # Get model info
        info = processor.get_model_info()
        print(f"  Model: {info['name']}")
        
        # Try to switch to MeTRAbs
        try:
            processor.switch_model('metrabs')
            print("✓ Switched to MeTRAbs")
            
            info = processor.get_model_info()
            print(f"  Model: {info['name']}")
            
        except Exception as e:
            print(f"⚠ MeTRAbs switching failed (expected if not installed): {e}")
        
        processor.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Model switching failed: {e}")
        return False

def test_video_processing(video_path: str):
    """Test video processing with available models."""
    print(f"\nTesting video processing with: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return False
    
    available_models = PoseProcessorManager.get_available_models()
    
    for model_type in available_models.keys():
        print(f"\nTesting {model_type}...")
        
        try:
            with UnifiedPoseProcessor(model_type=model_type) as processor:
                success = processor.process_video(video_path)
                
                if success:
                    print(f"✓ {model_type} processing successful")
                else:
                    print(f"✗ {model_type} processing failed")
                    
        except Exception as e:
            print(f"✗ {model_type} processing error: {e}")
    
    return True

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
    print("- MeTRAbs will work if metrabs and torch are installed")
    print("- Both models provide the same interface for easy switching")
    print("- Video processing test requires a video file in the data directory")

if __name__ == "__main__":
    main()
