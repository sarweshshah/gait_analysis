#!/usr/bin/env python3
"""
Simple MediaPipe Test Script
===========================

This script tests the MediaPipe integration to ensure it works correctly.
"""

import sys
import os
import numpy as np
import cv2

def test_mediapipe_import():
    """Test if MediaPipe can be imported."""
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe imported successfully (version: {mp.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import MediaPipe: {e}")
        return False

def test_mediapipe_processor():
    """Test MediaPipe processor initialization."""
    try:
        from core.mediapipe_integration import MediaPipeProcessor
        
        # Create processor
        processor = MediaPipeProcessor(
            output_dir='test_output',
            fps=30.0,
            model_complexity=1
        )
        
        print("✓ MediaPipe processor created successfully")
        print(f"  - Landmarks: {len(processor.mediapipe_landmarks)}")
        print(f"  - Body25 keypoints: {len(processor.body_25_keypoints)}")
        print(f"  - Gait keypoints: {len(processor.gait_keypoints)}")
        print(f"  - Landmark mapping: {len(processor.landmark_mapping)} entries")
        
        # Clean up
        processor.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Failed to create MediaPipe processor: {e}")
        return False

def test_mediapipe_pose_detection():
    """Test MediaPipe pose detection on a simple image."""
    try:
        import mediapipe as mp
        
        # Create a simple test image (white background)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # Process image
        results = pose.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        
        print("✓ MediaPipe pose detection test completed")
        print(f"  - Pose landmarks detected: {results.pose_landmarks is not None}")
        
        # Clean up
        pose.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to test MediaPipe pose detection: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing MediaPipe Integration")
    print("=" * 40)
    
    tests = [
        ("MediaPipe Import", test_mediapipe_import),
        ("MediaPipe Processor", test_mediapipe_processor),
        ("Pose Detection", test_mediapipe_pose_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  ❌ {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MediaPipe integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
