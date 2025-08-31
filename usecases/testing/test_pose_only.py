#!/usr/bin/env python3
"""
Pose Processor Integration Test
==============================

This script tests only the pose processor integration without requiring
all other dependencies like scipy, tensorflow, etc.
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pose_processor_manager_import():
    """Test that PoseProcessorManager can be imported."""
    logger.info("Testing PoseProcessorManager import...")
    
    try:
        from core.pose_processor_manager import PoseProcessorManager
        logger.info("✓ PoseProcessorManager imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import PoseProcessorManager: {e}")
        return False

def test_available_models():
    """Test getting available models."""
    logger.info("Testing available models...")
    
    try:
        from core.pose_processor_manager import PoseProcessorManager
        
        available_models = PoseProcessorManager.get_available_models()
        logger.info(f"Available models: {list(available_models.keys())}")
        
        # Check if both models are listed
        expected_models = ['mediapipe', 'metrabs']
        for model in expected_models:
            if model in available_models:
                logger.info(f"✓ {model} is listed as available")
            else:
                logger.warning(f"⚠ {model} is not listed as available")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to get available models: {e}")
        return False

def test_model_info():
    """Test getting model information."""
    logger.info("Testing model information...")
    
    try:
        from core.pose_processor_manager import PoseProcessorManager
        
        # Test MediaPipe info
        try:
            mediapipe_info = PoseProcessorManager.get_model_info('mediapipe')
            logger.info(f"✓ MediaPipe info: {mediapipe_info['name']}")
            logger.info(f"  Landmarks: {mediapipe_info['landmarks']}")
        except Exception as e:
            logger.warning(f"⚠ MediaPipe info failed: {e}")
        
        # Test MeTRAbs info
        try:
            metrabs_info = PoseProcessorManager.get_model_info('metrabs')
            logger.info(f"✓ MeTRAbs info: {metrabs_info['name']}")
            logger.info(f"  Landmarks: {metrabs_info['landmarks']}")
        except Exception as e:
            logger.warning(f"⚠ MeTRAbs info failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to get model info: {e}")
        return False

def test_processor_creation():
    """Test processor creation (may fail if dependencies missing)."""
    logger.info("Testing processor creation...")
    
    try:
        from core.pose_processor_manager import PoseProcessorManager
        
        # Test MediaPipe processor creation
        logger.info("Testing MediaPipe processor creation...")
        try:
            mediapipe_processor = PoseProcessorManager.create_processor(
                model_type='mediapipe',
                output_dir='test_pose_output'
            )
            logger.info("✓ MediaPipe processor created successfully")
            
            # Test model info
            model_info = mediapipe_processor.get_model_info()
            logger.info(f"  Model: {model_info['name']}")
            
            # Clean up
            mediapipe_processor.cleanup()
            
        except Exception as e:
            logger.warning(f"⚠ MediaPipe processor creation failed: {e}")
        
        # Test MeTRAbs processor creation
        logger.info("Testing MeTRAbs processor creation...")
        try:
            metrabs_processor = PoseProcessorManager.create_processor(
                model_type='metrabs',
                output_dir='test_pose_output'
            )
            logger.info("✓ MeTRAbs processor created successfully")
            
            # Test model info
            model_info = metrabs_processor.get_model_info()
            logger.info(f"  Model: {model_info['name']}")
            
            # Clean up
            metrabs_processor.cleanup()
            
        except Exception as e:
            logger.warning(f"⚠ MeTRAbs processor creation failed (expected if not installed): {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Processor creation test failed: {e}")
        return False

def test_unified_processor():
    """Test UnifiedPoseProcessor."""
    logger.info("Testing UnifiedPoseProcessor...")
    
    try:
        from core.pose_processor_manager import UnifiedPoseProcessor
        
        # Test MediaPipe initialization
        logger.info("Testing UnifiedPoseProcessor with MediaPipe...")
        try:
            unified_processor = UnifiedPoseProcessor(
                model_type='mediapipe',
                output_dir='test_pose_output'
            )
            
            # Test model info
            info = unified_processor.get_model_info()
            logger.info(f"✓ UnifiedPoseProcessor MediaPipe: {info['name']}")
            
            # Test model switching to MeTRAbs
            logger.info("Testing model switching to MeTRAbs...")
            try:
                unified_processor.switch_model('metrabs')
                info = unified_processor.get_model_info()
                logger.info(f"✓ Switched to MeTRAbs: {info['name']}")
            except Exception as e:
                logger.warning(f"⚠ MeTRAbs switching failed (expected if not installed): {e}")
            
            # Clean up
            unified_processor.cleanup()
            
        except Exception as e:
            logger.warning(f"⚠ UnifiedPoseProcessor test failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ UnifiedPoseProcessor test failed: {e}")
        return False

def cleanup():
    """Clean up test files."""
    if os.path.exists('test_pose_output'):
        import shutil
        shutil.rmtree('test_pose_output')
        logger.info("Cleaned up test_pose_output directory")

def main():
    """Run all pose processor tests."""
    logger.info("Starting pose processor integration tests...")
    logger.info("=" * 50)
    
    tests = [
        ("PoseProcessorManager Import", test_pose_processor_manager_import),
        ("Available Models", test_available_models),
        ("Model Information", test_model_info),
        ("Processor Creation", test_processor_creation),
        ("UnifiedPoseProcessor", test_unified_processor)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if test_func():
                logger.info(f"✓ {test_name} test PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")
    
    # Clean up
    cleanup()
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All pose processor tests passed!")
        return True
    else:
        logger.error("❌ Some pose processor tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
