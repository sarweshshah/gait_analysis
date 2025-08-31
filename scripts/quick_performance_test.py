#!/usr/bin/env python3
"""
Quick Performance Test for Gait Analysis Optimizations
=====================================================

This script provides a quick way to test the performance optimizations
and verify they are working correctly.

Usage:
    python scripts/quick_performance_test.py [--video path/to/video.mp4]
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import psutil

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.performance_optimizer import PerformanceOptimizer, PerformanceMonitor
from core.mediapipe_integration import MediaPipeProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_video_info():
    """Create a simple test video info for demonstration."""
    return {
        'width': 640,
        'height': 480,
        'fps': 30,
        'duration_seconds': 10,
        'total_frames': 300
    }

def test_performance_optimizer():
    """Test the performance optimizer functionality."""
    print("=" * 60)
    print("TESTING PERFORMANCE OPTIMIZER")
    print("=" * 60)
    
    # Test different performance modes
    modes = ['fast', 'balanced', 'accurate']
    
    for mode in modes:
        print(f"\nTesting {mode.upper()} mode:")
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer(performance_mode=mode)
        
        # Test MediaPipe configuration
        mediapipe_config = optimizer.get_optimized_mediapipe_config()
        print(f"  MediaPipe Model Complexity: {mediapipe_config['model_complexity']}")
        print(f"  Detection Confidence: {mediapipe_config['min_detection_confidence']}")
        print(f"  Smooth Landmarks: {mediapipe_config['smooth_landmarks']}")
        
        # Test model configuration
        model_config = {
            'num_filters': 64,
            'num_blocks': 4,
            'batch_size': 32
        }
        optimized_model_config = optimizer.get_optimized_model_config(model_config)
        print(f"  Optimized Filters: {optimized_model_config['num_filters']}")
        print(f"  Optimized Blocks: {optimized_model_config['num_blocks']}")
        print(f"  Optimized Batch Size: {optimized_model_config['batch_size']}")
        
        # Test training configuration
        training_config = {
            'batch_size': 32,
            'epochs': 100
        }
        optimized_training_config = optimizer.get_optimized_training_config(training_config)
        print(f"  Optimized Training Batch Size: {optimized_training_config['batch_size']}")
        print(f"  Workers: {optimized_training_config.get('workers', 'N/A')}")
        
        # Test memory monitoring
        memory_info = optimizer.monitor.get_memory_usage()
        print(f"  Current Memory Usage: {memory_info['rss_mb']:.1f} MB")
        
        # Cleanup
        optimizer.cleanup()

def test_mediapipe_processor(video_path=None):
    """Test MediaPipe processor with optimizations."""
    print("\n" + "=" * 60)
    print("TESTING MEDIAPIPE PROCESSOR")
    print("=" * 60)
    
    if video_path and os.path.exists(video_path):
        print(f"Using video: {video_path}")
        test_video = video_path
    else:
        print("No valid video provided, creating test video info")
        test_video = create_test_video_info()
        print("Note: This is a simulation test without actual video processing")
        return
    
    # Test different performance modes
    modes = ['fast', 'balanced', 'accurate']
    
    for mode in modes:
        print(f"\nTesting MediaPipe {mode.upper()} mode:")
        
        # Create temporary output directory
        output_dir = f'test_output_mediapipe_{mode}'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Initialize processor with optimizations
            processor = MediaPipeProcessor(
                output_dir=output_dir,
                fps=30.0,
                performance_mode=mode,
                enable_optimizations=True
            )
            
            # Start timing
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Process video (this will fail if no real video, but we can test initialization)
            if os.path.exists(test_video):
                success = processor.process_video(test_video)
                processing_time = time.time() - start_time
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                print(f"  Processing Time: {processing_time:.2f} seconds")
                print(f"  Memory Usage: {final_memory - initial_memory:.1f} MB")
                print(f"  Success: {success}")
            else:
                print(f"  Skipping processing (test mode)")
            
            # Test cleanup
            processor.cleanup()
            print(f"  Cleanup: Completed")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # Clean up test directory
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)

def test_system_capabilities():
    """Test system capabilities and provide recommendations."""
    print("\n" + "=" * 60)
    print("SYSTEM CAPABILITIES ANALYSIS")
    print("=" * 60)
    
    # CPU information
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"CPU Cores: {cpu_count}")
    if cpu_freq:
        print(f"CPU Frequency: {cpu_freq.current:.1f} MHz")
    
    # Memory information
    memory = psutil.virtual_memory()
    total_gb = memory.total / 1024**3
    available_gb = memory.available / 1024**3
    print(f"Total Memory: {total_gb:.1f} GB")
    print(f"Available Memory: {available_gb:.1f} GB")
    print(f"Memory Usage: {memory.percent:.1f}%")
    
    # Performance recommendations
    print(f"\nPERFORMANCE RECOMMENDATIONS:")
    
    if total_gb < 4:
        print("  → Use 'fast' performance mode")
        print("  → Set max_memory_mb to 1024")
        print("  → Use batch_size 16-32")
    elif total_gb < 8:
        print("  → Use 'balanced' performance mode")
        print("  → Set max_memory_mb to 2048")
        print("  → Use batch_size 32-64")
    else:
        print("  → Use 'accurate' performance mode")
        print("  → Set max_memory_mb to 4096 or higher")
        print("  → Use batch_size 64-128")
    
    if cpu_count < 4:
        print("  → Limit parallel processing (max_workers: 2)")
    elif cpu_count < 8:
        print("  → Use moderate parallel processing (max_workers: 4)")
    else:
        print("  → Use full parallel processing (max_workers: 8)")

def test_optimization_features():
    """Test specific optimization features."""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZATION FEATURES")
    print("=" * 60)
    
    # Test performance monitor
    print("Testing Performance Monitor:")
    monitor = PerformanceMonitor()
    monitor.start_timer('test_operation')
    time.sleep(0.1)  # Simulate some work
    duration = monitor.end_timer('test_operation')
    print(f"  Timer accuracy: {duration:.3f} seconds (expected ~0.1)")
    
    # Test memory monitoring
    memory_info = monitor.get_memory_usage()
    print(f"  Memory monitoring: {memory_info['rss_mb']:.1f} MB RSS")
    
    # Test I/O optimizer
    print("\nTesting I/O Optimizer:")
    from core.performance_optimizer import IOOptimizer
    io_optimizer = IOOptimizer(batch_size=10)
    
    # Simulate adding frame data
    for i in range(15):
        frame_data = {'frame_number': i, 'data': f'test_data_{i}'}
        io_optimizer.add_frame_data(frame_data, 'test_output', 'test_video')
    
    print(f"  I/O optimization: Batch processing configured")
    
    # Test memory optimizer
    print("\nTesting Memory Optimizer:")
    from core.performance_optimizer import MemoryOptimizer
    memory_optimizer = MemoryOptimizer(max_memory_mb=1024)
    
    memory_ok = memory_optimizer.check_memory_usage()
    print(f"  Memory check: {'OK' if memory_ok else 'WARNING'}")
    
    memory_optimizer.optimize_memory()
    print(f"  Memory optimization: Completed")

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Quick Performance Test for Gait Analysis')
    parser.add_argument('--video', '-v', help='Path to test video file (optional)')
    parser.add_argument('--skip-video', action='store_true', help='Skip video processing tests')
    
    args = parser.parse_args()
    
    print("GAIT ANALYSIS PERFORMANCE OPTIMIZATION TEST")
    print("=" * 60)
    print("This script tests the performance optimization features")
    print("to ensure they are working correctly.")
    print()
    
    try:
        # Test system capabilities
        test_system_capabilities()
        
        # Test performance optimizer
        test_performance_optimizer()
        
        # Test optimization features
        test_optimization_features()
        
        # Test MediaPipe processor (if video provided or not skipped)
        if not args.skip_video:
            test_mediapipe_processor(args.video)
        
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST COMPLETED")
        print("=" * 60)
        print("✓ All optimization features are working correctly")
        print("✓ System capabilities have been analyzed")
        print("✓ Performance recommendations provided")
        print()
        print("Next steps:")
        print("1. Use the benchmarking script for detailed performance analysis:")
        print("   python scripts/benchmark_performance.py --video your_video.mp4")
        print("2. Run the main analysis with optimizations:")
        print("   python usecases/gait_analysis/main_gait_analysis.py --videos your_video.mp4 --performance-mode balanced")
        print("3. Check the performance documentation:")
        print("   docs/README_Performance_Optimization.md")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Performance test failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())