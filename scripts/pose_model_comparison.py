#!/usr/bin/env python3
"""
Pose Model Comparison Script
===========================

This script demonstrates how to compare and switch between different pose estimation models
(MediaPipe and MeTRAbs) for gait analysis.

Usage:
    python pose_model_comparison.py --video path/to/video.mp4 --compare
    python pose_model_comparison.py --video path/to/video.mp4 --model metrabs
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_processor_manager import UnifiedPoseProcessor, PoseProcessorManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_model_info():
    """Print information about available models."""
    print("\n" + "="*60)
    print("AVAILABLE POSE ESTIMATION MODELS")
    print("="*60)
    
    available_models = PoseProcessorManager.get_available_models()
    
    for model_key, model_name in available_models.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        
        try:
            info = PoseProcessorManager.get_model_info(model_key)
            print(f"Description: {info['description']}")
            print(f"Landmarks: {info['landmarks']}")
            print(f"Output Keypoints: {info['keypoints']}")
            
            print("\nAdvantages:")
            for advantage in info['advantages']:
                print(f"  • {advantage}")
            
            print("\nDisadvantages:")
            for disadvantage in info['disadvantages']:
                print(f"  • {disadvantage}")
            
            print("\nBest for:")
            for use_case in info['best_for']:
                print(f"  • {use_case}")
                
        except Exception as e:
            print(f"Error getting info for {model_key}: {e}")

def compare_models(video_path: str, output_dir: str = "model_comparison"):
    """
    Compare different pose models on the same video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for comparison results
    """
    print(f"\nComparing pose models on: {video_path}")
    print("="*60)
    
    # Create unified processor for comparison
    with UnifiedPoseProcessor(
        model_type='mediapipe',  # Default, will be overridden
        output_dir=output_dir
    ) as processor:
        
        # Run comparison
        results = processor.compare_models(video_path)
        
        # Print results
        print("\nCOMPARISON RESULTS")
        print("-" * 40)
        
        for model_type, result in results.items():
            print(f"\n{model_type.upper()}:")
            
            if result['success']:
                print(f"  ✓ Processing successful")
                print(f"  ⏱️  Processing time: {result['processing_time']:.2f} seconds")
                print(f"  📊 Model: {result['model_info']['name']}")
                print(f"  📍 Landmarks: {result['model_info']['landmarks']}")
            else:
                print(f"  ✗ Processing failed")
                if 'error' in result:
                    print(f"  ❌ Error: {result['error']}")
        
        # Save detailed results
        results_file = os.path.join(output_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")

def process_with_model(video_path: str, model_type: str, output_dir: str = None):
    """
    Process video with a specific pose model.
    
    Args:
        video_path: Path to video file
        model_type: Type of pose model to use
        output_dir: Output directory (optional)
    """
    if output_dir is None:
        output_dir = f"{model_type}_output"
    
    print(f"\nProcessing video with {model_type.upper()}")
    print("="*60)
    
    # Get model info
    try:
        model_info = PoseProcessorManager.get_model_info(model_type)
        print(f"Model: {model_info['name']}")
        print(f"Description: {model_info['description']}")
        print(f"Landmarks: {model_info['landmarks']}")
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
    
    # Process video
    with UnifiedPoseProcessor(
        model_type=model_type,
        output_dir=output_dir
    ) as processor:
        
        success = processor.process_video(video_path)
        
        if success:
            print(f"✓ Successfully processed video with {model_type}")
            print(f"📁 Output saved to: {processor.processor.output_dir}")
        else:
            print(f"✗ Failed to process video with {model_type}")

def switch_models_demo(video_path: str, output_dir: str = "model_switching_demo"):
    """
    Demonstrate switching between models on the same video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
    """
    print(f"\nDemonstrating model switching on: {video_path}")
    print("="*60)
    
    available_models = list(PoseProcessorManager.get_available_models().keys())
    
    with UnifiedPoseProcessor(
        model_type=available_models[0],
        output_dir=output_dir
    ) as processor:
        
        for model_type in available_models:
            print(f"\n🔄 Switching to {model_type.upper()}...")
            
            # Switch model
            processor.switch_model(model_type)
            
            # Process video
            success = processor.process_video(video_path)
            
            if success:
                print(f"✓ Successfully processed with {model_type}")
            else:
                print(f"✗ Failed to process with {model_type}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Compare and switch between pose estimation models'
    )
    
    parser.add_argument('--video', '-v', required=True,
                       help='Path to video file to process')
    parser.add_argument('--model', '-m', choices=['mediapipe', 'metrabs'],
                       default='mediapipe',
                       help='Pose model to use (default: mediapipe)')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare all available models')
    parser.add_argument('--switch-demo', '-s', action='store_true',
                       help='Demonstrate model switching')
    parser.add_argument('--info', '-i', action='store_true',
                       help='Show information about available models')
    parser.add_argument('--output', '-o', default='pose_output',
                       help='Output directory (default: pose_output)')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Show model information
    if args.info:
        print_model_info()
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run requested operation
    if args.compare:
        compare_models(args.video, args.output)
    elif args.switch_demo:
        switch_models_demo(args.video, args.output)
    else:
        process_with_model(args.video, args.model, args.output)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
