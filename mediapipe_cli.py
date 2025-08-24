#!/usr/bin/env python3
"""
MediaPipe Command Line Interface for Gait Analysis
=================================================

Simple command-line interface for processing videos with MediaPipe pose estimation.

Usage:
    python mediapipe_cli.py --video path/to/video.mp4 --output output_dir/
    python mediapipe_cli.py --webcam --duration 10
"""

import argparse
import sys
import os
from pathlib import Path
from mediapipe_integration import MediaPipeProcessor

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='MediaPipe Pose Estimation for Gait Analysis')
    parser.add_argument('--video', '-v', help='Input video file path')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam instead of video file')
    parser.add_argument('--output', '-o', default='mediapipe_output', help='Output directory for results')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Video frame rate')
    parser.add_argument('--duration', '-d', type=float, default=10.0, help='Recording duration for webcam (seconds)')
    parser.add_argument('--model-complexity', '-m', type=int, default=1, choices=[0, 1, 2], 
                       help='Model complexity (0=fast, 1=balanced, 2=accurate)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, 
                       help='Minimum confidence for pose detection')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5, 
                       help='Minimum confidence for pose tracking')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and not args.webcam:
        parser.error("Either --video or --webcam must be specified")
    
    if args.video and not os.path.exists(args.video):
        parser.error(f"Video file not found: {args.video}")
    
    # Create MediaPipe processor
    processor = MediaPipeProcessor(
        output_dir=args.output,
        fps=args.fps,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    try:
        if args.webcam:
            print(f"Starting webcam recording for {args.duration} seconds...")
            success = processor.process_webcam(duration=args.duration)
            if success:
                print(f"✓ Webcam processing completed. Results saved to {args.output}")
            else:
                print("✗ Webcam processing failed")
                sys.exit(1)
        else:
            print(f"Processing video: {args.video}")
            success = processor.process_video(args.video)
            if success:
                print(f"✓ Video processing completed. Results saved to {args.output}")
            else:
                print("✗ Video processing failed")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
