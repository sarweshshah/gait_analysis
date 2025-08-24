#!/usr/bin/env python3
"""
Legacy Script Runner for OpenPose-based Gait Analysis
====================================================

This script provides a convenient way to run the legacy OpenPose-based
gait analysis scripts with proper configuration and error handling.

Author: Gait Analysis System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = ['cv2', 'numpy', 'imutils']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist."""
    required_files = [
        "dnn_models/pose/coco/pose_deploy_linevec.prototxt",
        "dnn_models/pose/coco/pose_iter_440000.caffemodel",
        "dnn_models/object_detection/MobileNetSSD_deploy.prototxt",
        "dnn_models/object_detection/MobileNetSSD_deploy.caffemodel"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required model files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run the setup script first:")
        if os.name == 'nt':  # Windows
            print("  setup_environment.bat")
        else:  # Unix/Linux/Mac
            print("  ./setup_environment.sh")
        return False
    
    return True

def check_video_file(video_path):
    """Check if video file exists and is accessible."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Unsupported video format: {video_path}")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        return False
    
    return True

def run_script(script_name, video_path=None, output_dir="results"):
    """Run a specific script with the given video file."""
    script_path = f"{script_name}.py"
    
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare command
    cmd = [sys.executable, script_path]
    
    print(f"Running {script_name}...")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    try:
        # Run the script
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n{script_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n{script_name} interrupted by user")
        return False

def list_available_scripts():
    """List all available scripts."""
    scripts = [
        "poseTrailVideo",
        "poseDetectVideo", 
        "hipsTrailVideo",
        "poseAndObjectDetectVideo"
    ]
    
    print("Available scripts:")
    for i, script in enumerate(scripts, 1):
        status = "✓" if os.path.exists(f"{script}.py") else "✗"
        print(f"  {i}. {status} {script}.py")
    
    print("\nScript descriptions:")
    print("  poseTrailVideo.py: Shows pose keypoints with trail effect")
    print("  poseDetectVideo.py: Shows pose detection with skeleton overlay")
    print("  hipsTrailVideo.py: Shows hip keypoints with trail effect")
    print("  poseAndObjectDetectVideo.py: Shows both pose and object detection")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run legacy OpenPose-based gait analysis scripts"
    )
    parser.add_argument(
        "script",
        nargs="?",
        choices=["poseTrailVideo", "poseDetectVideo", "hipsTrailVideo", "poseAndObjectDetectVideo"],
        help="Script to run"
    )
    parser.add_argument(
        "--video", "-v",
        help="Path to video file (will override script's default)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scripts"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check dependencies and model files"
    )
    
    args = parser.parse_args()
    
    # List available scripts
    if args.list:
        list_available_scripts()
        return
    
    # Check dependencies and model files
    if args.check:
        print("Checking dependencies...")
        deps_ok = check_dependencies()
        
        print("\nChecking model files...")
        models_ok = check_model_files()
        
        if deps_ok and models_ok:
            print("\n✓ All checks passed!")
        else:
            print("\n✗ Some checks failed. Please fix the issues above.")
        return
    
    # If no script specified, show help
    if not args.script:
        parser.print_help()
        print("\nUse --list to see available scripts")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model files
    if not check_model_files():
        return
    
    # Check video file if provided
    if args.video and not check_video_file(args.video):
        return
    
    # Run the script
    success = run_script(args.script, args.video, args.output)
    
    if success:
        print(f"\nResults saved in: {args.output}")
    else:
        print(f"\nFailed to run {args.script}")

if __name__ == "__main__":
    main()
