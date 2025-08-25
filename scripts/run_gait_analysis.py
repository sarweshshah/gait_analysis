#!/usr/bin/env python3
"""
Gait Analysis Entry Point
========================

This script is the main entry point for running gait analysis.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from usecases.gait_analysis import GaitAnalysisPipeline, create_default_config
from core.utils.config import ConfigManager
from core.utils.logging_config import setup_logging

def main():
    """Main entry point for gait analysis."""
    parser = argparse.ArgumentParser(
        description="Run gait analysis pipeline with feature toggles",
        epilog="""
Examples:
  # Full pipeline (pose detection + gait analysis)
  python scripts/run_gait_analysis.py --input video.mp4
  
  # Pose detection only
  python scripts/run_gait_analysis.py --input video.mp4 --pose-detection-only
  
  # Pose detection with real-time visualization
  python scripts/run_gait_analysis.py --input video.mp4 --with-visualization
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="gait_analysis",
        help="Configuration file name (without .json extension)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/gait_analysis",
        help="Output directory for results"
    )
    
    # Feature toggle arguments
    parser.add_argument(
        "--pose-detection-only", 
        action="store_true",
        help="Run only pose detection without gait analysis"
    )
    parser.add_argument(
        "--with-visualization", 
        action="store_true",
        help="Run pose detection with real-time visualization"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = f"results/logs/gait_analysis_{Path(args.input).stem}.log"
    logger = setup_logging(log_file=log_file, log_level=args.log_level)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Update config with command line arguments
        config["input_file"] = args.input
        config["output_dir"] = args.output
        
        # Handle feature toggles (pose detection is always enabled)
        if args.pose_detection_only:
            config["enable_realtime_visualization"] = False
            config["enable_gait_analysis"] = False
        elif args.with_visualization:
            config["enable_realtime_visualization"] = True
            config["enable_gait_analysis"] = False
        
        # Create and run pipeline
        pipeline = GaitAnalysisPipeline(config)
        
        logger.info(f"Starting gait analysis for: {args.input}")
        logger.info(f"Output directory: {args.output}")
        
        # Run the pipeline
        results = pipeline.run_complete_pipeline([args.input])
        
        if results:
            logger.info("Gait analysis completed successfully")
        else:
            logger.error("Gait analysis failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running gait analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
