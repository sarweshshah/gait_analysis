"""
Main Gait Analysis Pipeline
==========================

This script demonstrates the complete gait analysis pipeline:
1. OpenPose processing with BODY_25 model
2. Data preprocessing and feature extraction
3. TCN model training and evaluation
4. Results visualization and analysis

Author: Gait Analysis System
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import json

# Import our custom modules
from core.mediapipe_integration import MediaPipeProcessor
from core.gait_data_preprocessing import GaitDataPreprocessor
from core.tcn_gait_model import create_gait_tcn_model, compile_gait_model
from core.gait_training import GaitTrainer, GaitMetrics
from .gait_events import BasicGaitEvents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gait_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GaitAnalysisPipeline:
    """
    Complete gait analysis pipeline integrating all components.
    Supports feature toggles for different analysis modes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the gait analysis pipeline.
        
        Args:
            config: Configuration dictionary with feature toggles
        """
        self.config = config
        self.results_dir = config.get('results_dir', 'gait_analysis_results')
        
        # Feature toggles (pose detection is always enabled)
        self.enable_pose_detection = True  # Always enabled - fundamental requirement
        self.enable_realtime_visualization = config.get('enable_realtime_visualization', False)
        self.enable_gait_analysis = config.get('enable_gait_analysis', True)
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.mediapipe_processor = None
        self.data_preprocessor = None
        self.trainer = None
        self.visualizer = None
        self.events_detector = None
        
        logger.info("Initialized Gait Analysis Pipeline")
        logger.info(f"Features enabled: Pose Detection=Always, "
                   f"Real-time Visualization={self.enable_realtime_visualization}, "
                   f"Gait Analysis={self.enable_gait_analysis}")
    
    def setup_mediapipe(self):
        """Setup MediaPipe processor for pose estimation."""
        # Pose detection is always enabled - fundamental requirement
            
        logger.info("Setting up MediaPipe processor...")
        
        self.mediapipe_processor = MediaPipeProcessor(
            output_dir=self.config.get('mediapipe_output_dir', 'mediapipe_output'),
            fps=self.config.get('fps', 30.0),
            model_complexity=self.config.get('model_complexity', 1),
            min_detection_confidence=self.config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
        )
        
        logger.info("MediaPipe setup completed successfully")
        return True
    
    def setup_visualization(self):
        """Setup real-time visualization."""
        if not self.enable_realtime_visualization:
            logger.info("Real-time visualization disabled, skipping setup")
            return True
            
        logger.info("Setting up real-time visualization...")
        
        from .features.realtime_pose_visualization import RealTimePoseVisualizer
        
        self.visualizer = RealTimePoseVisualizer(
            model_complexity=self.config.get('model_complexity', 1),
            min_detection_confidence=self.config.get('min_detection_confidence', 0.5)
        )
        
        logger.info("Real-time visualization setup completed")
        return True
    
    def setup_data_preprocessing(self):
        """Setup data preprocessor."""
        logger.info("Setting up data preprocessor...")
        
        self.data_preprocessor = GaitDataPreprocessor(
            confidence_threshold=self.config.get('confidence_threshold', 0.3),
            filter_cutoff=self.config.get('filter_cutoff', 6.0),
            filter_order=self.config.get('filter_order', 4),
            window_size=self.config.get('window_size', 30),
            overlap=self.config.get('overlap', 0.5)
        )
        
        logger.info("Data preprocessor setup completed")

    def setup_events_detector(self):
        """Setup rule-based gait events detector."""
        logger.info("Setting up gait events detector...")
        self.events_detector = BasicGaitEvents(
            fps=self.config.get('fps', 30.0),
            confidence_threshold=self.config.get('confidence_threshold', 0.3),
            keypoint_format='body25'  # preprocessor outputs BODY_25 mapping
        )
        logger.info("Gait events detector setup completed")
    
    def setup_trainer(self):
        """Setup TCN trainer."""
        logger.info("Setting up TCN trainer...")
        
        model_config = {
            'num_classes': self.config.get('num_classes', 4),
            'num_filters': self.config.get('num_filters', 64),
            'kernel_size': self.config.get('kernel_size', 3),
            'num_blocks': self.config.get('num_blocks', 4),
            'dropout_rate': self.config.get('dropout_rate', 0.2),
            'learning_rate': self.config.get('learning_rate', 0.001)
        }
        
        self.trainer = GaitTrainer(
            data_preprocessor=self.data_preprocessor,
            model_config=model_config,
            task_type=self.config.get('task_type', 'phase_detection'),
            output_dir=self.results_dir
        )
        
        logger.info("TCN trainer setup completed")
    
    def process_videos(self, video_paths: List[str]) -> List[str]:
        """
        Process videos with MediaPipe to extract pose data.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of JSON output directories
        """
        logger.info(f"Processing {len(video_paths)} videos with MediaPipe...")
        
        json_dirs = []
        for i, video_path in enumerate(video_paths):
            try:
                logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                
                success = self.mediapipe_processor.process_video(video_path)
                
                if success:
                    json_dirs.append(self.mediapipe_processor.output_dir)
                    logger.info(f"Successfully processed {video_path}")
                else:
                    logger.warning(f"Failed to process {video_path}, skipping")
                    
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(json_dirs)} videos")
        return json_dirs

    def run_event_detection(self, json_dirs: List[str]) -> Dict[str, Any]:
        """
        Run rule-based gait events detection on processed MediaPipe outputs.
        """
        if self.data_preprocessor is None:
            self.setup_data_preprocessing()
        if self.events_detector is None:
            self.setup_events_detector()

        results: Dict[str, Any] = {"videos": []}
        for dir_path in json_dirs:
            try:
                processed = self.data_preprocessor.process_video_sequence(
                    json_directory=dir_path,
                    fps=self.config.get('fps', 30.0)
                )
                keypoints_seq = processed['keypoints_sequence'][:, :, :2]  # (n_frames, 25, 2)
                events = self.events_detector.detect_events(keypoints_seq)
                metrics = self.events_detector.calculate_gait_metrics(events)
                results["videos"].append({
                    "json_dir": dir_path,
                    "events": events,
                    "metrics": metrics,
                    "metadata": processed.get('metadata', {})
                })
            except Exception as e:
                logger.error(f"Event detection failed for {dir_path}: {e}", exc_info=True)
                results["videos"].append({
                    "json_dir": dir_path,
                    "error": str(e)
                })

        # Save consolidated results
        out_file = os.path.join(self.results_dir, 'gait_events_results.json')
        try:
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Gait events results saved to {out_file}")
        except Exception as e:
            logger.warning(f"Failed to save gait events results: {e}")

        return results
    
    def prepare_training_data(self, json_dirs: List[str], labels: List[int] = None) -> tuple:
        """
        Prepare training data from OpenPose JSON outputs.
        
        Args:
            json_dirs: List of JSON output directories
            labels: Optional labels for each video
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Preparing training data...")
        
        try:
            features, labels = self.trainer.prepare_data(json_dirs, labels)
            logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[2]} features")
            return features, labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def train_model(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Train TCN model with cross-validation.
        
        Args:
            features: Input features
            labels: Target labels
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        try:
            cv_results = self.trainer.train_with_cross_validation(
                features=features,
                labels=labels,
                n_folds=self.config.get('n_folds', 5),
                epochs=self.config.get('epochs', 100),
                batch_size=self.config.get('batch_size', 32),
                validation_split=self.config.get('validation_split', 0.2),
                early_stopping_patience=self.config.get('early_stopping_patience', 15)
            )
            
            logger.info("Model training completed successfully")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def evaluate_results(self, cv_results: Dict[str, Any]):
        """Evaluate and visualize training results."""
        logger.info("Evaluating model performance...")
        
        try:
            # Evaluate performance
            self.trainer.evaluate_model_performance(cv_results)
            
            # Plot training curves
            self.trainer.plot_training_curves(cv_results)
            
            # Save detailed results
            self._save_detailed_results(cv_results)
            
            logger.info("Results evaluation completed")
            
        except Exception as e:
            logger.error(f"Error during results evaluation: {e}")
            raise
    
    def _save_detailed_results(self, cv_results: Dict[str, Any]):
        """Save detailed results and analysis."""
        results_file = os.path.join(self.results_dir, 'detailed_results.json')
        
        # Prepare results for saving
        save_results = {
            'config': self.config,
            'overall_metrics': cv_results['overall_metrics'],
            'fold_scores': cv_results['fold_scores'],
            'summary': {
                'best_accuracy': max([score['test_accuracy'] for score in cv_results['fold_scores']]),
                'mean_accuracy': cv_results['overall_metrics']['mean_accuracy'],
                'std_accuracy': cv_results['overall_metrics']['std_accuracy'],
                'best_fold': max(cv_results['fold_scores'], key=lambda x: x['test_accuracy'])['fold']
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {results_file}")
    
    def run_complete_pipeline(self, video_paths: List[str], labels: List[int] = None):
        """
        Run the complete gait analysis pipeline with feature toggles.
        
        Args:
            video_paths: List of video file paths
            labels: Optional labels for each video
        """
        logger.info("Starting complete gait analysis pipeline...")
        
        try:
            # Setup components (pose detection is always enabled)
            if not self.setup_mediapipe():
                raise RuntimeError("MediaPipe setup failed")
            
            if self.enable_realtime_visualization:
                if not self.setup_visualization():
                    raise RuntimeError("Visualization setup failed")
            
            if self.enable_gait_analysis:
                self.setup_data_preprocessing()
                self.setup_trainer()
            
            # Run based on enabled features (pose detection is always enabled)
            if self.enable_gait_analysis:
                # Full pipeline: pose detection + gait analysis
                json_dirs = self.process_videos(video_paths)
                
                if not json_dirs:
                    raise RuntimeError("No videos were successfully processed")
                
                # Branch by task type
                if self.config.get('task_type', 'phase_detection') == 'event_detection':
                    event_results = self.run_event_detection(json_dirs)
                    logger.info("Gait event detection completed successfully!")
                    return event_results
                else:
                    features, labels = self.prepare_training_data(json_dirs, labels)
                    cv_results = self.train_model(features, labels)
                    self.evaluate_results(cv_results)
                    
                    logger.info("Complete gait analysis pipeline finished successfully!")
                    return cv_results
                
            elif self.enable_realtime_visualization:
                # Pose detection + real-time visualization
                if video_paths:
                    logger.info(f"Starting real-time visualization for: {video_paths[0]}")
                    self.visualizer.process_video(
                        video_path=video_paths[0],
                        loop_video=self.config.get('loop_video', False)
                    )
                    return {"visualization_completed": True}
                
            else:
                # Pose detection only
                json_dirs = self.process_videos(video_paths)
                logger.info("Pose detection completed successfully!")
                return {"pose_detection_completed": True, "json_dirs": json_dirs}
            
            logger.info("Pipeline completed successfully!")
            return {"pipeline_completed": True}
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_pose_detection_only(self, video_paths: List[str]):
        """Run only pose detection without gait analysis."""
        self.enable_gait_analysis = False
        self.enable_realtime_visualization = False
        return self.run_complete_pipeline(video_paths)
    
    def run_with_visualization(self, video_path: str):
        """Run pose detection with real-time visualization."""
        self.enable_gait_analysis = False
        self.enable_realtime_visualization = True
        return self.run_complete_pipeline([video_path])

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for gait analysis."""
    return {
        # Feature toggles (pose detection is always enabled)
        'enable_realtime_visualization': False,
        'enable_gait_analysis': True,
        
        # MediaPipe settings
        'mediapipe_output_dir': 'mediapipe_output',
        'fps': 30.0,
        'model_complexity': 1,  # 0, 1, or 2
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        
        # Data preprocessing settings
        'confidence_threshold': 0.3,
        'filter_cutoff': 6.0,
        'filter_order': 4,
        'window_size': 30,  # ~1 second at 30fps
        'overlap': 0.5,
        
        # Model settings
        'task_type': 'phase_detection',  # or 'event_detection'
        'num_classes': 4,  # For phase detection
        'num_filters': 64,
        'kernel_size': 3,
        'num_blocks': 4,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        
        # Training settings
        'n_folds': 5,
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping_patience': 15,
        
        # Output settings
        'results_dir': 'gait_analysis_results'
    }

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Complete Gait Analysis Pipeline')
    parser.add_argument('--videos', '-v', required=True, nargs='+', 
                       help='Video files to process')
    parser.add_argument('--labels', '-l', nargs='+', type=int,
                       help='Labels for each video (optional)')
    parser.add_argument('--config', '-c', help='Configuration file (JSON)')
    parser.add_argument('--task', choices=['phase_detection', 'event_detection'],
                       default='phase_detection', help='Analysis task type')
    parser.add_argument('--output', '-o', default='gait_analysis_results',
                       help='Output directory for results')
    
    # Feature toggle arguments
    parser.add_argument('--pose-detection-only', action='store_true',
                       help='Run only pose detection without gait analysis')
    parser.add_argument('--with-visualization', action='store_true',
                       help='Run pose detection with real-time visualization')
    parser.add_argument('--enable-visualization', action='store_true',
                       help='Enable real-time visualization feature')
    parser.add_argument('--enable-gait-analysis', action='store_true', default=True,
                       help='Enable gait analysis feature')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Update config with command line arguments
    config['task_type'] = args.task
    config['results_dir'] = args.output
    
    # Handle feature toggles (pose detection is always enabled)
    if args.pose_detection_only:
        config['enable_realtime_visualization'] = False
        config['enable_gait_analysis'] = False
    elif args.with_visualization:
        config['enable_realtime_visualization'] = True
        config['enable_gait_analysis'] = False
    else:
        # Use individual feature toggles
        config['enable_realtime_visualization'] = args.enable_visualization
        config['enable_gait_analysis'] = args.enable_gait_analysis
    
    # Validate inputs
    if not all(os.path.exists(video) for video in args.videos):
        logger.error("One or more video files not found")
        return
    
    if args.labels and len(args.labels) != len(args.videos):
        logger.error("Number of labels must match number of videos")
        return
    
    # Initialize and run pipeline
    pipeline = GaitAnalysisPipeline(config)
    
    try:
        results = pipeline.run_complete_pipeline(args.videos, args.labels)
        
        # Print summary
        print("\n" + "="*50)
        print("GAIT ANALYSIS PIPELINE COMPLETED")
        print("="*50)
        print(f"Processed {len(args.videos)} videos")
        print(f"Task: {args.task}")
        print(f"Results saved to: {args.output}")
        
        if results:
            metrics = results['overall_metrics']
            print(f"\nPerformance Summary:")
            print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
        
        print("\nCheck the results directory for detailed analysis and visualizations.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\nError: {e}")
        print("Check the log file for detailed error information.")

if __name__ == "__main__":
    main()
