#!/usr/bin/env python3
"""
Performance Benchmarking Script for Gait Analysis
================================================

This script benchmarks the performance of different optimization configurations
and provides detailed metrics for comparison.

Usage:
    python scripts/benchmark_performance.py --video path/to/video.mp4 --profiles fast,balanced,accurate
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import psutil
import cv2

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.performance_optimizer import PerformanceOptimizer, PerformanceMonitor
from core.mediapipe_integration import MediaPipeProcessor
from usecases.gait_analysis.main_gait_analysis import GaitAnalysisPipeline, create_default_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Benchmark different performance configurations."""
    
    def __init__(self, video_path: str, output_dir: str = 'benchmark_results'):
        """
        Initialize the performance benchmark.
        
        Args:
            video_path: Path to the test video
            output_dir: Directory to save benchmark results
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.monitor = PerformanceMonitor()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load performance profiles
        self.performance_profiles = self._load_performance_profiles()
    
    def _load_performance_profiles(self) -> Dict[str, Any]:
        """Load performance profiles from configuration."""
        config_path = Path(__file__).parent.parent / 'configs' / 'performance_optimized.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('performance_profiles', {})
        else:
            logger.warning("Performance profiles not found, using default profiles")
            return {
                'fast': {'performance_mode': 'fast'},
                'balanced': {'performance_mode': 'balanced'},
                'accurate': {'performance_mode': 'accurate'}
            }
    
    def benchmark_mediapipe_processing(self, profile_name: str, profile_config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark MediaPipe processing with specific profile."""
        logger.info(f"Benchmarking MediaPipe processing with {profile_name} profile")
        
        # Create temporary output directory
        temp_output_dir = os.path.join(self.output_dir, f'mediapipe_{profile_name}')
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Initialize MediaPipe processor with profile settings
        mediapipe_config = profile_config.get('mediapipe', {})
        
        processor = MediaPipeProcessor(
            output_dir=temp_output_dir,
            fps=30.0,
            model_complexity=mediapipe_config.get('model_complexity', 1),
            min_detection_confidence=mediapipe_config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=mediapipe_config.get('min_tracking_confidence', 0.5),
            performance_mode=profile_config.get('performance_mode', 'balanced'),
            enable_optimizations=profile_config.get('enable_optimizations', True)
        )
        
        # Start benchmarking
        self.monitor.start_timer(f'mediapipe_{profile_name}')
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process video
        success = processor.process_video(self.video_path)
        
        # Get final memory usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # End benchmarking
        processing_time = self.monitor.end_timer(f'mediapipe_{profile_name}')
        
        # Cleanup
        processor.cleanup()
        
        # Calculate metrics
        memory_usage = final_memory - initial_memory
        fps = self._calculate_fps(processing_time)
        
        results = {
            'profile': profile_name,
            'success': success,
            'processing_time_seconds': processing_time,
            'memory_usage_mb': memory_usage,
            'estimated_fps': fps,
            'output_files': len([f for f in os.listdir(temp_output_dir) if f.endswith('.json')])
        }
        
        logger.info(f"MediaPipe {profile_name} results: {processing_time:.2f}s, {memory_usage:.1f}MB, {fps:.1f} FPS")
        return results
    
    def benchmark_full_pipeline(self, profile_name: str, profile_config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark full gait analysis pipeline with specific profile."""
        logger.info(f"Benchmarking full pipeline with {profile_name} profile")
        
        # Create configuration
        config = create_default_config()
        
        # Apply profile settings
        config.update(profile_config)
        config['results_dir'] = os.path.join(self.output_dir, f'pipeline_{profile_name}')
        
        # Initialize pipeline
        pipeline = GaitAnalysisPipeline(config)
        
        # Start benchmarking
        self.monitor.start_timer(f'pipeline_{profile_name}')
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Run pipeline (pose detection only for benchmarking)
            results = pipeline.run_pose_detection_only([self.video_path])
            
            # Get final memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # End benchmarking
            processing_time = self.monitor.end_timer(f'pipeline_{profile_name}')
            
            # Calculate metrics
            memory_usage = final_memory - initial_memory
            fps = self._calculate_fps(processing_time)
            
            pipeline_results = {
                'profile': profile_name,
                'success': True,
                'processing_time_seconds': processing_time,
                'memory_usage_mb': memory_usage,
                'estimated_fps': fps,
                'pipeline_results': results
            }
            
        except Exception as e:
            processing_time = self.monitor.end_timer(f'pipeline_{profile_name}')
            pipeline_results = {
                'profile': profile_name,
                'success': False,
                'processing_time_seconds': processing_time,
                'error': str(e)
            }
        
        logger.info(f"Pipeline {profile_name} results: {processing_time:.2f}s, {memory_usage:.1f}MB, {fps:.1f} FPS")
        return pipeline_results
    
    def _calculate_fps(self, processing_time: float) -> float:
        """Calculate estimated FPS based on processing time."""
        # Get video duration (simplified estimation)
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if processing_time > 0:
            return total_frames / processing_time
        return 0.0
    
    def run_benchmarks(self, profiles: List[str], benchmark_type: str = 'both') -> Dict[str, Any]:
        """
        Run benchmarks for specified profiles.
        
        Args:
            profiles: List of profile names to benchmark
            benchmark_type: 'mediapipe', 'pipeline', or 'both'
            
        Returns:
            Dictionary containing all benchmark results
        """
        logger.info(f"Starting benchmarks for profiles: {profiles}")
        
        results = {
            'video_path': self.video_path,
            'profiles': profiles,
            'benchmark_type': benchmark_type,
            'system_info': self._get_system_info(),
            'mediapipe_results': {},
            'pipeline_results': {}
        }
        
        for profile_name in profiles:
            if profile_name not in self.performance_profiles:
                logger.warning(f"Profile {profile_name} not found, skipping")
                continue
            
            profile_config = self.performance_profiles[profile_name]
            
            # Benchmark MediaPipe processing
            if benchmark_type in ['mediapipe', 'both']:
                mediapipe_results = self.benchmark_mediapipe_processing(profile_name, profile_config)
                results['mediapipe_results'][profile_name] = mediapipe_results
            
            # Benchmark full pipeline
            if benchmark_type in ['pipeline', 'both']:
                pipeline_results = self.benchmark_full_pipeline(profile_name, profile_config)
                results['pipeline_results'][profile_name] = pipeline_results
        
        # Save results
        self._save_benchmark_results(results)
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'platform': sys.platform,
            'python_version': sys.version
        }
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'benchmark_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable comparison report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PERFORMANCE BENCHMARK RESULTS")
        report_lines.append("=" * 60)
        report_lines.append(f"Video: {results['video_path']}")
        report_lines.append(f"Benchmark Type: {results['benchmark_type']}")
        report_lines.append("")
        
        # System information
        report_lines.append("SYSTEM INFORMATION:")
        sys_info = results['system_info']
        report_lines.append(f"  CPU Cores: {sys_info['cpu_count']}")
        report_lines.append(f"  Total Memory: {sys_info['memory_total_gb']:.1f} GB")
        report_lines.append(f"  Available Memory: {sys_info['memory_available_gb']:.1f} GB")
        report_lines.append("")
        
        # MediaPipe results comparison
        if results['mediapipe_results']:
            report_lines.append("MEDIAPIPE PROCESSING RESULTS:")
            report_lines.append("-" * 40)
            
            for profile, data in results['mediapipe_results'].items():
                if data['success']:
                    report_lines.append(f"{profile.upper():<12} | "
                                      f"Time: {data['processing_time_seconds']:>6.2f}s | "
                                      f"Memory: {data['memory_usage_mb']:>6.1f}MB | "
                                      f"FPS: {data['estimated_fps']:>6.1f}")
                else:
                    report_lines.append(f"{profile.upper():<12} | FAILED")
            report_lines.append("")
        
        # Pipeline results comparison
        if results['pipeline_results']:
            report_lines.append("FULL PIPELINE RESULTS:")
            report_lines.append("-" * 40)
            
            for profile, data in results['pipeline_results'].items():
                if data['success']:
                    report_lines.append(f"{profile.upper():<12} | "
                                      f"Time: {data['processing_time_seconds']:>6.2f}s | "
                                      f"Memory: {data['memory_usage_mb']:>6.1f}MB | "
                                      f"FPS: {data['estimated_fps']:>6.1f}")
                else:
                    report_lines.append(f"{profile.upper():<12} | FAILED - {data.get('error', 'Unknown error')}")
            report_lines.append("")
        
        # Performance recommendations
        report_lines.append("PERFORMANCE RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if results['mediapipe_results']:
            fastest_mediapipe = min(results['mediapipe_results'].values(), 
                                  key=lambda x: x['processing_time_seconds'] if x['success'] else float('inf'))
            report_lines.append(f"Fastest MediaPipe: {fastest_mediapipe['profile']} "
                              f"({fastest_mediapipe['processing_time_seconds']:.2f}s)")
        
        if results['pipeline_results']:
            fastest_pipeline = min(results['pipeline_results'].values(), 
                                 key=lambda x: x['processing_time_seconds'] if x['success'] else float('inf'))
            report_lines.append(f"Fastest Pipeline: {fastest_pipeline['profile']} "
                              f"({fastest_pipeline['processing_time_seconds']:.2f}s)")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Performance Benchmarking for Gait Analysis')
    parser.add_argument('--video', '-v', required=True, help='Path to test video file')
    parser.add_argument('--profiles', '-p', default='fast,balanced,accurate',
                       help='Comma-separated list of profiles to benchmark')
    parser.add_argument('--type', '-t', choices=['mediapipe', 'pipeline', 'both'],
                       default='both', help='Type of benchmark to run')
    parser.add_argument('--output', '-o', default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Parse profiles
    profiles = [p.strip() for p in args.profiles.split(',')]
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.video, args.output)
    
    try:
        # Run benchmarks
        results = benchmark.run_benchmarks(profiles, args.type)
        
        # Generate and print report
        report = benchmark.generate_comparison_report(results)
        print(report)
        
        # Save report to file
        report_file = os.path.join(args.output, 'benchmark_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Benchmark report saved to {report_file}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()