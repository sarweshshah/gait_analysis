"""
Performance Optimization Module for Gait Analysis
================================================

This module provides utilities and optimizations for improving the performance
of the gait analysis pipeline, including I/O optimization, memory management,
and computational optimizations.

Author: Gait Analysis System
"""

import os
import json
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import logging
from pathlib import Path
from functools import wraps
import psutil
import gc

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)
            del self.start_times[name]
            return duration
        return 0.0
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation."""
        if name in self.metrics and self.metrics[name]:
            return np.mean(self.metrics[name])
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

def performance_timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class IOOptimizer:
    """Optimize I/O operations for better performance."""
    
    def __init__(self, batch_size: int = 100, use_compression: bool = True):
        """
        Initialize I/O optimizer.
        
        Args:
            batch_size: Number of frames to batch together
            use_compression: Whether to use compression for JSON files
        """
        self.batch_size = batch_size
        self.use_compression = use_compression
        self.buffer = []
        self.lock = threading.Lock()
    
    def add_frame_data(self, frame_data: Dict, output_dir: str, video_name: str):
        """Add frame data to buffer and flush if needed."""
        with self.lock:
            self.buffer.append(frame_data)
            
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer(output_dir, video_name)
    
    def _flush_buffer(self, output_dir: str, video_name: str):
        """Flush buffered data to disk."""
        if not self.buffer:
            return
        
        # Create batch file
        timestamp = int(time.time())
        batch_filename = f"{video_name}_batch_{timestamp}.json"
        batch_path = os.path.join(output_dir, batch_filename)
        
        batch_data = {
            "video_name": video_name,
            "timestamp": timestamp,
            "frame_count": len(self.buffer),
            "frames": self.buffer
        }
        
        # Write batch file
        with open(batch_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        logger.info(f"Flushed {len(self.buffer)} frames to {batch_filename}")
        self.buffer.clear()
    
    def flush_remaining(self, output_dir: str, video_name: str):
        """Flush any remaining buffered data."""
        if self.buffer:
            self._flush_buffer(output_dir, video_name)

class MemoryOptimizer:
    """Optimize memory usage and management."""
    
    def __init__(self, max_memory_mb: int = 2048):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.monitor = PerformanceMonitor()
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        memory_info = self.monitor.get_memory_usage()
        return memory_info['rss_mb'] < self.max_memory_mb
    
    def optimize_memory(self):
        """Perform memory optimization."""
        # Force garbage collection
        gc.collect()
        
        # Clear numpy cache
        if hasattr(np, 'clear_cache'):
            np.clear_cache()
    
    def process_in_chunks(self, data: List, chunk_size: int, process_func):
        """Process data in chunks to manage memory."""
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = process_func(chunk)
            results.extend(chunk_result)
            
            # Check memory and optimize if needed
            if not self.check_memory_usage():
                self.optimize_memory()
        
        return results

class ParallelProcessor:
    """Handle parallel processing for CPU-intensive tasks."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes/threads
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    def process_frames_parallel(self, frames: List[np.ndarray], process_func) -> List:
        """Process frames in parallel using thread pool."""
        futures = []
        
        for frame in frames:
            future = self.thread_pool.submit(process_func, frame)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Frame processing failed: {e}")
                results.append(None)
        
        return results
    
    def process_videos_parallel(self, video_paths: List[str], process_func) -> List:
        """Process videos in parallel using process pool."""
        futures = []
        
        for video_path in video_paths:
            future = self.process_pool.submit(process_func, video_path)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Video processing failed: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        """Shutdown thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class MediaPipeOptimizer:
    """Optimize MediaPipe configuration for better performance."""
    
    @staticmethod
    def get_optimized_config(performance_mode: str = 'balanced') -> Dict[str, Any]:
        """
        Get optimized MediaPipe configuration.
        
        Args:
            performance_mode: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Optimized configuration dictionary
        """
        configs = {
            'fast': {
                'static_image_mode': False,
                'model_complexity': 0,
                'smooth_landmarks': False,
                'enable_segmentation': False,
                'smooth_segmentation': False,
                'min_detection_confidence': 0.3,
                'min_tracking_confidence': 0.3
            },
            'balanced': {
                'static_image_mode': False,
                'model_complexity': 1,
                'smooth_landmarks': True,
                'enable_segmentation': False,
                'smooth_segmentation': False,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            'accurate': {
                'static_image_mode': False,
                'model_complexity': 2,
                'smooth_landmarks': True,
                'enable_segmentation': True,
                'smooth_segmentation': True,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.7
            }
        }
        
        return configs.get(performance_mode, configs['balanced'])
    
    @staticmethod
    def optimize_frame_size(frame: np.ndarray, target_width: int = 640) -> np.ndarray:
        """
        Optimize frame size for better performance.
        
        Args:
            frame: Input frame
            target_width: Target width for resizing
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        if width > target_width:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame

class ModelOptimizer:
    """Optimize model performance and memory usage."""
    
    @staticmethod
    def optimize_tcn_model(model_config: Dict[str, Any], performance_mode: str = 'balanced') -> Dict[str, Any]:
        """
        Optimize TCN model configuration for better performance.
        
        Args:
            model_config: Original model configuration
            performance_mode: Performance mode ('fast', 'balanced', 'accurate')
            
        Returns:
            Optimized model configuration
        """
        optimizations = {
            'fast': {
                'num_filters': 32,
                'num_blocks': 2,
                'kernel_size': 3,
                'dropout_rate': 0.1,
                'batch_size': 64
            },
            'balanced': {
                'num_filters': 64,
                'num_blocks': 4,
                'kernel_size': 3,
                'dropout_rate': 0.2,
                'batch_size': 32
            },
            'accurate': {
                'num_filters': 128,
                'num_blocks': 6,
                'kernel_size': 5,
                'dropout_rate': 0.3,
                'batch_size': 16
            }
        }
        
        optimized_config = model_config.copy()
        optimization = optimizations.get(performance_mode, optimizations['balanced'])
        
        for key, value in optimization.items():
            if key in optimized_config:
                optimized_config[key] = value
        
        return optimized_config
    
    @staticmethod
    def enable_mixed_precision(model):
        """Enable mixed precision training for better performance."""
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled")
        except ImportError:
            logger.warning("Mixed precision not available")
    
    @staticmethod
    def optimize_training_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize training configuration for better performance.
        
        Args:
            training_config: Original training configuration
            
        Returns:
            Optimized training configuration
        """
        optimized = training_config.copy()
        
        # Optimize batch size based on available memory
        memory_mb = psutil.virtual_memory().total / 1024 / 1024
        if memory_mb < 4096:  # Less than 4GB
            optimized['batch_size'] = min(16, optimized.get('batch_size', 32))
        elif memory_mb < 8192:  # Less than 8GB
            optimized['batch_size'] = min(32, optimized.get('batch_size', 32))
        else:
            optimized['batch_size'] = min(64, optimized.get('batch_size', 32))
        
        # Optimize number of workers for data loading
        optimized['workers'] = min(mp.cpu_count(), 4)
        
        return optimized

class PerformanceOptimizer:
    """Main performance optimizer that coordinates all optimizations."""
    
    def __init__(self, performance_mode: str = 'balanced', max_memory_mb: int = 2048):
        """
        Initialize performance optimizer.
        
        Args:
            performance_mode: Performance mode ('fast', 'balanced', 'accurate')
            max_memory_mb: Maximum memory usage in MB
        """
        self.performance_mode = performance_mode
        self.monitor = PerformanceMonitor()
        self.io_optimizer = IOOptimizer()
        self.memory_optimizer = MemoryOptimizer(max_memory_mb)
        self.parallel_processor = ParallelProcessor()
        
        logger.info(f"Initialized PerformanceOptimizer in {performance_mode} mode")
    
    def get_optimized_mediapipe_config(self) -> Dict[str, Any]:
        """Get optimized MediaPipe configuration."""
        return MediaPipeOptimizer.get_optimized_config(self.performance_mode)
    
    def get_optimized_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized model configuration."""
        return ModelOptimizer.optimize_tcn_model(model_config, self.performance_mode)
    
    def get_optimized_training_config(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized training configuration."""
        return ModelOptimizer.optimize_training_config(training_config)
    
    def optimize_frame_processing(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame for processing."""
        # Resize frame for better performance
        frame = MediaPipeOptimizer.optimize_frame_size(frame)
        return frame
    
    def process_video_optimized(self, video_path: str, process_func) -> List:
        """Process video with optimizations."""
        self.monitor.start_timer('video_processing')
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Optimize frame
                frame = self.optimize_frame_processing(frame)
                frames.append(frame)
                
                # Check memory usage
                if not self.memory_optimizer.check_memory_usage():
                    self.memory_optimizer.optimize_memory()
        
        finally:
            cap.release()
        
        # Process frames in parallel
        results = self.parallel_processor.process_frames_parallel(frames, process_func)
        
        self.monitor.end_timer('video_processing')
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        memory_info = self.monitor.get_memory_usage()
        
        report = {
            'performance_mode': self.performance_mode,
            'memory_usage': memory_info,
            'timing_metrics': {
                name: self.monitor.get_average_time(name)
                for name in self.monitor.metrics.keys()
            },
            'optimization_recommendations': self._get_recommendations()
        }
        
        return report
    
    def _get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        memory_info = self.monitor.get_memory_usage()
        
        if memory_info['rss_mb'] > 1024:
            recommendations.append("Consider reducing batch size or processing in smaller chunks")
        
        if memory_info['percent'] > 80:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        avg_times = {name: self.monitor.get_average_time(name) 
                    for name in self.monitor.metrics.keys()}
        
        if 'video_processing' in avg_times and avg_times['video_processing'] > 10:
            recommendations.append("Video processing is slow - consider using 'fast' performance mode")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources."""
        self.parallel_processor.shutdown()
        self.memory_optimizer.optimize_memory()
        logger.info("Performance optimizer cleanup completed")