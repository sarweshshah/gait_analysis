"""
Configuration Management
=======================

This module provides configuration management utilities for the gait analysis project.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration for different use cases."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_name: Name of the configuration file (without .json extension)
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_name: Name of the configuration file (without .json extension)
            config: Configuration dictionary to save
        """
        config_path = self.config_dir / f"{config_name}.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "mediapipe": {
                "model_complexity": 1,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "fps": 30.0
            },
            "preprocessing": {
                "confidence_threshold": 0.3,
                "filter_cutoff": 6.0,
                "filter_order": 4,
                "window_size": 30,
                "overlap": 0.5
            },
            "training": {
                "num_classes": 4,
                "num_filters": 64,
                "kernel_size": 3,
                "dilation_rate": 2,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            },
            "paths": {
                "data_dir": "data",
                "videos_dir": "videos",
                "results_dir": "outputs/gait_analysis",
                "models_dir": "data/models"
            }
        }

def create_default_configs():
    """Create default configuration files."""
    config_manager = ConfigManager()
    
    # Create default config
    default_config = config_manager.get_default_config()
    config_manager.save_config("default", default_config)
    
    # Create gait analysis specific config
    gait_config = default_config.copy()
    gait_config["use_case"] = "gait_analysis"
    config_manager.save_config("gait_analysis", gait_config)

if __name__ == "__main__":
    create_default_configs()
