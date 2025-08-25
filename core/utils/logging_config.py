"""
Logging Configuration
====================

This module provides logging configuration utilities for the gait analysis project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if log_file and not logger.handlers:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
