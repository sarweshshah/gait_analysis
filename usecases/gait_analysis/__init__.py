"""
Gait Analysis Use Case Package
=============================

This package contains the main gait analysis functionality.
"""

from .main_gait_analysis import GaitAnalysisPipeline, create_default_config

__all__ = [
    'GaitAnalysisPipeline',
    'create_default_config'
]
