"""
Gait Analysis Use Case Package
=============================

This package contains the main gait analysis functionality.
"""

from .main_gait_analysis import GaitAnalysisPipeline, create_default_config
from .utils import run_gait_events_on_json_dir

__all__ = [
    'GaitAnalysisPipeline',
    'create_default_config',
    'run_gait_events_on_json_dir'
]
