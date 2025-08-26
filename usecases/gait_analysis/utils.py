"""
Gait Analysis Utilities
=======================

Lightweight helpers for ad-hoc usage outside the full pipeline.
"""
from typing import Dict

from core.gait_data_preprocessing import GaitDataPreprocessor
from .gait_events import BasicGaitEvents


def run_gait_events_on_json_dir(json_directory: str, fps: float = 30.0) -> Dict:
    """
    Run rule-based gait event detection on a directory of BODY_25-style JSON files
    (produced by either OpenPose or this repo's MediaPipe integration).

    Args:
        json_directory: Directory containing BODY_25 JSON files ("people" -> "pose_keypoints_2d")
        fps: Video frame rate

    Returns:
        Dictionary with detected events, metrics, and processing info.
    """
    preprocessor = GaitDataPreprocessor()
    processed = preprocessor.process_video_sequence(json_directory=json_directory, fps=fps)
    keypoints_sequence = processed['keypoints_sequence'][:, :, :2]  # (n_frames, 25, 2)

    detector = BasicGaitEvents(fps=fps, keypoint_format='body25')
    events = detector.detect_events(keypoints_sequence)
    metrics = detector.calculate_gait_metrics(events)

    return {
        'events': events,
        'metrics': metrics,
        'processing_info': {
            'fps': fps,
            'total_frames': processed['metadata'].get('n_frames', keypoints_sequence.shape[0]),
            'window_size': processed['metadata'].get('window_size'),
            'n_features': processed['metadata'].get('n_features')
        }
    }
