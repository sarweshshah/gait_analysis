"""
Core-wide constants for gait analysis.
Place shared labels here so both core models and usecases agree on schemas.
"""
from typing import List, Dict

# Common sides
SIDES: List[str] = ["left", "right"]
LEFT: str = "left"
RIGHT: str = "right"

# Rule-based event types
# Basic set (commonly used)
EVENT_TYPES_BASIC: List[str] = [
    "heel_strike",
    "toe_off",
]

# Extended set (includes mid-stance markers)
EVENT_TYPES_EXTENDED: List[str] = [
    "heel_strike",  # initial contact
    "flat_foot",    # foot flat / loading response
    "heel_off",     # terminal stance initiation
    "toe_off",      # pre-swing/toe-off
]

# Default exported set (kept as basic to preserve current behavior)
EVENT_TYPES: List[str] = EVENT_TYPES_BASIC

# Individual event name constants (for direct imports)
HEEL_STRIKE: str = "heel_strike"
TOE_OFF: str = "toe_off"
FLAT_FOOT: str = "flat_foot"
HEEL_OFF: str = "heel_off"

# Support phase names (for clarity in keys/labels)
DOUBLE_SUPPORT: str = "double_support"
SINGLE_SUPPORT: str = "single_support"


# Default phase label sets
# Note: The pipeline currently uses 4 classes for phase detection.
PHASE_LABELS_4: List[str] = [
    "stance_left",
    "swing_left",
    "stance_right",
    "swing_right",
]

# Alternative, finer-grained 7-phase gait cycle (not currently used by default)
PHASE_LABELS_7: List[str] = [
    "initial_contact",
    "loading_response",
    "mid_stance",
    "terminal_stance",
    "pre_swing",
    "initial_swing",
    "terminal_swing",
]

# Task types
TASK_TYPES: List[str] = [
    "phase_detection",
    "event_detection",
]

DEFAULTS: Dict[str, int] = {
    "num_phase_classes": 4,
}


def get_phase_labels(num_classes: int = DEFAULTS["num_phase_classes"]) -> List[str]:
    """Return phase labels for the requested granularity."""
    if num_classes == 4:
        return PHASE_LABELS_4
    elif num_classes == 7:
        return PHASE_LABELS_7
    else:
        # Fallback: return indices as strings
        return [f"phase_{i}" for i in range(num_classes)]
