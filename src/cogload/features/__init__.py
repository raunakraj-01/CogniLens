"""Feature extraction modules for blink detection and windowed signal dynamics."""

from cogload.features.names import FEATURE_NAMES, FEATURE_GROUPS
from cogload.features.blinks import detect_blinks, compute_perclos, BlinkEvent
from cogload.features.window import compute_window_features

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "detect_blinks",
    "compute_perclos",
    "BlinkEvent",
    "compute_window_features",
]
