"""Canonical feature names, ordering, and group classifications (PRD Section 7.2 & 10.3)."""

from typing import Dict, List

# Core feature list (must remain strictly ordered across training, export, and inference)
FEATURE_NAMES: List[str] = [
    # Eye openness
    "ear_mean",
    "ear_std",
    "ear_min",
    "perclos",
    # Blinks
    "blink_rate",
    "blink_dur_mean",
    "blink_dur_std",
    "blink_count",
    # Gaze
    "gaze_x_std",
    "gaze_y_std",
    "gaze_speed_mean",
    "gaze_speed_p95",
    "saccade_rate",
    "fixation_ratio",
    "offscreen_ratio",
    # Head pose
    "yaw_std",
    "pitch_std",
    "roll_std",
    "head_speed_mean",
    "head_motion_energy",
    # Facial tension (S4 bio-behavioral stretch features)
    "brow_furrow_mean",
    "brow_furrow_std",
    "mouth_tension_mean",
    "mouth_tension_std",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "eye": [
        "ear_mean",
        "ear_std",
        "ear_min",
        "perclos",
        "blink_rate",
        "blink_dur_mean",
        "blink_dur_std",
        "blink_count",
    ],
    "gaze": [
        "gaze_x_std",
        "gaze_y_std",
        "gaze_speed_mean",
        "gaze_speed_p95",
        "saccade_rate",
        "fixation_ratio",
        "offscreen_ratio",
    ],
    "head": [
        "yaw_std",
        "pitch_std",
        "roll_std",
        "head_speed_mean",
        "head_motion_energy",
    ],
    "tension": [
        "brow_furrow_mean",
        "brow_furrow_std",
        "mouth_tension_mean",
        "mouth_tension_std",
    ],
}
