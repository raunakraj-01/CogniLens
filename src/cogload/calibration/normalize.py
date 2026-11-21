"""Feature normalization and signal re-centering (PRD Section 8.3)."""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from cogload.calibration.profile import CalibrationProfile
from cogload.features.names import FEATURE_NAMES


def recenter_signals(
    signals_df: pd.DataFrame, profile: Optional[CalibrationProfile]
) -> pd.DataFrame:
    """Re-center gaze and head-pose signals by baseline neutral offsets before feature extraction."""
    if profile is None or not profile.is_valid() or signals_df.empty:
        return signals_df

    out = signals_df.copy()
    # Gaze center correction
    gc = profile.gaze_center
    if "gaze_x" in out.columns and "x" in gc:
        out["gaze_x"] = out["gaze_x"] - gc["x"]
    if "gaze_y" in out.columns and "y" in gc:
        out["gaze_y"] = out["gaze_y"] - gc["y"]

    # Head neutral correction
    hn = profile.head_neutral
    if "yaw" in out.columns and "yaw" in hn:
        out["yaw"] = out["yaw"] - hn["yaw"]
    if "pitch" in out.columns and "pitch" in hn:
        out["pitch"] = out["pitch"] - hn["pitch"]
    if "roll" in out.columns and "roll" in hn:
        out["roll"] = out["roll"] - hn["roll"]

    return out


DEFAULT_FEATURE_SIGMA_FLOORS: Dict[str, float] = {
    "ear_mean": 0.01,
    "ear_std": 0.005,
    "ear_min": 0.01,
    "perclos": 0.01,
    "blink_rate": 0.5,
    "blink_dur_mean": 10.0,
    "blink_dur_std": 5.0,
    "blink_count": 0.5,
    "gaze_x_std": 0.005,
    "gaze_y_std": 0.005,
    "gaze_speed_mean": 0.02,
    "gaze_speed_p95": 0.05,
    "saccade_rate": 0.05,
    "fixation_ratio": 0.05,
    "offscreen_ratio": 0.01,
    "yaw_std": 0.2,
    "pitch_std": 0.2,
    "roll_std": 0.1,
    "head_speed_mean": 0.5,
    "head_motion_energy": 2.0,
    "brow_furrow_mean": 0.02,
    "brow_furrow_std": 0.01,
    "mouth_tension_mean": 0.15,
    "mouth_tension_std": 0.05,
}


def get_default_population_profile() -> CalibrationProfile:
    """Provides a realistic population baseline when the user has not run personal calibration."""
    return CalibrationProfile(
        participant_id="POPULATION_DEFAULT",
        created_at="2026-01-01T00:00:00Z",
        baseline_open_ear=0.28,
        gaze_center={"x": 0.0, "y": 0.0},
        head_neutral={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        features={
            "ear_mean": {"mu": 0.28, "sigma": 0.025},
            "ear_std": {"mu": 0.02, "sigma": 0.012},
            "ear_min": {"mu": 0.10, "sigma": 0.04},
            "perclos": {"mu": 0.04, "sigma": 0.03},
            "blink_rate": {"mu": 18.0, "sigma": 4.0},
            "blink_dur_mean": {"mu": 180.0, "sigma": 35.0},
            "blink_dur_std": {"mu": 35.0, "sigma": 15.0},
            "blink_count": {"mu": 3.0, "sigma": 1.0},
            "gaze_x_std": {"mu": 0.025, "sigma": 0.015},
            "gaze_y_std": {"mu": 0.025, "sigma": 0.015},
            "gaze_speed_mean": {"mu": 0.10, "sigma": 0.05},
            "gaze_speed_p95": {"mu": 0.30, "sigma": 0.10},
            "saccade_rate": {"mu": 0.25, "sigma": 0.15},
            "fixation_ratio": {"mu": 0.85, "sigma": 0.08},
            "offscreen_ratio": {"mu": 0.02, "sigma": 0.03},
            "yaw_std": {"mu": 2.0, "sigma": 1.0},
            "pitch_std": {"mu": 2.0, "sigma": 1.0},
            "roll_std": {"mu": 1.2, "sigma": 0.8},
            "head_speed_mean": {"mu": 4.0, "sigma": 2.5},
            "head_motion_energy": {"mu": 15.0, "sigma": 10.0},
            "brow_furrow_mean": {"mu": 0.15, "sigma": 0.06},
            "brow_furrow_std": {"mu": 0.03, "sigma": 0.02},
            "mouth_tension_mean": {"mu": 3.5, "sigma": 0.50},
            "mouth_tension_std": {"mu": 0.25, "sigma": 0.15},
        },
        quality={"valid_ratio": 1.0, "n_windows": 10, "passed": True},
    )


def normalize_features(
    features: Dict[str, float],
    profile: Optional[CalibrationProfile],
    scheme: str = "zbase",
    sigma_floors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Normalize feature vector using calibration profile under specified scheme.

    Schemes (PRD Section 8.3):
    - 'raw': returns unnormalized copy.
    - 'zbase': z = (f - mu_base) / max(sigma_base, sigma_floor)
    - 'ratio': f / max(mu_base, eps) for strictly non-negative features.
    """
    if scheme == "raw":
        return dict(features)

    # If uncalibrated, use default population baseline rather than passing unscaled raw values
    if profile is None or not profile.is_valid():
        profile = get_default_population_profile()

    if sigma_floors is None:
        sigma_floors = {}

    out: Dict[str, float] = {}

    for name in FEATURE_NAMES:
        f_val = features.get(name, 0.0)
        stat = profile.features.get(name, {"mu": 0.0, "sigma": 1.0})
        mu_base = stat["mu"]
        sigma_base = stat["sigma"]
        sigma_floor = sigma_floors.get(name, DEFAULT_FEATURE_SIGMA_FLOORS.get(name, 0.05))

        if scheme == "zbase":
            denom = max(sigma_base, sigma_floor)
            z = (f_val - mu_base) / denom
            # Clamp extreme outliers to [-4.0, 4.0]
            z = float(np.clip(z, -4.0, 4.0))
            out[name] = z
        elif scheme == "ratio":
            # Ratio normalization for positive features
            denom = max(abs(mu_base), 1e-3)
            r = f_val / denom
            out[name] = float(np.clip(r, 0.0, 10.0))
        else:
            out[name] = f_val

    return out
