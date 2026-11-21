"""Windowed feature extraction pipeline (PRD Section 7.2).

The exact same function serves both online (real-time stream) and offline (dataset generation).
"""

from typing import Optional, Dict
import numpy as np
import pandas as pd
from cogload.config import FeaturesConfig
from cogload.features.blinks import detect_blinks, compute_perclos
from cogload.features.names import FEATURE_NAMES


def compute_window_features(
    signal_df: pd.DataFrame,
    baseline_open_ear: float = 0.28,
    cfg: Optional[FeaturesConfig] = None,
) -> Optional[Dict[str, float]]:
    """Compute window features over a sliding window DataFrame.

    Returns None if valid_ratio < cfg.min_valid_ratio.
    """
    if cfg is None:
        cfg = FeaturesConfig()

    if signal_df.empty or len(signal_df) < 5:
        return None

    total_frames = len(signal_df)
    valid_mask = signal_df["valid"].to_numpy(dtype=bool, copy=True)
    valid_count = int(np.sum(valid_mask))
    valid_ratio = float(valid_count / total_frames)

    # Discard window if valid frame ratio is below minimum threshold (FR-FEAT-3)
    if valid_ratio < cfg.min_valid_ratio:
        return None

    # Timestamps
    t = signal_df["t"].to_numpy(dtype=np.float64)
    dt = float(t[-1] - t[0]) if len(t) > 1 else cfg.window_sec
    window_min = max(0.01, dt / 60.0)

    # Valid EAR values
    ear_all = signal_df["ear"].to_numpy(dtype=np.float64)
    ear_valid = ear_all[valid_mask]
    ear_mean = float(np.mean(ear_valid))
    ear_std = float(np.std(ear_valid)) if len(ear_valid) > 1 else 0.0
    ear_min = float(np.min(ear_valid))

    # Eye closure and blinks
    perclos = compute_perclos(
        ear_all, valid_mask, baseline_open_ear=baseline_open_ear, closure_ratio=cfg.closure_ratio
    )
    blinks = detect_blinks(
        timestamps=t,
        ear_series=ear_all,
        valid_mask=valid_mask,
        baseline_open_ear=baseline_open_ear,
        blink_ratio=cfg.blink_ratio,
        blink_min_ms=cfg.blink_min_ms,
        blink_max_ms=cfg.blink_max_ms,
        blink_refractory_ms=cfg.blink_refractory_ms,
    )
    blink_count = len(blinks)
    blink_rate = float(blink_count / window_min)
    if blink_count > 0:
        durations = [b.duration_ms for b in blinks]
        blink_dur_mean = float(np.mean(durations))
        blink_dur_std = float(np.std(durations)) if len(durations) > 1 else 0.0
    else:
        blink_dur_mean = 0.0
        blink_dur_std = 0.0

    # Gaze features
    gx = signal_df["gaze_x"].to_numpy(dtype=np.float64)
    gy = signal_df["gaze_y"].to_numpy(dtype=np.float64)
    gx_valid = gx[valid_mask]
    gy_valid = gy[valid_mask]

    gaze_x_std = float(np.std(gx_valid)) if len(gx_valid) > 1 else 0.0
    gaze_y_std = float(np.std(gy_valid)) if len(gy_valid) > 1 else 0.0

    # Gaze velocities with Gaussian smoothing to eliminate single-pixel iris jitter
    if len(gx) >= 5:
        from scipy.ndimage import gaussian_filter1d
        gx_smooth = gaussian_filter1d(gx, sigma=1.5)
        gy_smooth = gaussian_filter1d(gy, sigma=1.5)
    else:
        gx_smooth, gy_smooth = gx, gy

    dt_arr = np.diff(t)
    dt_arr[dt_arr <= 1e-4] = 1e-3  # Avoid divide-by-zero
    vx = np.diff(gx_smooth) / dt_arr
    vy = np.diff(gy_smooth) / dt_arr
    gaze_speeds = np.sqrt(vx**2 + vy**2)
    # Filter out speeds crossing invalid transitions
    valid_transitions = valid_mask[:-1] & valid_mask[1:]
    valid_speeds = gaze_speeds[valid_transitions] if np.any(valid_transitions) else np.array([0.0])

    gaze_speed_mean = float(np.mean(valid_speeds))
    gaze_speed_p95 = float(np.percentile(valid_speeds, 95)) if len(valid_speeds) > 0 else 0.0

    # Saccades and fixations
    saccades = np.sum(valid_speeds > cfg.saccade_speed_thresh)
    saccade_rate = float(saccades / dt) if dt > 0 else 0.0
    fixations = np.sum(valid_speeds < cfg.fixation_speed_thresh)
    fixation_ratio = float(fixations / len(valid_speeds)) if len(valid_speeds) > 0 else 1.0

    # Offscreen gaze (looking far away)
    dist_from_center = np.sqrt(gx_valid**2 + gy_valid**2)
    offscreen_ratio = (
        float(np.mean(dist_from_center > cfg.offscreen_gaze_bound))
        if len(dist_from_center) > 0
        else 0.0
    )

    # Head pose dynamics with median filter on angles to prevent derivative blow-ups from landmark jitter
    yaw_raw = signal_df["yaw"].to_numpy(dtype=np.float64)
    pitch_raw = signal_df["pitch"].to_numpy(dtype=np.float64)
    roll_raw = signal_df["roll"].to_numpy(dtype=np.float64)

    # Rolling median smoothing on head pose to filter sensor noise and rapid mastication tremor
    if len(yaw_raw) >= 15:
        from scipy.ndimage import median_filter, gaussian_filter1d
        yaw_s = gaussian_filter1d(median_filter(yaw_raw, size=15), sigma=2.0)
        pitch_s = gaussian_filter1d(median_filter(pitch_raw, size=15), sigma=2.0)
        roll_s = gaussian_filter1d(median_filter(roll_raw, size=15), sigma=2.0)
    elif len(yaw_raw) >= 5:
        from scipy.ndimage import median_filter, gaussian_filter1d
        yaw_s = gaussian_filter1d(median_filter(yaw_raw, size=3), sigma=1.0)
        pitch_s = gaussian_filter1d(median_filter(pitch_raw, size=3), sigma=1.0)
        roll_s = gaussian_filter1d(median_filter(roll_raw, size=3), sigma=1.0)
    else:
        yaw_s, pitch_s, roll_s = yaw_raw, pitch_raw, roll_raw

    yaw = yaw_s[valid_mask]
    pitch = pitch_s[valid_mask]
    roll = roll_s[valid_mask]

    yaw_std = float(np.std(yaw)) if len(yaw) > 1 else 0.0
    pitch_std = float(np.std(pitch)) if len(pitch) > 1 else 0.0
    roll_std = float(np.std(roll)) if len(roll) > 1 else 0.0

    # Head angular velocities
    dyaw = np.diff(yaw_s) / dt_arr
    dpitch = np.diff(pitch_s) / dt_arr
    droll = np.diff(roll_s) / dt_arr
    head_speeds = np.sqrt(dyaw**2 + dpitch**2 + droll**2)
    valid_head_speeds = (
        head_speeds[valid_transitions] if np.any(valid_transitions) else np.array([0.0])
    )
    head_speed_mean = (
        float(np.mean(valid_head_speeds))
        if len(valid_head_speeds) > 0 and not np.isnan(np.mean(valid_head_speeds))
        else 0.0
    )
    head_motion_energy = (
        float(np.sum(valid_head_speeds**2) / dt)
        if dt > 0 and not np.isnan(np.sum(valid_head_speeds**2))
        else 0.0
    )

    # Facial tension (S4)
    brow = (
        signal_df["brow_furrow"].to_numpy(dtype=np.float64)[valid_mask]
        if "brow_furrow" in signal_df.columns
        else np.zeros(valid_count)
    )
    mouth = (
        signal_df["mouth_tension"].to_numpy(dtype=np.float64)[valid_mask]
        if "mouth_tension" in signal_df.columns
        else np.zeros(valid_count)
    )

    brow_furrow_mean = float(np.mean(brow)) if len(brow) > 0 else 0.0
    brow_furrow_std = float(np.std(brow)) if len(brow) > 1 else 0.0
    mouth_tension_mean = float(np.mean(mouth)) if len(mouth) > 0 else 0.0
    mouth_tension_std = float(np.std(mouth)) if len(mouth) > 1 else 0.0

    feat_dict: Dict[str, float] = {
        "ear_mean": ear_mean,
        "ear_std": ear_std,
        "ear_min": ear_min,
        "perclos": perclos,
        "blink_rate": blink_rate,
        "blink_dur_mean": blink_dur_mean,
        "blink_dur_std": blink_dur_std,
        "blink_count": float(blink_count),
        "gaze_x_std": gaze_x_std,
        "gaze_y_std": gaze_y_std,
        "gaze_speed_mean": gaze_speed_mean,
        "gaze_speed_p95": gaze_speed_p95,
        "saccade_rate": saccade_rate,
        "fixation_ratio": fixation_ratio,
        "offscreen_ratio": offscreen_ratio,
        "yaw_std": yaw_std,
        "pitch_std": pitch_std,
        "roll_std": roll_std,
        "head_speed_mean": head_speed_mean,
        "head_motion_energy": head_motion_energy,
        "brow_furrow_mean": brow_furrow_mean,
        "brow_furrow_std": brow_furrow_std,
        "mouth_tension_mean": mouth_tension_mean,
        "mouth_tension_std": mouth_tension_std,
        "valid_ratio": valid_ratio,
    }

    return feat_dict
