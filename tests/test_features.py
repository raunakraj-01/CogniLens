"""Unit tests for blink detection and windowed feature extraction."""

import numpy as np
import pandas as pd
from cogload.features.blinks import detect_blinks, compute_perclos
from cogload.features.window import compute_window_features
from cogload.config import FeaturesConfig
from cogload.features.names import FEATURE_NAMES


def create_synthetic_blink_signal(fps: float, n_blinks: int = 3, blink_duration_ms: float = 200.0):
    """Generate a clean synthetic EAR time-series with known blinks."""
    total_sec = 10.0
    n_frames = int(total_sec * fps)
    t = np.linspace(0, total_sec, n_frames, endpoint=False)
    ear = np.ones(n_frames) * 0.30  # Baseline open EAR
    valid = np.ones(n_frames, dtype=bool)

    # Plant blinks separated by >= 1.5 seconds
    dur_sec = blink_duration_ms / 1000.0
    for i in range(n_blinks):
        blink_start_t = 1.0 + i * 2.5
        blink_end_t = blink_start_t + dur_sec
        blink_mask = (t >= blink_start_t) & (t <= blink_end_t)
        ear[blink_mask] = 0.08  # Closed EAR

    return t, ear, valid


def test_blink_detection_multiple_framerates():
    """Verify blink count and duration recovery at 15, 24, 30, and 60 FPS."""
    for fps in [15.0, 24.0, 30.0, 60.0]:
        t, ear, valid = create_synthetic_blink_signal(fps=fps, n_blinks=3, blink_duration_ms=200.0)
        blinks = detect_blinks(
            timestamps=t,
            ear_series=ear,
            valid_mask=valid,
            baseline_open_ear=0.30,
            blink_ratio=0.75,
            blink_min_ms=60.0,
            blink_max_ms=600.0,
        )
        assert len(blinks) == 3, f"Failed blink count at FPS={fps}: got {len(blinks)}"
        for b in blinks:
            # Duration recovered within ±1.5 frames
            frame_period_ms = (1.0 / fps) * 1000.0
            assert abs(b.duration_ms - 200.0) <= frame_period_ms * 1.5


def test_window_feature_filter_valid_ratio():
    """Verify window filter drops windows with < 80% valid frames (FR-FEAT-3)."""
    cfg = FeaturesConfig(min_valid_ratio=0.8)
    n_frames = 300
    t = np.linspace(0, 10, n_frames)

    # Scenario 1: 70% valid -> must return None
    valid_70 = np.array([True if i < 210 else False for i in range(n_frames)])
    df_low = pd.DataFrame({
        "t": t,
        "valid": valid_70,
        "ear": np.ones(n_frames) * 0.3,
        "gaze_x": np.zeros(n_frames),
        "gaze_y": np.zeros(n_frames),
        "yaw": np.zeros(n_frames),
        "pitch": np.zeros(n_frames),
        "roll": np.zeros(n_frames),
        "brow_furrow": np.zeros(n_frames),
        "mouth_tension": np.zeros(n_frames),
    })
    res_low = compute_window_features(df_low, baseline_open_ear=0.3, cfg=cfg)
    assert res_low is None

    # Scenario 2: 95% valid -> must return full feature dictionary
    valid_95 = np.array([True if i < 285 else False for i in range(n_frames)])
    df_high = pd.DataFrame({
        "t": t,
        "valid": valid_95,
        "ear": np.ones(n_frames) * 0.3,
        "gaze_x": np.zeros(n_frames),
        "gaze_y": np.zeros(n_frames),
        "yaw": np.zeros(n_frames),
        "pitch": np.zeros(n_frames),
        "roll": np.zeros(n_frames),
        "brow_furrow": np.zeros(n_frames),
        "mouth_tension": np.zeros(n_frames),
    })
    res_high = compute_window_features(df_high, baseline_open_ear=0.3, cfg=cfg)
    assert res_high is not None
    for name in FEATURE_NAMES:
        assert name in res_high, f"Feature {name} missing from compute_window_features"
