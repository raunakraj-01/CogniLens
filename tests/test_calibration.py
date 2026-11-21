"""Unit tests for calibration protocol, profile IO, and normalization."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from cogload.calibration.profile import CalibrationProfile
from cogload.calibration.protocol import run_calibration_from_signals
from cogload.calibration.normalize import normalize_features, recenter_signals
from cogload.config import CalibrationConfig, FeaturesConfig
from cogload.features.names import FEATURE_NAMES


def test_calibration_profile_serialization(tmp_path: Path):
    profile = CalibrationProfile(
        participant_id="P001",
        created_at="2026-11-21T00:00:00Z",
        baseline_open_ear=0.29,
        gaze_center={"x": 0.02, "y": -0.01},
        head_neutral={"yaw": 1.2, "pitch": -2.5, "roll": 0.3},
        features={"blink_rate": {"mu": 15.0, "sigma": 3.0}},
        quality={"valid_ratio": 0.95, "n_windows": 8, "passed": True},
    )
    p_file = tmp_path / "calibration.json"
    profile.save_json(p_file)

    loaded = CalibrationProfile.load_json(p_file)
    assert loaded.participant_id == "P001"
    assert loaded.baseline_open_ear == 0.29
    assert loaded.is_valid() is True
    assert loaded.features["blink_rate"]["mu"] == 15.0


def test_calibration_quality_rejection_low_validity():
    # 50% valid data (face lost half the time)
    n = 600
    df = pd.DataFrame({
        "t": np.linspace(0, 30, n),
        "valid": [True if i % 2 == 0 else False for i in range(n)],
        "ear": np.ones(n) * 0.30,
        "gaze_x": np.zeros(n),
        "gaze_y": np.zeros(n),
        "yaw": np.zeros(n),
        "pitch": np.zeros(n),
        "roll": np.zeros(n),
    })
    calib_cfg = CalibrationConfig(min_valid_ratio=0.85)
    prof, msg = run_calibration_from_signals(df, "P002", calib_cfg=calib_cfg)
    assert prof is None
    assert "Calibration rejected" in msg


def test_normalization_schemes():
    profile = CalibrationProfile(
        participant_id="P003",
        created_at="2026-11-21T00:00:00Z",
        baseline_open_ear=0.30,
        gaze_center={"x": 0.0, "y": 0.0},
        head_neutral={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        features={
            "blink_rate": {"mu": 10.0, "sigma": 2.0},
            "ear_mean": {"mu": 0.30, "sigma": 0.02},
        },
        quality={"valid_ratio": 0.95, "n_windows": 5, "passed": True},
    )

    sample_feats = {
        "blink_rate": 14.0,  # +2 sigma
        "ear_mean": 0.28,    # -1 sigma
    }

    # Raw
    norm_raw = normalize_features(sample_feats, profile, scheme="raw")
    assert norm_raw["blink_rate"] == 14.0

    # Zbase
    norm_z = normalize_features(sample_feats, profile, scheme="zbase")
    assert abs(norm_z["blink_rate"] - 2.0) < 1e-4
    assert abs(norm_z["ear_mean"] - (-1.0)) < 1e-4

    # Ratio
    norm_ratio = normalize_features(sample_feats, profile, scheme="ratio")
    assert abs(norm_ratio["blink_rate"] - 1.4) < 1e-4
