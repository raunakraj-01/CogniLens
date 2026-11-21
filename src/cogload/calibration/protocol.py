"""Guided calibration protocol implementation and quality validation (PRD Section 8.1 & 8.2)."""

from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from cogload.config import CalibrationConfig, FeaturesConfig
from cogload.features.window import compute_window_features
from cogload.features.names import FEATURE_NAMES
from cogload.calibration.profile import CalibrationProfile


def run_calibration_from_signals(
    signals_df: pd.DataFrame,
    participant_id: str = "P000",
    calib_cfg: Optional[CalibrationConfig] = None,
    feat_cfg: Optional[FeaturesConfig] = None,
) -> Tuple[Optional[CalibrationProfile], str]:
    """Execute calibration analysis from recorded calibration signals.

    Returns:
        (CalibrationProfile or None, status_message)
    """
    if calib_cfg is None:
        calib_cfg = CalibrationConfig()
    if feat_cfg is None:
        feat_cfg = FeaturesConfig()

    if signals_df.empty or len(signals_df) < 50:
        return None, "Insufficient data: calibration stream has fewer than 50 frames."

    # Frame validity assessment
    valid_mask = signals_df["valid"].to_numpy(dtype=bool)
    valid_ratio = float(np.mean(valid_mask))
    if valid_ratio < calib_cfg.min_valid_ratio:
        return None, (
            f"Calibration rejected: face tracking valid ratio ({valid_ratio:.1%}) "
            f"below required threshold ({calib_cfg.min_valid_ratio:.1%}). "
            "Please ensure face is centered, stable, and well-lit."
        )

    # Valid frames only for baseline calculations
    df_valid = signals_df[valid_mask]

    # Baseline open EAR (90th percentile of EAR outside blinks, PRD 8.1)
    ear_vals = df_valid["ear"].to_numpy(dtype=np.float64)
    baseline_open_ear = float(np.percentile(ear_vals, 90))
    if baseline_open_ear < 0.15 or baseline_open_ear > 0.50:
        return None, f"Abnormal baseline EAR ({baseline_open_ear:.3f}). Check eye visibility or glasses glare."

    # Neutral head pose (median yaw, pitch, roll)
    head_neutral = {
        "yaw": float(np.median(df_valid["yaw"])),
        "pitch": float(np.median(df_valid["pitch"])),
        "roll": float(np.median(df_valid["roll"])),
    }

    # Neutral gaze center (median gaze_x, gaze_y)
    gaze_center = {
        "x": float(np.median(df_valid["gaze_x"])),
        "y": float(np.median(df_valid["gaze_y"])),
    }

    # Head motion check (reject if excessive motion during fixation cross)
    yaw_range = float(np.ptp(df_valid["yaw"]))
    pitch_range = float(np.ptp(df_valid["pitch"]))
    if yaw_range > 35.0 or pitch_range > 35.0:
        return None, "Calibration rejected: excessive head motion detected. Please remain still while looking at center cross."

    # Compute sliding calibration windows (10s window, 5s stride per PRD 8.2)
    window_sec = feat_cfg.window_sec
    stride_sec = 5.0
    t_start = signals_df["t"].iloc[0]
    t_end = signals_df["t"].iloc[-1]
    curr_t = t_start

    window_feats = []
    while curr_t + window_sec <= t_end + 1e-4:
        w_df = signals_df[(signals_df["t"] >= curr_t) & (signals_df["t"] < curr_t + window_sec)]
        feat = compute_window_features(w_df, baseline_open_ear=baseline_open_ear, cfg=feat_cfg)
        if feat is not None:
            window_feats.append(feat)
        curr_t += stride_sec

    if len(window_feats) < 2:
        return None, f"Calibration rejected: only {len(window_feats)} valid windows extracted (minimum 2 needed)."

    # Compute baseline mu and sigma per feature
    w_df_features = pd.DataFrame(window_feats)
    feature_stats: dict[str, dict[str, float]] = {}

    for fname in FEATURE_NAMES:
        if fname in w_df_features.columns:
            vals = w_df_features[fname].to_numpy(dtype=np.float64)
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            # Safe floor if standard deviation is zero
            if sigma < 1e-4:
                sigma = 0.05 if mu == 0 else abs(mu) * 0.1
            feature_stats[fname] = {"mu": mu, "sigma": sigma}
        else:
            feature_stats[fname] = {"mu": 0.0, "sigma": 1.0}

    profile = CalibrationProfile(
        participant_id=participant_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        baseline_open_ear=baseline_open_ear,
        gaze_center=gaze_center,
        head_neutral=head_neutral,
        features=feature_stats,
        quality={
            "valid_ratio": valid_ratio,
            "n_windows": len(window_feats),
            "passed": True,
            "rejection_reason": None,
        },
    )

    return profile, "Calibration successful."


class CalibrationRunner:
    """Manages online stateful calibration buffer."""

    def __init__(self, participant_id: str, calib_cfg: CalibrationConfig, feat_cfg: FeaturesConfig):
        self.participant_id = participant_id
        self.calib_cfg = calib_cfg
        self.feat_cfg = feat_cfg
        self.rows: list[dict] = []
        self._start_t: Optional[float] = None
        self._is_complete = False

    def add_signal_row(self, row_dict: dict) -> None:
        if self._start_t is None:
            self._start_t = row_dict["t"]
        self.rows.append(row_dict)

    @property
    def elapsed_seconds(self) -> float:
        if not self.rows or self._start_t is None:
            return 0.0
        return self.rows[-1]["t"] - self._start_t

    def finish(self) -> Tuple[Optional[CalibrationProfile], str]:
        if not self.rows:
            return None, "No data collected during calibration."
        df = pd.DataFrame(self.rows)
        return run_calibration_from_signals(
            df,
            participant_id=self.participant_id,
            calib_cfg=self.calib_cfg,
            feat_cfg=self.feat_cfg,
        )
