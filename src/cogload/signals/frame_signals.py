"""Per-frame behavioral signal extraction and preprocessing (PRD Section 6.2 & 7.1)."""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from cogload.vision.landmarker import LandmarkResult
from cogload.signals.ear import compute_both_ears
from cogload.signals.gaze import compute_gaze_ratios
from cogload.signals.head_pose import HeadPoseEstimator
from cogload.signals.tension import compute_facial_tension


@dataclass
class FrameSignalRow:
    t: float
    valid: bool
    ear_l: float
    ear_r: float
    ear: float
    gaze_x_l: float
    gaze_x_r: float
    gaze_x: float
    gaze_y_l: float
    gaze_y_r: float
    gaze_y: float
    yaw: float
    pitch: float
    roll: float
    brow_furrow: float = 0.0
    mouth_tension: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class FrameSignalExtractor:
    """Extracts per-frame behavioral signals and enforces validity checks."""

    def __init__(self):
        self.head_pose_estimator = HeadPoseEstimator()

    def process_frame(
        self,
        t: float,
        landmark_result: LandmarkResult,
        image_shape: Tuple[int, int] = (480, 640),
    ) -> FrameSignalRow:
        """Extract signals for a single frame."""
        if not landmark_result.face_detected or landmark_result.landmarks_pixel is None:
            return FrameSignalRow(
                t=t,
                valid=False,
                ear_l=float("nan"),
                ear_r=float("nan"),
                ear=float("nan"),
                gaze_x_l=float("nan"),
                gaze_x_r=float("nan"),
                gaze_x=float("nan"),
                gaze_y_l=float("nan"),
                gaze_y_r=float("nan"),
                gaze_y=float("nan"),
                yaw=float("nan"),
                pitch=float("nan"),
                roll=float("nan"),
                brow_furrow=float("nan"),
                mouth_tension=float("nan"),
            )

        pts = landmark_result.landmarks_pixel
        ear_l, ear_r, ear = compute_both_ears(pts)
        gx_l, gx_r, gx, gy_l, gy_r, gy = compute_gaze_ratios(pts)
        yaw, pitch, roll = self.head_pose_estimator.estimate(pts, image_shape)
        brow_furrow, mouth_tension = compute_facial_tension(pts)

        # Validity check per PRD 7.1: face detected, EAR is not NaN and within [0.0, 0.6]
        valid = bool(
            not np.isnan(ear)
            and 0.0 <= ear <= 0.6
            and not (np.isnan(yaw) or np.isnan(pitch) or np.isnan(roll))
        )

        return FrameSignalRow(
            t=t,
            valid=valid,
            ear_l=ear_l,
            ear_r=ear_r,
            ear=ear,
            gaze_x_l=gx_l,
            gaze_x_r=gx_r,
            gaze_x=gx,
            gaze_y_l=gy_l,
            gaze_y_r=gy_r,
            gaze_y=gy,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            brow_furrow=brow_furrow,
            mouth_tension=mouth_tension,
        )

    def reset(self) -> None:
        self.head_pose_estimator.reset()


def interpolate_short_gaps(df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Linearly interpolate short invalid gaps (<= max_gap frames).

    Leaves gaps longer than max_gap as NaN (PRD Section 7.1).
    """
    if df.empty or "valid" not in df.columns:
        return df

    out = df.copy()
    signal_cols = [
        "ear_l",
        "ear_r",
        "ear",
        "gaze_x_l",
        "gaze_x_r",
        "gaze_x",
        "gaze_y_l",
        "gaze_y_r",
        "gaze_y",
        "yaw",
        "pitch",
        "roll",
        "brow_furrow",
        "mouth_tension",
    ]

    # Identify contiguous blocks of invalid frames
    is_invalid = ~out["valid"].fillna(False)
    # Group contiguous blocks of booleans
    blocks = (is_invalid != is_invalid.shift()).cumsum()

    for col in signal_cols:
        if col not in out.columns:
            continue
        # Only interpolate gaps where block length <= max_gap
        col_vals = out[col].to_numpy(copy=True)
        # Find runs of NaN / invalid
        for block_id, group in is_invalid.groupby(blocks):
            if group.iloc[0] and len(group) <= max_gap:
                indices = group.index
                start_idx = indices[0] - 1
                end_idx = indices[-1] + 1
                if start_idx >= 0 and end_idx < len(col_vals):
                    v_start = col_vals[start_idx]
                    v_end = col_vals[end_idx]
                    if not (np.isnan(v_start) or np.isnan(v_end)):
                        # Linear interpolation across the short gap
                        step = (v_end - v_start) / (len(indices) + 1)
                        for step_i, idx in enumerate(indices, start=1):
                            col_vals[idx] = v_start + step * step_i
        out[col] = col_vals

    if "ear" in out.columns:
        out["valid"] = ~out["ear"].isna()

    return out


def smooth_gaze_signals(df: pd.DataFrame, kernel_size: int = 3) -> pd.DataFrame:
    """Light median filter on gaze signals before velocity computation.

    Do not smooth EAR before blink detection (PRD Section 7.1).
    """
    if df.empty or len(df) < kernel_size:
        return df

    out = df.copy()
    gaze_cols = ["gaze_x", "gaze_y", "gaze_x_l", "gaze_x_r", "gaze_y_l", "gaze_y_r"]
    for col in gaze_cols:
        if col in out.columns:
            out[col] = out[col].rolling(window=kernel_size, center=True, min_periods=1).median()
    return out
