"""Unit tests for per-frame signal extractors (EAR, Gaze, Head Pose, Interpolation)."""

import numpy as np
import pandas as pd
from cogload.signals.ear import compute_ear, compute_both_ears
from cogload.signals.gaze import compute_eye_gaze, compute_gaze_ratios
from cogload.signals.head_pose import HeadPoseEstimator
from cogload.signals.frame_signals import interpolate_short_gaps, smooth_gaze_signals
from cogload.vision.landmark_map import LandmarkIndices


def test_ear_hand_computed_geometry():
    # Build synthetic 6-point eye landmark coordinates
    # p1 = (0, 0) [outer corner], p4 = (100, 0) [inner corner], width = 100
    # p2 = (30, 20), p6 = (30, -20) -> vertical height = 40
    # p3 = (70, 20), p5 = (70, -20) -> vertical height = 40
    # Expected EAR = (40 + 40) / (2 * 100) = 80 / 200 = 0.40
    landmarks = np.zeros((478, 3), dtype=np.float64)
    indices = LandmarkIndices.RIGHT_EYE_EAR

    landmarks[indices[0]] = [0.0, 0.0, 0.0]
    landmarks[indices[1]] = [30.0, 20.0, 0.0]
    landmarks[indices[2]] = [70.0, 20.0, 0.0]
    landmarks[indices[3]] = [100.0, 0.0, 0.0]
    landmarks[indices[4]] = [70.0, -20.0, 0.0]
    landmarks[indices[5]] = [30.0, -20.0, 0.0]

    ear = compute_ear(landmarks, indices)
    assert abs(ear - 0.40) < 1e-4

    # When eye closes (vertical distances shrink to 0)
    landmarks[indices[1]] = [30.0, 0.0, 0.0]
    landmarks[indices[6 - 1]] = [30.0, 0.0, 0.0]
    landmarks[indices[2]] = [70.0, 0.0, 0.0]
    landmarks[indices[4]] = [70.0, 0.0, 0.0]
    ear_closed = compute_ear(landmarks, indices)
    assert ear_closed == 0.0


def test_gaze_ratio_centering_and_monotonicity():
    landmarks = np.zeros((478, 3), dtype=np.float64)
    # Eye corners: outer at x=0, inner at x=100
    outer_idx = LandmarkIndices.RIGHT_EYE_CORNERS[0]
    inner_idx = LandmarkIndices.RIGHT_EYE_CORNERS[1]
    upper_idx = LandmarkIndices.RIGHT_EYE_LIDS[0]
    lower_idx = LandmarkIndices.RIGHT_EYE_LIDS[1]
    iris_idx = LandmarkIndices.RIGHT_IRIS_CENTER

    landmarks[outer_idx] = [0.0, 50.0, 0.0]
    landmarks[inner_idx] = [100.0, 50.0, 0.0]
    landmarks[upper_idx] = [50.0, 30.0, 0.0]
    landmarks[lower_idx] = [50.0, 70.0, 0.0]

    # Iris centered at (50, 50) -> gaze_x = 0.0, gaze_y = 0.0
    landmarks[iris_idx] = [50.0, 50.0, 0.0]
    gx, gy = compute_eye_gaze(landmarks, iris_idx, outer_idx, inner_idx, upper_idx, lower_idx)
    assert abs(gx) < 1e-4
    assert abs(gy) < 1e-4

    # Moving iris to right (x=80) should increase gaze_x monotonically
    landmarks[iris_idx] = [80.0, 50.0, 0.0]
    gx_right, _ = compute_eye_gaze(landmarks, iris_idx, outer_idx, inner_idx, upper_idx, lower_idx)
    assert gx_right > gx

    # Moving iris to left (x=20) should decrease gaze_x
    landmarks[iris_idx] = [20.0, 50.0, 0.0]
    gx_left, _ = compute_eye_gaze(landmarks, iris_idx, outer_idx, inner_idx, upper_idx, lower_idx)
    assert gx_left < gx


def test_head_pose_estimator():
    estimator = HeadPoseEstimator()
    # Create canonical landmarks projected at center of 640x480 frame
    landmarks = np.zeros((478, 3), dtype=np.float64)
    # Put standard 2D positions for the 6 head landmarks
    head_indices = LandmarkIndices.HEAD_POSE_LANDMARKS
    landmarks[head_indices[0]] = [320.0, 240.0, 0.0]  # nose tip
    landmarks[head_indices[1]] = [320.0, 340.0, 0.0]  # chin
    landmarks[head_indices[2]] = [240.0, 200.0, 0.0]  # right eye outer
    landmarks[head_indices[3]] = [400.0, 200.0, 0.0]  # left eye outer
    landmarks[head_indices[4]] = [270.0, 300.0, 0.0]  # mouth right
    landmarks[head_indices[5]] = [370.0, 300.0, 0.0]  # mouth left

    yaw, pitch, roll = estimator.estimate(landmarks, (480, 640))
    assert not np.isnan(yaw)
    assert not np.isnan(pitch)
    assert not np.isnan(roll)


def test_interpolate_short_gaps():
    df = pd.DataFrame({
        "t": np.arange(10) * 0.033,
        "valid": [True, True, False, False, True, False, False, False, False, True],
        "ear": [0.30, 0.30, np.nan, np.nan, 0.30, np.nan, np.nan, np.nan, np.nan, 0.30],
        "gaze_x": [0.0, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
    })

    interpolated = interpolate_short_gaps(df, max_gap=2)
    # Gap of length 2 (indices 2, 3) should be interpolated
    assert not np.isnan(interpolated.loc[2, "ear"])
    assert not np.isnan(interpolated.loc[3, "ear"])
    # Gap of length 4 (indices 5..8) exceeds max_gap=2, so remains NaN
    assert np.isnan(interpolated.loc[7, "ear"])
