"""Iris-based gaze ratio computation (PRD Section 7.1)."""

from typing import Tuple
import numpy as np
from cogload.vision.landmark_map import LandmarkIndices


def compute_eye_gaze(
    landmarks_pixel: np.ndarray,
    iris_center_idx: int,
    outer_corner_idx: int,
    inner_corner_idx: int,
    upper_lid_idx: int,
    lower_lid_idx: int,
) -> Tuple[float, float]:
    """Compute horizontal (gaze_x) and vertical (gaze_y) gaze coordinates for one eye.

    Horizontal: projection of iris center onto eye axis (outer to inner corner),
    normalized to [-0.5, 0.5] (0.0 = looking straight).

    Vertical: normalized position between upper and lower lids, centered to [-0.5, 0.5].
    """
    if landmarks_pixel is None or landmarks_pixel.shape[0] < 478:
        return float("nan"), float("nan")

    iris = landmarks_pixel[iris_center_idx][:2]
    outer = landmarks_pixel[outer_corner_idx][:2]
    inner = landmarks_pixel[inner_corner_idx][:2]
    upper = landmarks_pixel[upper_lid_idx][:2]
    lower = landmarks_pixel[lower_lid_idx][:2]

    # Horizontal axis vector from outer to inner corner
    eye_axis = inner - outer
    eye_width = float(np.linalg.norm(eye_axis))
    if eye_width < 1e-4:
        return 0.0, 0.0

    eye_unit = eye_axis / eye_width
    iris_rel = iris - outer
    proj = float(np.dot(iris_rel, eye_unit))
    ratio_x = proj / eye_width
    gaze_x = ratio_x - 0.5

    # Vertical distance between upper and lower lid
    lid_dist = float(lower[1] - upper[1])
    if abs(lid_dist) < 5.0:
        ratio_y = 0.5
    else:
        ratio_y = float(iris[1] - upper[1]) / lid_dist
    gaze_y = float(np.clip(ratio_y - 0.5, -0.5, 0.5))
    gaze_x = float(np.clip(gaze_x, -0.5, 0.5))

    return gaze_x, gaze_y


def compute_gaze_ratios(landmarks_pixel: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Compute (gaze_x_l, gaze_x_r, gaze_x, gaze_y_l, gaze_y_r, gaze_y)."""
    gx_r, gy_r = compute_eye_gaze(
        landmarks_pixel,
        LandmarkIndices.RIGHT_IRIS_CENTER,
        LandmarkIndices.RIGHT_EYE_CORNERS[0],
        LandmarkIndices.RIGHT_EYE_CORNERS[1],
        LandmarkIndices.RIGHT_EYE_LIDS[0],
        LandmarkIndices.RIGHT_EYE_LIDS[1],
    )
    gx_l, gy_l = compute_eye_gaze(
        landmarks_pixel,
        LandmarkIndices.LEFT_IRIS_CENTER,
        LandmarkIndices.LEFT_EYE_CORNERS[0],
        LandmarkIndices.LEFT_EYE_CORNERS[1],
        LandmarkIndices.LEFT_EYE_LIDS[0],
        LandmarkIndices.LEFT_EYE_LIDS[1],
    )

    gx = (gx_l + gx_r) / 2.0 if not (np.isnan(gx_l) or np.isnan(gx_r)) else gx_r
    gy = (gy_l + gy_r) / 2.0 if not (np.isnan(gy_l) or np.isnan(gy_r)) else gy_r

    return gx_l, gx_r, gx, gy_l, gy_r, gy
