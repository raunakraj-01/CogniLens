"""Eye Aspect Ratio (EAR) computation (PRD Section 7.1)."""

from typing import Tuple
import numpy as np
from cogload.vision.landmark_map import LandmarkIndices


def compute_ear(landmarks_pixel: np.ndarray, eye_indices: Tuple[int, int, int, int, int, int]) -> float:
    """Compute Eye Aspect Ratio (EAR) from 6 landmark points.

    Formula (Soukupová and Cech, 2016):
        EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )

    Points in eye_indices:
        p1 (0): outer corner
        p2 (1): top outer
        p3 (2): top inner
        p4 (3): inner corner
        p5 (4): bottom inner
        p6 (5): bottom outer
    """
    if landmarks_pixel is None or landmarks_pixel.shape[0] < 478:
        return float("nan")

    p1 = landmarks_pixel[eye_indices[0]][:2]
    p2 = landmarks_pixel[eye_indices[1]][:2]
    p3 = landmarks_pixel[eye_indices[2]][:2]
    p4 = landmarks_pixel[eye_indices[3]][:2]
    p5 = landmarks_pixel[eye_indices[4]][:2]
    p6 = landmarks_pixel[eye_indices[5]][:2]

    # Vertical eye distances
    d_v1 = float(np.linalg.norm(p2 - p6))
    d_v2 = float(np.linalg.norm(p3 - p5))

    # Horizontal eye distance (width)
    d_h = float(np.linalg.norm(p1 - p4))

    if d_h < 1e-4:
        return 0.0

    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return float(ear)


def compute_both_ears(landmarks_pixel: np.ndarray) -> Tuple[float, float, float]:
    """Compute (ear_left, ear_right, ear_mean).

    Note: LEFT_EYE corresponds to anatomical left (viewer's right).
    """
    ear_r = compute_ear(landmarks_pixel, LandmarkIndices.RIGHT_EYE_EAR)
    ear_l = compute_ear(landmarks_pixel, LandmarkIndices.LEFT_EYE_EAR)

    if np.isnan(ear_l) and np.isnan(ear_r):
        return float("nan"), float("nan"), float("nan")
    if np.isnan(ear_l):
        return ear_r, ear_r, ear_r
    if np.isnan(ear_r):
        return ear_l, ear_l, ear_l

    ear_mean = (ear_l + ear_r) / 2.0
    return ear_l, ear_r, ear_mean
