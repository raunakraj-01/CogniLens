"""Facial tension behavioral proxies (PRD Section 3.2 S4 stretch feature)."""

import numpy as np
from cogload.vision.landmark_map import LandmarkIndices


def compute_facial_tension(landmarks_pixel: np.ndarray) -> tuple[float, float]:
    """Compute (brow_furrow, mouth_tension) behavioral proxies.

    - Brow furrow: distance between inner eyebrows normalized by inter-pupillary distance.
      When furrowing brow under cognitive strain, this ratio decreases (we invert so higher = more furrowed).
    - Mouth tension: mouth width / (mouth height + eps). Higher = tighter/compressed lips.
    """
    if landmarks_pixel is None or landmarks_pixel.shape[0] < 478:
        return float("nan"), float("nan")

    # Brow furrow: distance between inner eyebrows normalized by inter-pupillary distance.
    # On a relaxed face, inner-brow distance / IPD is ~0.35 - 0.45.
    # When furrowing under mental stress/effort, inner brows pull together, dropping the ratio.
    brow_l = landmarks_pixel[LandmarkIndices.BROW_INNER_LEFT][:2]
    brow_r = landmarks_pixel[LandmarkIndices.BROW_INNER_RIGHT][:2]
    inter_brow_dist = float(np.linalg.norm(brow_r - brow_l))

    # Inter-pupillary distance for scale normalization
    pupil_l = landmarks_pixel[LandmarkIndices.LEFT_IRIS_CENTER][:2]
    pupil_r = landmarks_pixel[LandmarkIndices.RIGHT_IRIS_CENTER][:2]
    ipd = float(np.linalg.norm(pupil_l - pupil_r))

    if ipd < 1e-4:
        brow_furrow = 0.0
    else:
        norm_dist = inter_brow_dist / (0.50 * ipd)
        brow_furrow = float(np.clip(1.0 - norm_dist, 0.0, 1.0))

    # Mouth tension (lip compression ratio proxy)
    # Using outer lip borders (0, 17): resting lips = ~3.0 - 3.5; compressed tight lips in tension = ~5.5 - 9.0
    mouth_l = landmarks_pixel[LandmarkIndices.MOUTH_CORNER_LEFT][:2]
    mouth_r = landmarks_pixel[LandmarkIndices.MOUTH_CORNER_RIGHT][:2]
    lip_top = landmarks_pixel[LandmarkIndices.LIP_OUTER_UPPER][:2]
    lip_bot = landmarks_pixel[LandmarkIndices.LIP_OUTER_LOWER][:2]

    mouth_w = float(np.linalg.norm(mouth_r - mouth_l))
    mouth_h = float(np.linalg.norm(lip_bot - lip_top))

    # Inner mouth opening (mastication / eating / speech detector)
    inner_top = landmarks_pixel[LandmarkIndices.LIP_INNER_UPPER][:2]
    inner_bot = landmarks_pixel[LandmarkIndices.LIP_INNER_LOWER][:2]
    inner_gap = float(np.linalg.norm(inner_bot - inner_top))

    # When mouth is actively open for chewing, eating, or speaking:
    # it is not under isometric tension; set mouth tension to relaxed baseline (~2.5 - 3.2)
    if ipd > 1e-4 and (inner_gap / ipd) > 0.04:
        mouth_tension = float(min(3.2, mouth_w / max(mouth_h, 1e-4)))
    else:
        if mouth_h < 1e-4:
            mouth_tension = 3.5
        else:
            mouth_tension = float(np.clip(mouth_w / mouth_h, 1.0, 8.0))

    return brow_furrow, mouth_tension
