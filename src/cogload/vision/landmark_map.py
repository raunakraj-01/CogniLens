"""MediaPipe Face Landmarker 478-point topology index mappings (PRD Section 7.1)."""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass(frozen=True)
class LandmarkIndices:
    # Right eye EAR points (p1..p6) in image-left perspective
    # p1: outer corner, p2: top-outer, p3: top-inner, p4: inner corner, p5: bottom-inner, p6: bottom-outer
    RIGHT_EYE_EAR: Tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)

    # Left eye EAR points (p1..p6) in image-right perspective
    LEFT_EYE_EAR: Tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)

    # Iris points (center, then 4 boundary points)
    RIGHT_IRIS_CENTER: int = 468
    RIGHT_IRIS_BOUNDARY: Tuple[int, int, int, int] = (469, 470, 471, 472)

    LEFT_IRIS_CENTER: int = 473
    LEFT_IRIS_BOUNDARY: Tuple[int, int, int, int] = (474, 475, 476, 477)

    # Eye corners (outer, inner)
    RIGHT_EYE_CORNERS: Tuple[int, int] = (33, 133)   # outer, inner
    LEFT_EYE_CORNERS: Tuple[int, int] = (263, 362)    # outer, inner

    # Eye upper and lower lid center points for vertical gaze
    RIGHT_EYE_LIDS: Tuple[int, int] = (159, 145)     # upper, lower
    LEFT_EYE_LIDS: Tuple[int, int] = (386, 374)      # upper, lower

    # Head pose landmark indices for 2D-3D correspondence (PRD Section 7.1)
    # Nose tip (1), chin (152), right eye outer (33), left eye outer (263), mouth right (61), mouth left (291)
    HEAD_POSE_LANDMARKS: Tuple[int, int, int, int, int, int] = (1, 152, 33, 263, 61, 291)

    # Optional S4: Brow furrow & mouth tension landmarks
    BROW_INNER_LEFT: int = 285   # Left inner eyebrow point
    BROW_INNER_RIGHT: int = 55   # Right inner eyebrow point
    BROW_LEFT: int = 70          # Outer left brow
    BROW_RIGHT: int = 300        # Outer right brow
    GLABELLA_MID_BROW: int = 9   # Mid-brow glabella
    LIP_OUTER_UPPER: int = 0     # Top vermilion border (Cupid's bow center)
    LIP_OUTER_LOWER: int = 17    # Bottom vermilion border
    LIP_INNER_UPPER: int = 13    # Upper inner lip margin
    LIP_INNER_LOWER: int = 14    # Lower inner lip margin
    LIP_UPPER: int = 0
    LIP_LOWER: int = 17
    MOUTH_CORNER_LEFT: int = 61
    MOUTH_CORNER_RIGHT: int = 291


# Canonical 3D anthropometric face model coordinates (in millimeters)
# Corresponding to: nose tip (1), chin (152), right eye outer (33), left eye outer (263), mouth right (61), mouth left (291)
CANONICAL_3D_FACE_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (1)
        (0.0, -330.0, -65.0),     # Chin (152)
        (-225.0, 170.0, -135.0),  # Right eye outer (33)
        (225.0, 170.0, -135.0),   # Left eye outer (263)
        (-150.0, -150.0, -125.0), # Mouth right corner (61)
        (150.0, -150.0, -125.0),  # Mouth left corner (291)
    ],
    dtype=np.float64,
)
