"""Head pose estimation using cv2.solvePnP and 3D face model (PRD Section 7.1)."""

from typing import Tuple, Optional
import cv2
import numpy as np
from cogload.vision.landmark_map import LandmarkIndices, CANONICAL_3D_FACE_MODEL


class HeadPoseEstimator:
    """Estimates head pose (yaw, pitch, roll in degrees) with phase unwrapping."""

    def __init__(self):
        self.prev_yaw: Optional[float] = None
        self.prev_pitch: Optional[float] = None
        self.prev_roll: Optional[float] = None

    def estimate(
        self, landmarks_pixel: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """Estimate (yaw, pitch, roll) in degrees from facial landmarks.

        Sign convention (documented per PRD):
        - Yaw: turning right > 0, turning left < 0
        - Pitch: looking up > 0, looking down < 0
        - Roll: tilting right > 0, tilting left < 0
        """
        if landmarks_pixel is None or landmarks_pixel.shape[0] < 478:
            return float("nan"), float("nan"), float("nan")

        h, w = image_shape[:2]
        # Approximate camera intrinsics
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 2D image points for corresponding 3D canonical points (PRD 7.1)
        image_points = np.array(
            [landmarks_pixel[idx][:2] for idx in LandmarkIndices.HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_3D_FACE_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return float("nan"), float("nan"), float("nan")

        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles from rotation matrix
        # Decompose projection matrix or RQ decomposition
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Apply phase unwrapping to prevent ±180° discontinuous jumps
        yaw = self._unwrap(yaw, self.prev_yaw)
        pitch = self._unwrap(pitch, self.prev_pitch)
        roll = self._unwrap(roll, self.prev_roll)

        self.prev_yaw = yaw
        self.prev_pitch = pitch
        self.prev_roll = roll

        return yaw, pitch, roll

    @staticmethod
    def _unwrap(current: float, previous: Optional[float]) -> float:
        if previous is None or np.isnan(previous) or np.isnan(current):
            return current
        diff = current - previous
        if diff > 180.0:
            current -= 360.0
        elif diff < -180.0:
            current += 360.0
        return current

    def reset(self) -> None:
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_roll = None
