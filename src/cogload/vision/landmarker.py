"""MediaPipe Face Landmarker wrapper using Tasks API with robust error handling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


@dataclass
class LandmarkResult:
    face_detected: bool
    landmarks_norm: Optional[np.ndarray] = None   # (478, 3) normalized [0, 1]
    landmarks_pixel: Optional[np.ndarray] = None  # (478, 3) in image pixels
    trans_matrix: Optional[np.ndarray] = None     # (4, 4) facial transformation matrix if available
    blendshapes: Optional[dict[str, float]] = None # Facial blendshape scores
    bounding_box_area: float = 0.0


class FaceLandmarkerWrapper:
    """Detects 478 face landmarks from RGB frames using MediaPipe Face Landmarker."""

    def __init__(self, model_path: Optional[str | Path] = None):
        if model_path is None:
            # Look in standard locations
            candidate = Path(__file__).resolve().parent.parent.parent.parent / "models" / "face_landmarker.task"
            if candidate.exists():
                model_path = candidate
            else:
                model_path = Path("models/face_landmarker.task")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {self.model_path}. Run `make setup`."
            )

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> LandmarkResult:
        """Process a single BGR image and return landmarks.

        Handles no-face, multiple faces (selects largest), and partial occlusions safely.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return LandmarkResult(face_detected=False)

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return LandmarkResult(face_detected=False)

        # In case multiple faces detected, pick face with largest bounding box
        best_idx = 0
        max_area = 0.0
        best_coords: Optional[np.ndarray] = None

        for idx, face in enumerate(detection_result.face_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float64)
            min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
            max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
            area = (max_x - min_x) * (max_y - min_y)
            if area > max_area:
                max_area = area
                best_idx = idx
                best_coords = coords

        if best_coords is None or max_area <= 1e-4:
            return LandmarkResult(face_detected=False)

        # Convert to pixel coordinates: x * w, y * h, z * w
        pixel_coords = best_coords.copy()
        pixel_coords[:, 0] *= w
        pixel_coords[:, 1] *= h
        pixel_coords[:, 2] *= w

        # Transformation matrix if available
        matrix = None
        if (
            detection_result.facial_transformation_matrixes
            and len(detection_result.facial_transformation_matrixes) > best_idx
        ):
            matrix = np.array(
                detection_result.facial_transformation_matrixes[best_idx], dtype=np.float64
            )

        # Blendshapes if available
        blendshapes = {}
        if (
            detection_result.face_blendshapes
            and len(detection_result.face_blendshapes) > best_idx
        ):
            for category in detection_result.face_blendshapes[best_idx]:
                blendshapes[category.category_name] = float(category.score)

        return LandmarkResult(
            face_detected=True,
            landmarks_norm=best_coords,
            landmarks_pixel=pixel_coords,
            trans_matrix=matrix,
            blendshapes=blendshapes if blendshapes else None,
            bounding_box_area=max_area,
        )

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()
