"""Debug overlay script for landmarks, eyes, iris, and head pose (PRD Section 16 Phase 1)."""

import argparse
import sys
import time
import cv2
import numpy as np
from cogload.vision.landmarker import FaceLandmarkerWrapper
from cogload.vision.landmark_map import LandmarkIndices


def draw_debug_overlay(frame: np.ndarray, result) -> np.ndarray:
    """Draw landmark points, eye contours, and iris centers on frame."""
    out = frame.copy()
    if not result.face_detected or result.landmarks_pixel is None:
        cv2.putText(
            out,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return out

    pts = result.landmarks_pixel

    # Draw right eye EAR landmarks
    for idx in LandmarkIndices.RIGHT_EYE_EAR:
        pt = tuple(pts[idx][:2].astype(int))
        cv2.circle(out, pt, 2, (0, 255, 0), -1)

    # Draw left eye EAR landmarks
    for idx in LandmarkIndices.LEFT_EYE_EAR:
        pt = tuple(pts[idx][:2].astype(int))
        cv2.circle(out, pt, 2, (0, 255, 0), -1)

    # Draw iris centers
    r_iris = tuple(pts[LandmarkIndices.RIGHT_IRIS_CENTER][:2].astype(int))
    l_iris = tuple(pts[LandmarkIndices.LEFT_IRIS_CENTER][:2].astype(int))
    cv2.circle(out, r_iris, 3, (0, 0, 255), -1)
    cv2.circle(out, l_iris, 3, (0, 0, 255), -1)

    # Draw head pose points
    for idx in LandmarkIndices.HEAD_POSE_LANDMARKS:
        pt = tuple(pts[idx][:2].astype(int))
        cv2.circle(out, pt, 4, (255, 255, 0), -1)

    cv2.putText(
        out,
        "Face Detected (478 landmarks)",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Debug overlay for facial landmarks")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or path to video file")
    parser.add_argument("--save-test-frame", type=str, default=None, help="Save a test screenshot")
    args = parser.parse_args()

    landmarker = FaceLandmarkerWrapper()

    # Determine input
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error opening source: {args.source}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame from source.")
        sys.exit(1)

    result = landmarker.process(frame)
    annotated = draw_debug_overlay(frame, result)

    if args.save_test_frame:
        cv2.imwrite(args.save_test_frame, annotated)
        print(f"Saved annotated frame to {args.save_test_frame}")

    cap.release()
    landmarker.close()


if __name__ == "__main__":
    main()
