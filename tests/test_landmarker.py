"""Unit tests for FaceLandmarker wrapper and landmark mapping."""

import numpy as np
import cv2
from cogload.vision.landmarker import FaceLandmarkerWrapper
from cogload.vision.landmark_map import LandmarkIndices, CANONICAL_3D_FACE_MODEL


def test_landmark_map_indices():
    assert len(LandmarkIndices.RIGHT_EYE_EAR) == 6
    assert len(LandmarkIndices.LEFT_EYE_EAR) == 6
    assert LandmarkIndices.RIGHT_IRIS_CENTER == 468
    assert LandmarkIndices.LEFT_IRIS_CENTER == 473
    assert len(LandmarkIndices.HEAD_POSE_LANDMARKS) == 6
    assert CANONICAL_3D_FACE_MODEL.shape == (6, 3)


def test_landmarker_no_face():
    # Blank black image has no face
    wrapper = FaceLandmarkerWrapper()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = wrapper.process(blank)

    assert result.face_detected is False
    assert result.landmarks_norm is None
    assert result.landmarks_pixel is None
    wrapper.close()


def test_landmarker_synthetic_face_drawing():
    # Render a simple synthetic face diagram to test landmark pipeline
    wrapper = FaceLandmarkerWrapper()
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Draw face oval
    cv2.ellipse(img, (320, 240), (140, 180), 0, 0, 360, (160, 140, 120), -1)
    # Eyes
    cv2.circle(img, (270, 200), 20, (255, 255, 255), -1)
    cv2.circle(img, (370, 200), 20, (255, 255, 255), -1)
    cv2.circle(img, (270, 200), 8, (50, 30, 20), -1)
    cv2.circle(img, (370, 200), 8, (50, 30, 20), -1)
    # Nose
    cv2.line(img, (320, 210), (320, 260), (100, 80, 60), 4)
    # Mouth
    cv2.ellipse(img, (320, 310), (40, 15), 0, 0, 180, (80, 50, 50), 3)

    result = wrapper.process(img)
    # Whether or not ML face detector fires on simplified geometric sketch, it must not crash
    assert isinstance(result.face_detected, bool)
    wrapper.close()
