"""Video source abstractions for camera, video files, and replay."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class BaseVideoSource(ABC):
    """Abstract video frame source."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read the next frame.

        Returns:
            (success, frame, timestamp_sec)
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying video capture resources."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second."""
        pass


class WebcamSource(BaseVideoSource):
    """Live webcam capture source."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, target_fps: int = 30):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self._target_fps = float(target_fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        import time

        ret, frame = self.cap.read()
        ts = time.monotonic()
        if not ret:
            return False, None, ts
        return True, frame, ts

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    @property
    def fps(self) -> float:
        val = self.cap.get(cv2.CAP_PROP_FPS)
        return float(val) if val > 0 else self._target_fps


class FileVideoSource(BaseVideoSource):
    """File video source with synthetic monotonic timestamps based on frame index / FPS."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.path}")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(raw_fps) if raw_fps > 0 else 30.0
        self._frame_idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None, self._frame_idx / self._fps
        ts = self._frame_idx / self._fps
        self._frame_idx += 1
        return True, frame, ts

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    @property
    def fps(self) -> float:
        return self._fps


class ReplaySource(BaseVideoSource):
    """Replays frames or pre-recorded signal rows at a target rate."""

    def __init__(self, frames: list[np.ndarray], fps: float = 30.0):
        self.frames = frames
        self._fps = float(fps)
        self._idx = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self._idx >= len(self.frames):
            return False, None, self._idx / self._fps
        frame = self.frames[self._idx]
        ts = self._idx / self._fps
        self._idx += 1
        return True, frame, ts

    def release(self) -> None:
        self.frames.clear()

    @property
    def fps(self) -> float:
        return self._fps
