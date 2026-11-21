"""Threaded video capture with bounded queue and frame dropping."""

from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Optional
import numpy as np
from cogload.capture.source import BaseVideoSource

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_sec: float
    frame_index: int
    dropped_before: int = 0


class CaptureThread:
    """Captures frames in a background thread to prevent pipeline latency from blocking camera."""

    def __init__(self, source: BaseVideoSource, max_queue: int = 2):
        self.source = source
        self.max_queue = max(1, max_queue)
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=self.max_queue)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._dropped_count = 0
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._thread.start()

    def _capture_loop(self) -> None:
        dropped_since_last = 0
        while self._running:
            success, frame, ts = self.source.read()
            if not success or frame is None:
                # End of stream or capture failure
                time.sleep(0.005)
                continue

            with self._lock:
                self._frame_count += 1
                self._last_ts = ts

            packet = FramePacket(
                frame=frame,
                timestamp_sec=ts,
                frame_index=self._frame_count,
                dropped_before=dropped_since_last,
            )

            try:
                # Put without blocking; if queue full, drop the oldest frame
                self.queue.put_nowait(packet)
                dropped_since_last = 0
            except queue.Full:
                try:
                    # Drop oldest frame
                    _ = self.queue.get_nowait()
                    with self._lock:
                        self._dropped_count += 1
                    dropped_since_last += 1
                    self.queue.put_nowait(packet)
                except (queue.Empty, queue.Full):
                    with self._lock:
                        self._dropped_count += 1
                    dropped_since_last += 1

    def get_latest(self, timeout: float = 0.1) -> Optional[FramePacket]:
        """Fetch the next available frame packet, or None if timed out."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.source.release()

    @property
    def effective_fps(self) -> float:
        """Effective FPS processed from start."""
        with self._lock:
            if self._start_time is None or self._frame_count == 0:
                return self.source.fps
            elapsed = time.monotonic() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else self.source.fps

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def total_frames(self) -> int:
        with self._lock:
            return self._frame_count
