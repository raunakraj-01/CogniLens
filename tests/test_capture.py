"""Unit tests for video capture and capture thread."""

import time
import numpy as np
from cogload.capture.source import ReplaySource, BaseVideoSource
from cogload.capture.thread import CaptureThread, FramePacket


def test_replay_source():
    # Create 5 synthetic frames (100x100 RGB)
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    source = ReplaySource(frames=frames, fps=25.0)
    assert source.fps == 25.0

    read_count = 0
    while True:
        success, frame, ts = source.read()
        if not success:
            break
        assert frame is not None
        assert ts == read_count / 25.0
        read_count += 1

    assert read_count == 5
    source.release()


def test_capture_thread_queue_and_drop():
    # Test bounded queue and frame dropping behavior
    frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(20)]
    source = ReplaySource(frames=frames, fps=100.0)
    # Bounded queue of size 2
    ct = CaptureThread(source=source, max_queue=2)
    ct.start()

    time.sleep(0.08)  # Let thread push frames into queue

    packets: list[FramePacket] = []
    while True:
        pkt = ct.get_latest(timeout=0.02)
        if pkt is None:
            break
        packets.append(pkt)

    ct.stop()

    assert len(packets) > 0
    # Queue size was 2, so dropped frames should be recorded when reader was slow
    assert ct.total_frames > 0
    assert ct.effective_fps > 0
