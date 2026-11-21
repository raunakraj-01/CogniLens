"""Unit tests for OutputSmoother and RealtimeInferenceEngine."""

import numpy as np
import pandas as pd
from cogload.realtime.smoothing import OutputSmoother
from cogload.realtime.engine import RealtimeInferenceEngine
from cogload.calibration.profile import CalibrationProfile


def test_output_smoother_ema_and_hysteresis():
    smoother = OutputSmoother(alpha=0.5, hysteresis_margin=0.15)
    assert smoother.current_level == "LOW"

    # Step 1: Input mostly LOW
    res1 = smoother.update(np.array([0.8, 0.1, 0.1]))
    assert res1.level == "LOW"
    assert res1.score < 0.2

    # Step 2: Sudden single spike to MEDIUM (0.5), but hysteresis requires prob > curr_prob + 0.15
    res2 = smoother.update(np.array([0.2, 0.6, 0.2]))
    # EMA: probs become 0.5 * [0.8, 0.1, 0.1] + 0.5 * [0.2, 0.6, 0.2] = [0.5, 0.35, 0.15]
    # LOW prob = 0.5, MEDIUM prob = 0.35 -> still LOW
    assert res2.level == "LOW"

    # Step 3: Sustained HIGH input
    for _ in range(5):
        res3 = smoother.update(np.array([0.05, 0.15, 0.80]))
    assert res3.level == "HIGH"
    assert res3.score > 0.7


def test_realtime_engine_model_missing_graceful(tmp_path):
    # FR-UI-2: Dashboard/engine works when model is missing (shows indicators only + warning)
    from cogload.config import load_config
    cfg = load_config()
    cfg.model.path = str(tmp_path / "empty_models")
    engine = RealtimeInferenceEngine(cfg=cfg, model_dir=tmp_path / "empty_models")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process several frames
    for i in range(15):
        res_packet, _ = engine.process_frame(blank_frame, timestamp_sec=i * 0.1)

    # Must not crash
    assert engine.model is None
    engine.close()


def test_oral_activity_artifact_filtering():
    from cogload.config import load_config
    cfg = load_config()
    engine = RealtimeInferenceEngine(cfg=cfg)
    assert engine.load_model("models/best")

    # Simulate 30 frames with oscillating mouth tension (chewing/eating) but relaxed brows
    for i in range(30):
        # Mouth tension oscillating between 2.0 and 4.0 (m_std > 0.5), relaxed brow (0.16)
        mt = 3.0 + 1.0 * np.sin(i * 0.8)
        row = {
            "t": float(i * 0.35),
            "valid": True,
            "ear_l": 0.28,
            "ear_r": 0.28,
            "ear": 0.28,
            "gaze_x_l": 0.0,
            "gaze_x_r": 0.0,
            "gaze_x": 0.0,
            "gaze_y_l": 0.0,
            "gaze_y_r": 0.0,
            "gaze_y": 0.0,
            "yaw": 0.0,
            "pitch": 2.0 * np.sin(i * 0.8),  # Jaw-induced pitch oscillation
            "roll": 0.0,
            "brow_furrow": 0.16,  # Relaxed brow
            "mouth_tension": float(mt),
        }
        engine.buffer.append(row)

    # Process frame to trigger inference
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_packet, _ = engine.process_frame(blank_frame, timestamp_sec=11.0)

    # Must detect oral activity and MUST NOT falsely predict HIGH cognitive load
    assert res_packet is not None
    assert "oral_activity" in res_packet.warnings
    assert res_packet.load["level"] != "HIGH"
    assert res_packet.load["probs"]["high"] <= 0.25
    engine.close()
