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


def test_realtime_xai_attributions():
    from cogload.config import load_config
    cfg = load_config()
    engine = RealtimeInferenceEngine(cfg=cfg)
    assert engine.load_model("models/best")

    z_feats = {
        "brow_furrow_mean": 2.5,
        "mouth_tension_mean": 1.2,
        "gaze_speed_p95": 1.8,
        "blink_rate": -2.0,
        "head_speed_mean": 0.5,
    }
    raw_feats = {
        "brow_furrow_mean": 0.35,
        "mouth_tension_mean": 2.5,
        "gaze_speed_p95": 120.0,
        "blink_rate": 6.5,
        "head_speed_mean": 12.0,
    }
    res = engine.compute_feature_attributions(z_feats, raw_feats)

    assert "modalities" in res
    assert "top_drivers" in res
    mods = res["modalities"]
    assert set(mods.keys()) == {"facial_tension", "gaze_dynamics", "blink_fatigue", "head_pose"}
    # Sum of percentages should be ~100%
    assert 99.0 <= sum(mods.values()) <= 101.0

    drivers = res["top_drivers"]
    assert len(drivers) <= 3
    for d in drivers:
        assert "feature" in d
        assert "label" in d
        assert "z_score" in d
        assert "z_str" in d
        assert "raw" in d
    engine.close()


def test_realtime_interventions():
    from cogload.config import load_config
    cfg = load_config()
    engine = RealtimeInferenceEngine(cfg=cfg)

    # 1. Test Sustained High Load Intervention
    engine._high_load_streak_sec = 44.5
    raw_dummy = {"blink_rate": 18.0}
    empty_df = pd.DataFrame()

    alerts1 = engine.check_interventions(
        t_now=100.0, level="HIGH", score=0.85, raw_feats=raw_dummy, sig_df=empty_df
    )
    assert len(alerts1) >= 1
    strain_alert = next(a for a in alerts1 if a["id"] == "cognitive_strain")
    assert strain_alert["severity"] == "warning"
    assert "Cognitive Pacing" in strain_alert["title"]

    # Cooldown should prevent duplicate alerts within cooldown_sec (60s)
    alerts_cooldown = engine.check_interventions(
        t_now=110.0, level="HIGH", score=0.85, raw_feats=raw_dummy, sig_df=empty_df
    )
    assert not any(a["id"] == "cognitive_strain" for a in alerts_cooldown)

    # 2. Test Digital Eye Strain Intervention (low blink rate)
    engine._low_blink_streak_sec = 24.5
    raw_low_blink = {"blink_rate": 4.5}
    alerts2 = engine.check_interventions(
        t_now=200.0, level="LOW", score=0.1, raw_feats=raw_low_blink, sig_df=empty_df
    )
    eye_alert = next((a for a in alerts2 if a["id"] == "eye_strain"), None)
    assert eye_alert is not None
    assert "20-20-20" in eye_alert["title"]

    # 3. Test Ergonomic Posture Advisory (downward pitch)
    engine._posture_streak_sec = 29.5
    posture_df = pd.DataFrame({"pitch": [-18.0] * 30})
    alerts3 = engine.check_interventions(
        t_now=300.0, level="LOW", score=0.1, raw_feats=raw_dummy, sig_df=posture_df
    )
    posture_alert = next((a for a in alerts3 if a["id"] == "neck_posture"), None)
    assert posture_alert is not None
    assert "Ergonomic Neck Posture" in posture_alert["title"]
    engine.close()
