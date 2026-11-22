"""Real-time inference pipeline and ring buffer (PRD Section 6.7 & 12.3)."""

from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from cogload.config import AppConfig, load_config
from cogload.capture.source import BaseVideoSource
from cogload.vision.landmarker import FaceLandmarkerWrapper, LandmarkResult
from cogload.signals.frame_signals import FrameSignalExtractor, FrameSignalRow, interpolate_short_gaps
from cogload.features.window import compute_window_features
from cogload.features.names import FEATURE_NAMES
from cogload.calibration.profile import CalibrationProfile
from cogload.calibration.normalize import normalize_features, recenter_signals
from cogload.ml.export import load_exported_model
from cogload.realtime.smoothing import OutputSmoother, SmoothedPrediction


@dataclass
class InferenceResult:
    t: float
    face_detected: bool
    fps: float
    calibrated: bool
    load: Dict[str, Any]       # {"level": str, "probs": dict, "score": float, "confidence": float}
    indicators: Dict[str, Any] # {"blink_rate": ..., "ear": ..., "z": {...}}
    warnings: List[str]
    latency_ms: float
    attributions: Dict[str, Any] = field(default_factory=dict)
    interventions: List[Dict[str, Any]] = field(default_factory=list)

    def to_sse_dict(self) -> dict:
        return asdict(self)


class RealtimeInferenceEngine:
    """Executes the full real-time feature extraction and classification pipeline."""

    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        model_dir: Optional[str | Path] = None,
        calibration: Optional[CalibrationProfile] = None,
    ):
        self.cfg = cfg or load_config()
        self.landmarker = FaceLandmarkerWrapper()
        self.signal_extractor = FrameSignalExtractor()
        self.smoother = OutputSmoother(
            alpha=self.cfg.model.smoothing_alpha,
            hysteresis_margin=self.cfg.model.hysteresis_margin,
        )

        # Ring buffer storing signal dictionaries for the last N seconds
        self.window_sec = self.cfg.features.window_sec
        # Max buffer size: window_sec * 60 FPS
        self.buffer: deque[dict] = deque(maxlen=int(self.window_sec * 60))

        # Model state
        self.model = None
        self.model_card: Optional[dict] = None
        self.feature_names: List[str] = FEATURE_NAMES
        self.sigma_floors: Dict[str, float] = {}

        if model_dir is not None:
            self.load_model(model_dir)
        else:
            # Check default path
            cand = Path(self.cfg.model.path)
            if (cand / "model.joblib").exists():
                self.load_model(cand)

        self.calibration = calibration
        self._last_inference_t: float = 0.0
        self._frame_count = 0
        self._start_t: float = time.monotonic()
        self._last_fps = 30.0

        # Closed-loop intervention tracking states
        self._high_load_streak_sec: float = 0.0
        self._low_blink_streak_sec: float = 0.0
        self._posture_streak_sec: float = 0.0
        self._last_alert_t: Dict[str, float] = {}

    def compute_feature_attributions(
        self, z_feats: Dict[str, float], raw_feats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compute interpretable feature attribution and behavioral modality contributions."""
        weights = getattr(self.model, "feature_importances_", None)
        if weights is not None and len(weights) == len(self.feature_names):
            weight_map = dict(zip(self.feature_names, weights))
        else:
            weight_map = {
                "brow_furrow_mean": 0.25,
                "mouth_tension_mean": 0.18,
                "gaze_speed_p95": 0.12,
                "gaze_speed_mean": 0.08,
                "head_speed_mean": 0.06,
                "saccade_rate": 0.05,
                "blink_rate": 0.05,
                "perclos": 0.05,
                "gaze_x_std": 0.04,
                "gaze_y_std": 0.04,
                "yaw_std": 0.04,
                "pitch_std": 0.04,
            }

        modalities = {
            "facial_tension": {
                "name": "Facial Tension",
                "features": ["brow_furrow_mean", "brow_furrow_std", "mouth_tension_mean", "mouth_tension_std"],
                "icon": "⚡",
            },
            "gaze_dynamics": {
                "name": "Ocular Gaze Dynamics",
                "features": ["gaze_speed_mean", "gaze_speed_p95", "gaze_x_std", "gaze_y_std", "saccade_rate", "fixation_ratio"],
                "icon": "🎯",
            },
            "blink_fatigue": {
                "name": "Blink Frequency & Fatigue",
                "features": ["blink_rate", "blink_dur_mean", "blink_dur_std", "perclos", "ear_mean"],
                "icon": "👁️",
            },
            "head_pose": {
                "name": "Head Pose & Velocity",
                "features": ["head_speed_mean", "head_motion_energy", "pitch_std", "yaw_std", "roll_std"],
                "icon": "🧭",
            },
        }

        modality_scores = {}
        feature_impacts = []

        for mod_key, mod_info in modalities.items():
            mod_score = 0.0
            for fn in mod_info["features"]:
                z_val = abs(float(z_feats.get(fn, 0.0)))
                w = float(weight_map.get(fn, 0.03))
                impact = z_val * w
                mod_score += impact
                feature_impacts.append((fn, impact, float(z_feats.get(fn, 0.0)), float(raw_feats.get(fn, 0.0))))
            modality_scores[mod_key] = mod_score

        total_mod_score = sum(modality_scores.values()) or 1e-4
        modality_pcts = {
            k: round((v / total_mod_score) * 100.0, 1) for k, v in modality_scores.items()
        }

        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        name_map = {
            "brow_furrow_mean": "Corrugator brow furrow",
            "mouth_tension_mean": "Mouth compression tension",
            "gaze_speed_p95": "Peak saccadic gaze velocity",
            "gaze_speed_mean": "Mean gaze scanpath speed",
            "head_speed_mean": "Head angular speed",
            "blink_rate": "Blink frequency suppression",
            "perclos": "Eye closure index (PERCLOS)",
            "gaze_x_std": "Horizontal focal dispersion",
            "gaze_y_std": "Vertical focal dispersion",
            "saccade_rate": "Saccade frequency",
            "pitch_std": "Vertical pitch tremor",
            "yaw_std": "Horizontal yaw variance",
        }

        top_drivers = []
        for fn, impact, z_val, raw_val in feature_impacts[:3]:
            label = name_map.get(fn, fn.replace("_", " ").title())
            dir_str = f"+{z_val:.1f}σ" if z_val > 0 else f"{z_val:.1f}σ"
            top_drivers.append({
                "feature": fn,
                "label": label,
                "z_score": round(z_val, 2),
                "z_str": dir_str,
                "raw": round(raw_val, 3),
            })

        return {
            "modalities": modality_pcts,
            "top_drivers": top_drivers,
        }

    def check_interventions(
        self,
        t_now: float,
        level: str,
        score: float,
        raw_feats: Dict[str, float],
        sig_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Evaluate closed-loop ergonomic and adaptive cognitive pacing rules."""
        if not hasattr(self.cfg, "interventions") or not getattr(self.cfg.interventions, "enabled", True):
            return []

        interventions = []
        icfg = self.cfg.interventions
        stride = self.cfg.features.stride_sec_live
        cooldown = getattr(icfg, "cooldown_sec", 60.0)

        # 1. Sustained High Workload / Cognitive Pacing
        if level == "HIGH" or score >= 0.70:
            self._high_load_streak_sec += stride
        else:
            self._high_load_streak_sec = max(0.0, self._high_load_streak_sec - stride * 0.5)

        high_thresh = getattr(icfg, "high_load_streak_sec", 45.0)
        last_high = self._last_alert_t.get("high_load", 0.0)
        if self._high_load_streak_sec >= high_thresh and (t_now - last_high) >= cooldown:
            self._last_alert_t["high_load"] = t_now
            interventions.append({
                "id": "cognitive_strain",
                "severity": "warning",
                "title": "Cognitive Pacing Recommendation",
                "message": f"Sustained elevated workload detected ({int(self._high_load_streak_sec)}s). Take a 2-minute cognitive breather to maintain peak focus.",
                "action": "Take a 2-min breather",
            })

        # 2. Digital Eye Strain (20-20-20 Rule)
        blink_rate = float(raw_feats.get("blink_rate", 18.0))
        blink_thresh = getattr(icfg, "low_blink_bpm_thresh", 8.0)
        if blink_rate < blink_thresh:
            self._low_blink_streak_sec += stride
        else:
            self._low_blink_streak_sec = max(0.0, self._low_blink_streak_sec - stride * 0.5)

        blink_streak_thresh = getattr(icfg, "low_blink_streak_sec", 25.0)
        last_blink = self._last_alert_t.get("low_blink", 0.0)
        if self._low_blink_streak_sec >= blink_streak_thresh and (t_now - last_blink) >= cooldown:
            self._last_alert_t["low_blink"] = t_now
            interventions.append({
                "id": "eye_strain",
                "severity": "info",
                "title": "20-20-20 Visual Rest Advisory",
                "message": f"Blink frequency suppressed ({round(blink_rate, 1)} bpm). Risk of dry eye strain. Look 20 feet away for 20 seconds.",
                "action": "Rest eyes for 20s",
            })

        # 3. Ergonomic Neck / Posture Alert (Downward Head Pitch)
        pitch_vals = sig_df["pitch"].dropna().to_numpy(dtype=np.float64) if "pitch" in sig_df.columns else np.array([])
        mean_pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0
        pitch_thresh = getattr(icfg, "forward_pitch_thresh_deg", -12.0)
        if mean_pitch < pitch_thresh:
            self._posture_streak_sec += stride
        else:
            self._posture_streak_sec = max(0.0, self._posture_streak_sec - stride * 0.5)

        posture_thresh = getattr(icfg, "posture_streak_sec", 30.0)
        last_posture = self._last_alert_t.get("posture", 0.0)
        if self._posture_streak_sec >= posture_thresh and (t_now - last_posture) >= cooldown:
            self._last_alert_t["posture"] = t_now
            interventions.append({
                "id": "neck_posture",
                "severity": "info",
                "title": "Ergonomic Neck Posture Advisory",
                "message": f"Persistent downward cervical pitch ({round(mean_pitch, 1)}°). Elevate laptop display to eye level.",
                "action": "Adjust screen height",
            })

        return interventions

    def load_model(self, model_dir: str | Path) -> bool:
        try:
            model, card = load_exported_model(model_dir)
            self.model = model
            self.model_card = card
            self.feature_names = card.get("feature_names", FEATURE_NAMES)
            self.sigma_floors = card.get("sigma_floors", {})
            return True
        except Exception as e:
            print(f"Warning: Failed to load model from {model_dir}: {e}")
            self.model = None
            self.model_card = None
            return False

    def set_calibration(self, profile: CalibrationProfile) -> None:
        self.calibration = profile

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp_sec: float
    ) -> Tuple[Optional[InferenceResult], LandmarkResult]:
        """Process one video frame.

        Appends to ring buffer, and every stride_sec_live (1.0s), computes window features
        and produces an InferenceResult packet.
        """
        start_proc = time.perf_counter()
        self._frame_count += 1
        elapsed = time.monotonic() - self._start_t
        if elapsed > 1.0:
            self._last_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_t = time.monotonic()

        # 1. Landmark extraction
        res = self.landmarker.process(frame_bgr)

        # 2. Per-frame signal extraction
        row = self.signal_extractor.process_frame(
            t=timestamp_sec, landmark_result=res, image_shape=frame_bgr.shape[:2]
        )
        self.buffer.append(row.to_dict())

        # Check if 1 second has elapsed since last window inference (stride_sec_live)
        stride = self.cfg.features.stride_sec_live
        if (timestamp_sec - self._last_inference_t) < stride and self._last_inference_t > 0:
            return None, res

        self._last_inference_t = timestamp_sec

        # 3. Buffer check
        warnings: List[str] = []
        if not res.face_detected:
            warnings.append("no_face")
        if self.calibration is None or not self.calibration.is_valid():
            warnings.append("not_calibrated")
        if self.model is None:
            warnings.append("model_missing")

        # Convert ring buffer to DataFrame
        buf_list = list(self.buffer)
        if len(buf_list) < 10:
            return None, res

        sig_df = pd.DataFrame(buf_list)
        # Keep only the last window_sec
        t_max = sig_df["t"].iloc[-1]
        sig_df = sig_df[sig_df["t"] >= (t_max - self.window_sec)].copy()

        # Preprocessing: interpolate short gaps and recenter if calibrated
        sig_df = interpolate_short_gaps(sig_df, max_gap=self.cfg.processing.interpolate_gap_frames)
        if self.calibration is not None:
            sig_df = recenter_signals(sig_df, self.calibration)

        # 4. Compute window features
        baseline_ear = (
            self.calibration.baseline_open_ear if self.calibration else 0.28
        )
        raw_feats = compute_window_features(sig_df, baseline_open_ear=baseline_ear, cfg=self.cfg.features)

        if raw_feats is None:
            # Low validity
            warnings.append("low_validity")
            latency_ms = (time.perf_counter() - start_proc) * 1000.0
            prev_level = self.smoother.current_level
            prev_probs = self.smoother.prev_probs if self.smoother.prev_probs is not None else np.array([0.5, 0.3, 0.2])
            prev_score = float(prev_probs[1] * 0.5 + prev_probs[2] * 1.0)
            return (
                InferenceResult(
                    t=timestamp_sec,
                    face_detected=res.face_detected,
                    fps=round(self._last_fps, 1),
                    calibrated=bool(self.calibration and self.calibration.is_valid()),
                    load={
                        "level": prev_level,
                        "probs": {
                            "low": round(float(prev_probs[0]), 3),
                            "medium": round(float(prev_probs[1]), 3),
                            "high": round(float(prev_probs[2]), 3),
                        },
                        "score": round(prev_score, 2),
                        "confidence": 0.5,
                    },
                    indicators={
                        "blink_rate": 0.0,
                        "ear": 0.0,
                        "gaze_dispersion": 0.0,
                        "head_motion": 0.0,
                        "z": {"blink_rate": 0.0, "gaze_dispersion": 0.0, "head_motion": 0.0},
                    },
                    warnings=warnings,
                    latency_ms=round(latency_ms, 2),
                ),
                res,
            )

        # 5. Normalization
        z_feats = normalize_features(
            raw_feats,
            self.calibration,
            scheme="zbase",
            sigma_floors=self.sigma_floors,
        )

        def clean_num(v, default=0.0):
            if v is None:
                return default
            try:
                fv = float(v)
                return default if (np.isnan(fv) or np.isinf(fv)) else fv
            except Exception:
                return default

        # Detect oral activity (chewing, eating, speaking) artifacts:
        # Signature: mouth aperture oscillates (mouth tension std >= threshold) while brow furrow is relaxed
        is_oral_activity = False
        if hasattr(self.cfg, "artifacts") and getattr(self.cfg.artifacts, "filter_oral_motion", True):
            m_std = clean_num(raw_feats.get("mouth_tension_std", 0.0))
            b_val = clean_num(raw_feats.get("brow_furrow_mean", 0.0))
            z_brow = clean_num(z_feats.get("brow_furrow_mean", 0.0))
            m_std_thresh = getattr(self.cfg.artifacts, "oral_mouth_std_thresh", 0.20)
            b_max_thresh = getattr(self.cfg.artifacts, "oral_brow_max_thresh", 0.28)

            if m_std >= m_std_thresh and (b_val <= b_max_thresh or z_brow <= 0.7):
                is_oral_activity = True
                warnings.append("oral_activity")

        # 6. Classification & Smoothing
        if self.model is not None:
            # If oral motion detected (eating/chewing), neutralize mastication artifacts:
            z_feats_filtered = dict(z_feats)
            if is_oral_activity:
                z_feats_filtered["mouth_tension_mean"] = 0.0
                z_feats_filtered["mouth_tension_std"] = 0.0
                # Dampen jaw-coupled head motion artifacts
                z_feats_filtered["head_speed_mean"] = min(0.2, clean_num(z_feats.get("head_speed_mean", 0.0)))
                z_feats_filtered["pitch_std"] = min(0.2, clean_num(z_feats.get("pitch_std", 0.0)))
                z_feats_filtered["yaw_std"] = min(0.3, clean_num(z_feats.get("yaw_std", 0.0)))
                z_feats_filtered["head_motion_energy"] = min(0.3, clean_num(z_feats.get("head_motion_energy", 0.0)))
                # Dampen chewing-coupled gaze flutter
                z_feats_filtered["gaze_speed_mean"] = min(0.3, clean_num(z_feats.get("gaze_speed_mean", 0.0)))
                z_feats_filtered["gaze_speed_p95"] = min(0.3, clean_num(z_feats.get("gaze_speed_p95", 0.0)))
                z_feats_filtered["saccade_rate"] = min(0.2, clean_num(z_feats.get("saccade_rate", 0.0)))

            # Prepare feature vector matching expected training order
            feat_vec = np.array(
                [[z_feats_filtered.get(fn, 0.0) for fn in self.feature_names]], dtype=np.float64
            )
            # Impute NaNs if any
            feat_vec = np.nan_to_num(feat_vec, nan=0.0)

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feat_vec)[0]
            else:
                pred_cls = int(self.model.predict(feat_vec)[0])
                probs = np.zeros(3)
                probs[min(2, pred_cls)] = 1.0

            # Suppress false high-load alarms from oral activity (eating/chewing/speaking) or gross head motion
            b_val = clean_num(raw_feats.get("brow_furrow_mean", 0.0))
            head_speed = clean_num(raw_feats.get("head_speed_mean", 0.0))
            b_max_thresh = getattr(self.cfg.artifacts, "oral_brow_max_thresh", 0.28) if hasattr(self.cfg, "artifacts") else 0.28

            if is_oral_activity and b_val <= b_max_thresh:
                # Eating/chewing with relaxed eyebrows: cap high-load probability to prevent false alarms
                if probs[2] > 0.15:
                    excess = probs[2] - 0.15
                    probs[2] = 0.15
                    probs[0] += excess * 0.75
                    probs[1] += excess * 0.25
            elif head_speed > 45.0 and b_val <= b_max_thresh:
                # Physical head motion without brow furrow: cap high load
                if probs[2] > 0.25:
                    excess = probs[2] - 0.25
                    probs[2] = 0.25
                    probs[0] += excess * 0.70
                    probs[1] += excess * 0.30

            smoothed = self.smoother.update(probs)
            load_dict = {
                "level": smoothed.level,
                "probs": smoothed.probs,
                "score": smoothed.score,
                "confidence": smoothed.confidence,
            }
        else:
            # Missing model fallback (FR-UI-2: indicators only)
            load_dict = {
                "level": "UNKNOWN",
                "probs": {"low": 0.33, "medium": 0.33, "high": 0.34},
                "score": 0.0,
                "confidence": 0.0,
            }

        # Check motion warnings (suppress if jaw chewing motion is active)
        head_speed = clean_num(raw_feats.get("head_speed_mean", 0.0))
        if head_speed > 45.0 and not is_oral_activity:
            warnings.append("high_head_motion")

        brow_f = clean_num(raw_feats.get("brow_furrow_mean", 0.0))
        mouth_t = clean_num(raw_feats.get("mouth_tension_mean", 0.0))
        indicators = {
            "blink_rate": round(clean_num(raw_feats.get("blink_rate", 0.0)), 1),
            "ear": round(clean_num(raw_feats.get("ear_mean", 0.0)), 3),
            "gaze_dispersion": round(clean_num(raw_feats.get("gaze_x_std", 0.0)) + clean_num(raw_feats.get("gaze_y_std", 0.0)), 3),
            "head_motion": round(head_speed, 2),
            "brow_furrow": round(brow_f, 2),
            "mouth_tension": round(mouth_t, 2),
            "z": {
                "blink_rate": round(clean_num(z_feats.get("blink_rate", 0.0)), 2),
                "gaze_dispersion": round(clean_num(z_feats.get("gaze_x_std", 0.0)) + clean_num(z_feats.get("gaze_y_std", 0.0)), 2),
                "head_motion": round(clean_num(z_feats.get("head_speed_mean", 0.0)), 2),
                "brow_furrow": round(clean_num(z_feats.get("brow_furrow_mean", 0.0)), 2),
                "mouth_tension": round(clean_num(z_feats.get("mouth_tension_mean", 0.0)), 2),
            },
        }

        latency_ms = (time.perf_counter() - start_proc) * 1000.0

        # Compute XAI Feature & Modality Attributions
        attributions = self.compute_feature_attributions(
            z_feats_filtered if "z_feats_filtered" in locals() else z_feats, raw_feats
        )

        # Check Closed-Loop Ergonomic & Adaptive Pacing Interventions
        interventions = self.check_interventions(
            t_now=timestamp_sec,
            level=load_dict["level"],
            score=load_dict["score"],
            raw_feats=raw_feats,
            sig_df=sig_df,
        )

        return (
            InferenceResult(
                t=timestamp_sec,
                face_detected=res.face_detected,
                fps=round(self._last_fps, 1),
                calibrated=bool(self.calibration and self.calibration.is_valid()),
                load=load_dict,
                indicators=indicators,
                warnings=warnings,
                latency_ms=round(latency_ms, 2),
                attributions=attributions,
                interventions=interventions,
            ),
            res,
        )

    def close(self) -> None:
        self.landmarker.close()
