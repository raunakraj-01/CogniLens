"""Typed configuration loader for cogload."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    target_fps: int = 30


@dataclass
class ProcessingConfig:
    max_queue: int = 2
    interpolate_gap_frames: int = 3


@dataclass
class FeaturesConfig:
    window_sec: float = 10.0
    stride_sec_live: float = 1.0
    stride_sec_offline: float = 5.0
    min_valid_ratio: float = 0.5
    drop_block_start_sec: float = 15.0
    blink_ratio: float = 0.75
    blink_min_ms: float = 60.0
    blink_max_ms: float = 600.0
    blink_refractory_ms: float = 100.0
    closure_ratio: float = 0.2
    saccade_speed_thresh: float = 0.6
    fixation_speed_thresh: float = 0.15
    offscreen_gaze_bound: float = 0.35


@dataclass
class CalibrationConfig:
    phase_a_sec: float = 60.0
    phase_b_sec: float = 30.0
    min_valid_ratio: float = 0.55
    normalization: str = "zbase"


@dataclass
class ModelConfig:
    path: str = "models/best"
    smoothing_alpha: float = 0.4
    hysteresis_margin: float = 0.1


@dataclass
class ArtifactConfig:
    filter_oral_motion: bool = True
    oral_mouth_std_thresh: float = 0.20
    oral_brow_max_thresh: float = 0.28
    high_motion_speed_thresh: float = 45.0


@dataclass
class InterventionConfig:
    enabled: bool = True
    high_load_streak_sec: float = 45.0
    low_blink_streak_sec: float = 25.0
    low_blink_bpm_thresh: float = 8.0
    forward_pitch_thresh_deg: float = -12.0
    posture_streak_sec: float = 30.0
    cooldown_sec: float = 60.0


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug_preview: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    interventions: InterventionConfig = field(default_factory=InterventionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    random_seed: int = 42


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(
        camera=CameraConfig(**raw.get("camera", {})),
        processing=ProcessingConfig(**raw.get("processing", {})),
        features=FeaturesConfig(**raw.get("features", {})),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        model=ModelConfig(**raw.get("model", {})),
        artifacts=ArtifactConfig(**raw.get("artifacts", {})),
        interventions=InterventionConfig(**raw.get("interventions", {})),
        server=ServerConfig(**raw.get("server", {})),
        random_seed=raw.get("random_seed", 42),
    )
