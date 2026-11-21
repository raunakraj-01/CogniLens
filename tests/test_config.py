"""Unit tests for configuration loading."""

from pathlib import Path
from cogload.config import AppConfig, load_config


def test_default_config():
    cfg = AppConfig()
    assert cfg.camera.index == 0
    assert cfg.camera.width == 640
    assert cfg.camera.target_fps == 30
    assert cfg.features.window_sec == 10.0
    assert cfg.calibration.normalization == "zbase"
    assert cfg.random_seed == 42


def test_load_config_file(tmp_path: Path):
    yaml_content = """
camera:
  index: 1
  width: 1280
  height: 720
features:
  window_sec: 15.0
calibration:
  normalization: raw
"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.camera.index == 1
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.features.window_sec == 15.0
    assert cfg.calibration.normalization == "raw"
    assert cfg.camera.target_fps == 30  # Default preserved
