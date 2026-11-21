"""Unit tests for dataset builder and window segmentation."""

from pathlib import Path
import pandas as pd
from scripts.make_synthetic import generate_synthetic_session
from cogload.data.build_dataset import build_dataset_from_sessions
from cogload.config import AppConfig, FeaturesConfig


def test_build_dataset_from_synthetic(tmp_path: Path):
    # Generate 2 synthetic participants
    generate_synthetic_session("P001", base_dir=tmp_path, block_duration_sec=35.0)
    generate_synthetic_session("P002", base_dir=tmp_path, block_duration_sec=35.0)

    cfg = AppConfig()
    cfg.features = FeaturesConfig(
        window_sec=10.0,
        stride_sec_offline=5.0,
        drop_block_start_sec=5.0,  # Trim 5s instead of 15s for short test
        min_valid_ratio=0.8,
    )

    output_parquet = tmp_path / "processed" / "windows.parquet"
    df = build_dataset_from_sessions(
        sessions_dir=tmp_path,
        output_path=output_parquet,
        cfg=cfg,
    )

    assert not df.empty
    assert output_parquet.exists()
    assert "participant_id" in df.columns
    assert "level_label" in df.columns
    assert "ear_mean_raw" in df.columns
    assert "ear_mean_zbase" in df.columns
    assert "blink_rate_raw" in df.columns
    assert "blink_rate_zbase" in df.columns
    assert set(df["participant_id"].unique()) == {"P001", "P002"}
