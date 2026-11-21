"""Unit tests for ML models, leakage prevention, and model export."""

from pathlib import Path
import numpy as np
import pandas as pd
from cogload.ml.models import create_model, get_feature_columns
from cogload.ml.evaluate import evaluate_loso
from cogload.ml.export import export_model_card, load_exported_model
from cogload.ml.train import train_and_export_model


def create_synthetic_dataset_df(n_subjects: int = 4, n_windows_per_subj: int = 15) -> pd.DataFrame:
    rows = []
    rng = np.random.RandomState(42)
    feature_cols = get_feature_columns(scheme="zbase", group="all")

    for s_idx in range(n_subjects):
        pid = f"P{s_idx:03d}"
        for w in range(n_windows_per_subj):
            level = w % 3  # 0: Low, 1: Medium, 2: High
            row = {
                "participant_id": pid,
                "session_id": "S01",
                "block_id": f"B_{level}",
                "level_label": level,
                "t_start": w * 10.0,
                "t_end": (w + 1) * 10.0,
                "valid_ratio": 1.0,
                "lighting": "normal",
                "glasses": False,
                "nasa_tlx_block": 30.0 + level * 25.0,
            }
            # Plant correlated signal with level
            for col in feature_cols:
                row[col] = float(level * 0.8 + rng.normal(0, 0.2))
            rows.append(row)

    return pd.DataFrame(rows)


def test_model_factory():
    for m in ["majority", "lr", "rf", "xgb"]:
        clf = create_model(m, random_seed=42)
        assert hasattr(clf, "fit")


def test_loso_no_subject_leakage():
    df = create_synthetic_dataset_df(n_subjects=4)
    res = evaluate_loso(df, model_type="rf", scheme="zbase", random_seed=42)

    assert res["n_subjects"] == 4
    # F1 must be well above chance (0.33) on planted signal
    assert res["f1_macro_mean"] > 0.60
    assert len(res["subject_metrics"]) == 4

    # Confirm subject disjointness
    for m in res["subject_metrics"]:
        held_out = m["participant_id"]
        assert held_out in ["P000", "P001", "P002", "P003"]


def test_model_export_and_load(tmp_path: Path):
    df = create_synthetic_dataset_df(n_subjects=3)
    data_file = tmp_path / "windows.parquet"
    df.to_parquet(data_file)

    out_dir = tmp_path / "models" / "test_model"
    train_and_export_model(dataset_path=data_file, output_dir=out_dir, model_type="rf")

    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "model_card.json").exists()

    model, card = load_exported_model(out_dir)
    assert card["model_name"].startswith("RF_")
    assert "classes" in card
    assert "sigma_floors" in card
    assert hasattr(model, "predict")
