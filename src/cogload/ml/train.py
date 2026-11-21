"""Model training and export orchestrator (PRD Section 10 & 16 Phase 7)."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from cogload.config import AppConfig, load_config
from cogload.features.names import FEATURE_NAMES
from cogload.ml.models import create_model, get_feature_columns
from cogload.ml.evaluate import evaluate_loso
from cogload.ml.export import export_model_card


def train_and_export_model(
    dataset_path: str | Path = "data/processed/windows_W10_S5.parquet",
    output_dir: str | Path = "models/best",
    model_type: str = "rf",
    scheme: str = "zbase",
    cfg: Optional[AppConfig] = None,
) -> Path:
    """Train classifier on dataset and export model artifact + model card."""
    if cfg is None:
        cfg = load_config()

    data_p = Path(dataset_path)
    if not data_p.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_p}. Run `make dataset` first.")

    df = pd.read_parquet(data_p)
    data = df[df["level_label"] != "rest"].copy()
    data = data.dropna(subset=["level_label", "participant_id"])
    data["target"] = data["level_label"].astype(int)

    feature_cols = get_feature_columns(scheme=scheme, group="all")
    available_cols = [c for c in feature_cols if c in data.columns]
    feature_base_names = [c.replace(f"_{scheme}", "") for c in available_cols]

    # Compute sigma floors from pooled training data on RAW features (PRD 8.3)
    from cogload.calibration.normalize import DEFAULT_FEATURE_SIGMA_FLOORS
    sigma_floors: Dict[str, float] = {}
    for base_name in feature_base_names:
        raw_col = f"{base_name}_raw"
        if raw_col in data.columns:
            subj_sigmas = data.groupby("participant_id")[raw_col].std()
            valid_sigmas = subj_sigmas.dropna()
            floor_val = float(np.percentile(valid_sigmas, 25)) if not valid_sigmas.empty else DEFAULT_FEATURE_SIGMA_FLOORS.get(base_name, 0.05)
            sigma_floors[base_name] = max(1e-4, floor_val)
        else:
            sigma_floors[base_name] = DEFAULT_FEATURE_SIGMA_FLOORS.get(base_name, 0.05)

    # LOSO evaluation to obtain unbiased generalization metrics
    loso_eval = evaluate_loso(
        df, model_type=model_type, scheme=scheme, group="all", random_seed=cfg.random_seed
    )

    # Train final production model on all subjects
    X = data[available_cols].to_numpy(dtype=np.float64, copy=True)
    y = data["target"].to_numpy(dtype=int, copy=True)

    # Impute training medians
    train_medians = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
    for c_i in range(X.shape[1]):
        X[np.isnan(X[:, c_i]), c_i] = train_medians[c_i]

    model = create_model(model_type, random_seed=cfg.random_seed)
    model.fit(X, y)

    # Export
    out_dir = export_model_card(
        model=model,
        output_dir=output_dir,
        model_name=f"{model_type.upper()}_CognitiveLoad_{scheme}",
        feature_names=feature_base_names,
        classes=["LOW", "MEDIUM", "HIGH"],
        metrics={
            "loso_macro_f1_mean": loso_eval["f1_macro_mean"],
            "loso_macro_f1_std": loso_eval["f1_macro_std"],
            "loso_ci_95": loso_eval["f1_ci_95"],
            "loso_accuracy_mean": loso_eval["accuracy_mean"],
            "confusion_matrix": loso_eval["pooled_confusion_matrix"],
        },
        sigma_floors=sigma_floors,
        normalization_scheme=scheme,
        training_data_summary={
            "n_windows": len(data),
            "n_participants": data["participant_id"].nunique(),
            "class_distribution": data["target"].value_counts().to_dict(),
        },
    )

    print(f"Exported trained model and model card to {out_dir}")
    return out_dir
