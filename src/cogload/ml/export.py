"""Model export and Model Card metadata generation (PRD Section 6.6 & 10.4)."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from cogload.features.names import FEATURE_NAMES


def export_model_card(
    model: BaseEstimator,
    output_dir: str | Path,
    model_name: str,
    feature_names: List[str],
    classes: List[str | int],
    metrics: Dict[str, Any],
    sigma_floors: Optional[Dict[str, float]] = None,
    normalization_scheme: str = "zbase",
    training_data_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model.joblib and model_card.json to output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model artifact
    model_file = out_dir / "model.joblib"
    joblib.dump(model, model_file)

    # Save model card
    card = {
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": [str(c) for c in classes],
        "feature_names": feature_names,
        "normalization_scheme": normalization_scheme,
        "sigma_floors": sigma_floors or {},
        "metrics": metrics,
        "training_data_summary": training_data_summary or {},
        "proxy_disclaimer": (
            "Webcam-derived behavioral signals are proxies for cognitive workload and not "
            "direct measurements of neural activity. Not a medical or diagnostic device."
        ),
    }

    card_file = out_dir / "model_card.json"
    with open(card_file, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    return out_dir


def load_exported_model(model_dir: str | Path) -> tuple[BaseEstimator, Dict[str, Any]]:
    """Load model artifact and model card from directory."""
    m_dir = Path(model_dir)
    model_file = m_dir / "model.joblib"
    card_file = m_dir / "model_card.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}")
    if not card_file.exists():
        raise FileNotFoundError(f"Model card not found at {card_file}")

    model = joblib.load(model_file)
    with open(card_file, "r", encoding="utf-8") as f:
        card = json.load(f)

    return model, card
