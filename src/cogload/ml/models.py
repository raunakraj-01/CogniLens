"""Model definitions for baseline, Random Forest, and Gradient Boosting (PRD Section 10.2)."""

from typing import List, Optional
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cogload.features.names import FEATURE_NAMES, FEATURE_GROUPS

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def get_feature_columns(
    scheme: str = "zbase", group: str = "all"
) -> List[str]:
    """Get feature column names for dataset given normalization scheme and feature group."""
    if group == "all":
        base_features = FEATURE_NAMES
    elif group in FEATURE_GROUPS:
        base_features = FEATURE_GROUPS[group]
    else:
        raise ValueError(f"Unknown feature group: {group}. Available: {list(FEATURE_GROUPS.keys())} or 'all'")

    suffix = f"_{scheme}" if scheme in ["raw", "zbase", "ratio"] else "_zbase"
    return [f"{f}{suffix}" for f in base_features]


def create_model(
    model_type: str = "rf",
    random_seed: int = 42,
    binary: bool = False,
) -> BaseEstimator:
    """Factory creating sklearn-compatible estimator for 3-class or binary classification."""
    model_type = model_type.lower()

    if model_type == "majority":
        return DummyClassifier(strategy="most_frequent")

    elif model_type in ["lr", "logistic"]:
        # Standardized Logistic Regression baseline
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ])

    elif model_type in ["rf", "random_forest"]:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            max_features=0.4,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=1,
        )

    elif model_type in ["xgb", "xgboost"]:
        if HAS_XGBOOST:
            objective = "binary:logistic" if binary else "multi:softprob"
            return XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective=objective,
                random_state=random_seed,
                eval_metric="logloss" if binary else "mlogloss",
            )
        else:
            # Fallback to high-performance scikit-learn histogram gradient boosting
            return HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_seed,
            )

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'majority', 'lr', 'rf', 'xgb'.")
