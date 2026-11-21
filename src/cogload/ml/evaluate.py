"""Rigorous Leave-One-Subject-Out (LOSO) and ablation evaluation (PRD Section 11)."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from scipy.stats import wilcoxon
from cogload.ml.models import create_model, get_feature_columns


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, binary: bool = False
) -> Dict[str, Any]:
    """Compute classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    out: Dict[str, Any] = {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
    }

    if binary and y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            # For binary, use probability of positive class
            p_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            out["roc_auc"] = float(roc_auc_score(y_true, p_pos))
        except Exception:
            out["roc_auc"] = 0.5

    return out


def evaluate_loso(
    df: pd.DataFrame,
    model_type: str = "rf",
    scheme: str = "zbase",
    group: str = "all",
    binary: bool = False,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Leave-One-Subject-Out cross-validation across participants (FR-ML-3 & Section 11.2).

    Guarantees no leakage:
    Held-out subject is never in the training set of any fold.
    """
    # Filter dataset
    data = df[df["level_label"] != "rest"].copy()
    data = data.dropna(subset=["level_label", "participant_id"])
    data["level_label"] = data["level_label"].astype(int)

    if binary:
        # Low (0) vs High (2), drop Medium (1)
        data = data[data["level_label"].isin([0, 2])].copy()
        data["target"] = (data["level_label"] == 2).astype(int)
    else:
        data["target"] = data["level_label"].astype(int)

    feature_cols = get_feature_columns(scheme=scheme, group=group)
    available_cols = [c for c in feature_cols if c in data.columns]
    if not available_cols:
        raise ValueError(f"No matching feature columns found in dataset for scheme={scheme}, group={group}")

    participants = sorted(data["participant_id"].unique())
    if len(participants) < 2:
        raise ValueError(f"LOSO requires at least 2 participants. Got: {len(participants)}")

    subject_metrics: List[Dict[str, Any]] = []
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for held_out in participants:
        train_df = data[data["participant_id"] != held_out]
        test_df = data[data["participant_id"] == held_out]

        if test_df.empty or train_df.empty:
            continue

        X_train = train_df[available_cols].to_numpy(dtype=np.float64, copy=True)
        y_train = train_df["target"].to_numpy(dtype=int, copy=True)

        X_test = test_df[available_cols].to_numpy(dtype=np.float64, copy=True)
        y_test = test_df["target"].to_numpy(dtype=int, copy=True)

        # Impute missing values if any with training medians
        train_medians = np.nanmedian(X_train, axis=0)
        # In case all nan for a col
        train_medians = np.nan_to_num(train_medians, nan=0.0)

        for col_i in range(X_train.shape[1]):
            nan_mask_tr = np.isnan(X_train[:, col_i])
            nan_mask_te = np.isnan(X_test[:, col_i])
            X_train[nan_mask_tr, col_i] = train_medians[col_i]
            X_test[nan_mask_te, col_i] = train_medians[col_i]

        model = create_model(model_type=model_type, random_seed=random_seed, binary=binary)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        fold_metric = compute_metrics(y_test, y_pred, y_proba, binary=binary)
        fold_metric["participant_id"] = held_out
        fold_metric["n_windows"] = len(test_df)
        subject_metrics.append(fold_metric)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Macro averages across subjects (PRD Section 11)
    f1_scores = [m["f1_macro"] for m in subject_metrics]
    acc_scores = [m["accuracy"] for m in subject_metrics]

    f1_mean = float(np.mean(f1_scores)) if f1_scores else 0.0
    f1_std = float(np.std(f1_scores)) if len(f1_scores) > 1 else 0.0
    acc_mean = float(np.mean(acc_scores)) if acc_scores else 0.0

    # 95% Bootstrap confidence interval over subjects
    n_bootstrap = 1000
    rng = np.random.RandomState(random_seed)
    boot_means = []
    if len(f1_scores) >= 3:
        for _ in range(n_bootstrap):
            sample = rng.choice(f1_scores, size=len(f1_scores), replace=True)
            boot_means.append(float(np.mean(sample)))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ci_lower = f1_mean
        ci_upper = f1_mean

    return {
        "model_type": model_type,
        "scheme": scheme,
        "group": group,
        "binary": binary,
        "n_subjects": len(subject_metrics),
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "f1_ci_95": [ci_lower, ci_upper],
        "accuracy_mean": acc_mean,
        "subject_metrics": subject_metrics,
        "pooled_confusion_matrix": confusion_matrix(all_y_true, all_y_pred).tolist(),
    }


def evaluate_within_subject(
    df: pd.DataFrame,
    model_type: str = "rf",
    scheme: str = "zbase",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Within-subject blockwise evaluation (PRD Section 11.2).

    Trains on subset of blocks and tests on held-out blocks of the same subject.
    Never randomly splits adjacent windows.
    """
    data = df[df["level_label"] != "rest"].copy()
    data = data.dropna(subset=["level_label", "participant_id", "block_id"])
    data["target"] = data["level_label"].astype(int)

    feature_cols = get_feature_columns(scheme=scheme, group="all")
    available_cols = [c for c in feature_cols if c in data.columns]

    subject_results = []
    for pid, group in data.groupby("participant_id"):
        blocks = sorted(group["block_id"].unique())
        if len(blocks) < 2:
            continue

        # Leave one block out for this subject
        fold_f1s = []
        for held_block in blocks:
            train_g = group[group["block_id"] != held_block]
            test_g = group[group["block_id"] == held_block]
            if train_g.empty or test_g.empty:
                continue

            X_tr = train_g[available_cols].to_numpy(dtype=np.float64, copy=True)
            y_tr = train_g["target"].to_numpy(dtype=int, copy=True)
            X_te = test_g[available_cols].to_numpy(dtype=np.float64, copy=True)
            y_te = test_g["target"].to_numpy(dtype=int, copy=True)

            # Impute
            meds = np.nan_to_num(np.nanmedian(X_tr, axis=0), nan=0.0)
            for c_i in range(X_tr.shape[1]):
                X_tr[np.isnan(X_tr[:, c_i]), c_i] = meds[c_i]
                X_te[np.isnan(X_te[:, c_i]), c_i] = meds[c_i]

            model = create_model(model_type, random_seed=random_seed)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
            fold_f1s.append(f1)

        if fold_f1s:
            subject_results.append({
                "participant_id": pid,
                "within_f1": float(np.mean(fold_f1s)),
            })

    within_f1s = [r["within_f1"] for r in subject_results]
    return {
        "within_f1_mean": float(np.mean(within_f1s)) if within_f1s else 0.0,
        "within_f1_std": float(np.std(within_f1s)) if len(within_f1s) > 1 else 0.0,
        "subject_results": subject_results,
    }


def run_all_ablations(df: pd.DataFrame, random_seed: int = 42) -> Dict[str, Any]:
    """Run full suite of ablations defined in PRD Section 11.3."""
    results: Dict[str, Any] = {}

    # 1. Calibration Ablation (raw vs zbase vs ratio)
    calib_results = {}
    for sch in ["raw", "zbase", "ratio"]:
        calib_results[sch] = evaluate_loso(df, model_type="rf", scheme=sch, random_seed=random_seed)
    results["calibration"] = calib_results

    # Wilcoxon signed-rank test comparing zbase vs raw per-subject F1
    raw_f1s = [m["f1_macro"] for m in calib_results["raw"]["subject_metrics"]]
    zbase_f1s = [m["f1_macro"] for m in calib_results["zbase"]["subject_metrics"]]
    if len(raw_f1s) >= 5 and raw_f1s != zbase_f1s:
        try:
            stat, pval = wilcoxon(zbase_f1s, raw_f1s)
            results["wilcoxon_zbase_vs_raw"] = {"stat": float(stat), "p_value": float(pval)}
        except Exception:
            results["wilcoxon_zbase_vs_raw"] = None
    else:
        results["wilcoxon_zbase_vs_raw"] = None

    # 2. Model Comparison (majority vs lr vs rf vs xgb)
    model_results = {}
    for m in ["majority", "lr", "rf", "xgb"]:
        model_results[m] = evaluate_loso(df, model_type=m, scheme="zbase", random_seed=random_seed)
    results["models"] = model_results

    # 3. Feature Group Ablations (eye vs gaze vs head vs tension vs all)
    group_results = {}
    for g in ["eye", "gaze", "head", "tension", "all"]:
        group_results[g] = evaluate_loso(df, model_type="rf", scheme="zbase", group=g, random_seed=random_seed)
    results["feature_groups"] = group_results

    # 4. Binary (Low vs High) vs 3-Class
    binary_res = evaluate_loso(df, model_type="rf", scheme="zbase", binary=True, random_seed=random_seed)
    results["binary_low_vs_high"] = binary_res

    # 5. Within-Subject Reference
    within_res = evaluate_within_subject(df, model_type="rf", scheme="zbase", random_seed=random_seed)
    results["within_subject"] = within_res

    return results
