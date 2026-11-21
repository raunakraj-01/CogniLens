"""CLI runner for evaluation suite, ablations, and report figures (PRD Section 11 & 17)."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from cogload.ml.evaluate import evaluate_loso, run_all_ablations
from cogload.ml.train import train_and_export_model


def plot_confusion_matrix(cm_data: list, title: str, output_file: Path) -> None:
    """Plot and save confusion matrix figure."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    cm_arr = np.array(cm_data)
    cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
    labels = ["Low", "Med", "High"] if len(cm_data) == 3 else ["Low", "High"]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def plot_ablation_bars(ablation_dict: dict, title: str, output_file: Path) -> None:
    """Plot bar chart for ablation comparisons."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    keys = list(ablation_dict.keys())
    means = [ablation_dict[k]["f1_macro_mean"] for k in keys]
    stds = [ablation_dict[k]["f1_macro_std"] for k in keys]

    plt.bar(keys, means, yerr=stds, capsize=5, color="#2b5c8f", alpha=0.85)
    plt.axhline(0.33, color="red", linestyle="--", label="Chance (3-class)")
    plt.ylabel("LOSO Macro F1")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate cognitive load models")
    parser.add_argument("--dataset", type=str, default="data/processed/windows_W10_S5.parquet")
    parser.add_argument("--train-only", action="store_true", help="Only train and export model")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    # Train and export primary model
    train_and_export_model(dataset_path=args.dataset, output_dir="models/best")
    if args.train_only:
        return

    df = pd.read_parquet(args.dataset)
    print("Running full ablation suite...")
    ablations = run_all_ablations(df)

    reports_p = Path(args.reports_dir)
    fig_dir = reports_p / "figures"
    tab_dir = reports_p / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(tab_dir / "ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy(ablations), f, indent=2)

    # Plot Figures
    # 1. Calibration Ablation
    plot_ablation_bars(
        ablations["calibration"],
        "Calibration Normalization Impact (LOSO)",
        fig_dir / "ablation_calibration.png",
    )

    # 2. Model Comparison
    plot_ablation_bars(
        ablations["models"],
        "Model Architecture Comparison (LOSO)",
        fig_dir / "ablation_models.png",
    )

    # 3. Feature Groups
    plot_ablation_bars(
        ablations["feature_groups"],
        "Feature Group Ablation (LOSO)",
        fig_dir / "ablation_features.png",
    )

    # 4. Confusion Matrix for Best Model (RF + zbase)
    cm = ablations["calibration"]["zbase"]["pooled_confusion_matrix"]
    plot_confusion_matrix(cm, "LOSO Confusion Matrix (RF + zbase)", fig_dir / "confusion_matrix.png")

    print(f"Evaluation figures and tables generated in {reports_p}")


if __name__ == "__main__":
    main()
