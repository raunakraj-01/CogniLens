"""Report generator: assembles reports/report.md with evaluation tables and figures."""

import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def generate_report(
    dataset_path: str | Path = "data/processed/windows_W10_S5.parquet",
    metrics_json: str | Path = "reports/tables/ablation_metrics.json",
    output_report: str | Path = "reports/report.md",
) -> None:
    data_p = Path(dataset_path)
    metrics_p = Path(metrics_json)
    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    has_data = data_p.exists()
    has_metrics = metrics_p.exists()

    df = pd.read_parquet(data_p) if has_data else pd.DataFrame()
    metrics = {}
    if has_metrics:
        with open(metrics_p, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    n_windows = len(df) if not df.empty else 0
    n_subjects = df["participant_id"].nunique() if not df.empty else 0

    calib_table = ""
    if "calibration" in metrics:
        calib_table = "| Normalization | LOSO Macro F1 (Mean ± Std) | Accuracy |\n|---|---|---|\n"
        for k, v in metrics["calibration"].items():
            calib_table += f"| `{k}` | {v['f1_macro_mean']:.3f} ± {v['f1_macro_std']:.3f} | {v['accuracy_mean']:.1%} |\n"

    models_table = ""
    if "models" in metrics:
        models_table = "| Model | LOSO Macro F1 (Mean ± Std) | Accuracy |\n|---|---|---|\n"
        for k, v in metrics["models"].items():
            models_table += f"| {k.upper()} | {v['f1_macro_mean']:.3f} ± {v['f1_macro_std']:.3f} | {v['accuracy_mean']:.1%} |\n"

    groups_table = ""
    if "feature_groups" in metrics:
        groups_table = "| Feature Group | LOSO Macro F1 (Mean ± Std) | Accuracy |\n|---|---|---|\n"
        for k, v in metrics["feature_groups"].items():
            groups_table += f"| `{k}` | {v['f1_macro_mean']:.3f} ± {v['f1_macro_std']:.3f} | {v['accuracy_mean']:.1%} |\n"

    wilcoxon_str = ""
    if metrics.get("wilcoxon_zbase_vs_raw"):
        w = metrics["wilcoxon_zbase_vs_raw"]
        wilcoxon_str = f"Wilcoxon signed-rank test (zbase vs raw): statistic={w['stat']}, p={w['p_value']:.4f}."

    binary_str = ""
    if metrics.get("binary_low_vs_high"):
        b = metrics["binary_low_vs_high"]
        binary_str = f"- Binary (Low vs High) LOSO Macro F1: **{b['f1_macro_mean']:.3f}** (Accuracy: **{b['accuracy_mean']:.1%}**)"

    within_str = ""
    if metrics.get("within_subject"):
        w = metrics["within_subject"]
        within_str = f"- Within-Subject Blockwise Reference F1: **{w['within_f1_mean']:.3f} ± {w['within_f1_std']:.3f}**"

    report_content = f"""# Empirical Evaluation Report: Real-Time Cognitive Load Estimation

**Project:** `cogload`  
**Generated at:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Scope:** Facial and Ocular Behavioral Signal Workload Modeling  

---

## 1. Executive Summary & Framing

Webcam-derived behavioral signals (Eye Aspect Ratio, blink dynamics, gaze dispersion, head motion, and facial tension) provide a contactless, cost-effective proxy for mental workload. This report presents a quantitative evaluation across controlled cognitive load conditions.

> **Critical Disclaimer:** Behavioral signals from a webcam are proxies for cognitive workload and not direct measurements of neural activity. The system classifies an estimated workload level, not cognitive ability or identity, and is not a medical device.

---

## 2. Dataset Summary

- **Total Analyzed Windows:** {n_windows} (10 s sliding windows, 5 s stride)
- **Number of Participants:** {n_subjects}
- **Condition Distribution:**
{df['level_label'].value_counts().to_string() if not df.empty else 'No data'}

---

## 3. Quantitative Results & Ablations

### 3.1 Personal Baseline Calibration Ablation (Central Project Claim)
Comparing uncalibrated raw features against personalized z-score baseline (`zbase`) and ratio normalization:

{calib_table}

{wilcoxon_str}

### 3.2 Model Comparison (Cross-Subject LOSO)
{models_table}

### 3.3 Feature Group Ablations
{groups_table}

### 3.4 Binary vs 3-Class & Within-Subject Reference
{binary_str}
{within_str}

---

## 4. Evaluation Figures

- **Calibration Ablation:** `figures/ablation_calibration.png`
- **Model Architecture Comparison:** `figures/ablation_models.png`
- **Feature Group Impact:** `figures/ablation_features.png`
- **Confusion Matrix:** `figures/confusion_matrix.png`

---

## 5. Honest Limitations and Responsible Use

1. **Behavioral Proxy Nature:** Blink rates and gaze fixations are influenced by eye dryness, lighting, screen brightness, and fatigue in addition to workload.
2. **Subject Disjointness:** Leave-One-Subject-Out guarantees no leakage across participant identities.
3. **Responsible Deployment:** Intended strictly for human-in-the-loop adaptive systems and educational research. Unacceptable uses include automated hiring decisions, punitive employee surveillance, and exam proctoring.
"""

    out_p.write_text(report_content, encoding="utf-8")
    print(f"Report generated at {out_p}")


if __name__ == "__main__":
    generate_report()
