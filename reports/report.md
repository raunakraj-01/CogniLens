# Empirical Evaluation Report: Real-Time Cognitive Load Estimation

**Project:** `cogload`  
**Generated at:** 2026-11-21 05:25:43 UTC  
**Scope:** Facial and Ocular Behavioral Signal Workload Modeling  

---

## 1. Executive Summary & Framing

Webcam-derived behavioral signals (Eye Aspect Ratio, blink dynamics, gaze dispersion, head motion, and facial tension) provide a contactless, cost-effective proxy for mental workload. This report presents a quantitative evaluation across controlled cognitive load conditions.

> **Critical Disclaimer:** Behavioral signals from a webcam are proxies for cognitive workload and not direct measurements of neural activity. The system classifies an estimated workload level, not cognitive ability or identity, and is not a medical device.

---

## 2. Dataset Summary

- **Total Analyzed Windows:** 45 (10 s sliding windows, 5 s stride)
- **Number of Participants:** 5
- **Condition Distribution:**
level_label
1    15
2    15
0    15

---

## 3. Quantitative Results & Ablations

### 3.1 Personal Baseline Calibration Ablation (Central Project Claim)
Comparing uncalibrated raw features against personalized z-score baseline (`zbase`) and ratio normalization:

| Normalization | LOSO Macro F1 (Mean ± Std) | Accuracy |
|---|---|---|
| `raw` | 1.000 ± 0.000 | 100.0% |
| `zbase` | 0.659 ± 0.094 | 68.9% |
| `ratio` | 1.000 ± 0.000 | 100.0% |


Wilcoxon signed-rank test (zbase vs raw): statistic=0.0, p=0.0625.

### 3.2 Model Comparison (Cross-Subject LOSO)
| Model | LOSO Macro F1 (Mean ± Std) | Accuracy |
|---|---|---|
| MAJORITY | 0.167 ± 0.000 | 33.3% |
| LR | 0.662 ± 0.121 | 71.1% |
| RF | 0.659 ± 0.094 | 68.9% |
| XGB | 0.687 ± 0.102 | 71.1% |


### 3.3 Feature Group Ablations
| Feature Group | LOSO Macro F1 (Mean ± Std) | Accuracy |
|---|---|---|
| `eye` | 0.523 ± 0.125 | 55.6% |
| `gaze` | 0.247 ± 0.154 | 40.0% |
| `head` | 0.167 ± 0.000 | 33.3% |
| `tension` | 0.711 ± 0.106 | 73.3% |
| `all` | 0.659 ± 0.094 | 68.9% |


### 3.4 Binary vs 3-Class & Within-Subject Reference
- Binary (Low vs High) LOSO Macro F1: **1.000** (Accuracy: **100.0%**)
- Within-Subject Blockwise Reference F1: **0.000 ± 0.000**

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
