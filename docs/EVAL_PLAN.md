# Evaluation Plan & Integrity Protocol

## Primary Evaluation Strategy
1. **Leave-One-Subject-Out (LOSO) Cross-Validation:**
   - Primary metric: Macro F1 across held-out subjects.
   - Guard against leakage: Held-out subject is never seen in training, scaler fitting, or sigma_floor computation.
2. **Within-Subject (Blockwise CV):**
   - Train on subset of blocks, evaluate on held-out blocks of the same user.
   - Never randomly split adjacent sliding windows (prevents inflation via near-duplicate frames).
3. **Ablation Studies:**
   - Calibration normalization: `raw` vs `zbase` vs `ratio`.
   - Feature subsets: Eye-only vs Gaze-only vs Head-pose vs All.
   - Window size: 5s, 10s, 20s, 30s.
   - Model comparison: Logistic Regression vs Random Forest vs XGBoost.
4. **Subjective Ground-Truth Validation:**
   - Correlation between predicted cognitive load score and post-block NASA-TLX scores.
