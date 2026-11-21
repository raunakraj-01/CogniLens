# PRD: Real-Time Cognitive Load Estimation via Facial and Ocular Behavioral Signals

| | |
|---|---|
| **Project codename** | `cogload` |
| **Document version** | 1.0 |
| **Team** | Neeraj Gannamani, Rishabh Jain, Raunak Raj (RVCE, CSE) |
| **Status** | Ready for implementation |
| **Primary audience** | AI coding agent (Antigravity or similar) + the 3 student developers |

---

## 0. How to use this document (READ FIRST, agent)

You are an AI coding agent implementing this project. Follow these rules:

1. **Build in phase order** (Section 16). Do not start Phase N+1 until every acceptance criterion of Phase N passes.
2. **Everything must be runnable and testable without a webcam.** Cloud and CI environments have no camera. Every component must work from (a) a video file, (b) a recorded-signal replay, or (c) the synthetic signal generator (FR-SYN). Only the final live demo needs a real webcam.
3. **Do not invent scope.** If it is not in Sections 3 to 15, do not build it. Put ideas in `docs/IDEAS.md` instead.
4. **Make thresholds and constants configurable** in `config.yaml`. Never hard-code magic numbers in logic.
5. **Write tests with the code.** Each module ships with unit tests in `tests/`. A phase is not done without passing tests.
6. **Keep the ML honest.** No data leakage. Subject-wise splits only. No tuning on the held-out subject. See Section 11.
7. **Privacy by default.** Never write raw video frames or images to disk. See Section 13.2.
8. **Ask the humans** (flag in `docs/QUESTIONS.md`) rather than guessing when a decision in Section 20 is blocking. Tasks tagged **[HUMAN]** need real people (data collection, ethics approval) and cannot be done by you; prepare everything they need and stop.
9. At the end of each phase, update `docs/PROGRESS.md` (what was done, what is left, known issues).
10. Prefer boring, widely used libraries. Pin versions in `requirements.txt`.

---

## 1. Overview

### 1.1 Problem
Cognitive load (the mental effort a person is investing at a given moment) matters for adaptive learning, safety monitoring (drivers, operators), and human-computer interaction. The usual way to measure it is EEG, fNIRS, or other physiological sensors, which are costly, intrusive, and impractical outside labs.

### 1.2 Proposed solution
A **webcam-only** system that:
1. Extracts facial landmarks in real time using computer vision.
2. Derives behavioral features: Eye Aspect Ratio (EAR), blink statistics, gaze movement, and head-pose variation.
3. **Personalizes** by measuring each user's baseline in a short calibration and expressing features as deviations from that baseline.
4. Classifies cognitive-load state (Low / Medium / High) using classical ML (Random Forest, XGBoost).
5. Shows the live estimate and supporting indicators on a Flask dashboard.
6. Is evaluated on labeled data from controlled cognitive tasks of varying difficulty, with NASA-TLX as a subjective cross-check.

### 1.3 Important framing (must appear in the UI, README, and paper)
Webcam-derived signals are **behavioral proxies** for cognitive load, **not** direct measurements of neural activity. The system estimates a load *level*, not a person's thoughts or abilities, and is not a medical or diagnostic device.

### 1.4 Goals
- G1. End-to-end working system: capture, features, calibration, classifier, live dashboard.
- G2. A reproducible feature-extraction pipeline that works offline (from recordings) and online (live) with **identical code**.
- G3. A rigorous evaluation: within-subject vs cross-subject, calibrated vs uncalibrated, environment variation.
- G4. A portfolio-grade, well-documented repo and a paper-ready results set.

### 1.5 Non-goals (do NOT build)
- Pupil diameter measurement (not reliable from a consumer webcam).
- Emotion recognition, identity recognition, or demographic inference.
- Mobile app, cloud deployment, multi-user server, or accounts/authentication.
- Deep end-to-end video models (CNN/ViT on raw frames). Stretch only, see 3.2.
- Any medical claim.

---

## 2. Users and use cases

| Persona | Use case |
|---|---|
| **Participant** | Gives consent, does calibration, performs tasks while being recorded. |
| **Experimenter** (team member) | Runs data-collection sessions, exports data, trains and evaluates models. |
| **Demo user** | Sits at the webcam, calibrates for about 1 minute, watches the live load estimate while doing a task. |
| **Reviewer / faculty** | Reads the evaluation report and runs the demo. |

---

## 3. Scope

### 3.1 MVP (must ship)
1. Webcam and video-file capture, MediaPipe landmark extraction.
2. Per-frame signals: EAR (L/R/mean), iris-based gaze ratios, head pose (yaw/pitch/roll).
3. Windowed features (blink stats, PERCLOS, gaze dispersion and velocity, head motion).
4. Personalized calibration (about 90 s) and baseline normalization.
5. Data-collection mode with an in-browser **n-back** task at three difficulty levels, synchronized event logging, and NASA-TLX questionnaire.
6. Dataset builder, model training (Logistic Regression baseline, Random Forest, XGBoost), leave-one-subject-out evaluation.
7. Real-time inference engine and Flask dashboard.
8. Evaluation report with metrics, confusion matrices, and ablations.

### 3.2 Stretch (only after MVP is complete and tested)
- S1. Second task family (mental arithmetic with click-to-answer) for cross-task generalization.
- S2. Light sequence model (1D-CNN or small LSTM) on windowed signals, compared to RF/XGBoost.
- S3. Few-shot personalization: fine-tune or re-weight with a few minutes of the user's labeled data.
- S4. Brow-furrow and mouth-tension features.
- S5. Model explainability (SHAP) panel in the report.

---

## 4. System architecture

```
                 ┌────────────────────────── ONLINE (live) ──────────────────────────┐
 Webcam / video  │                                                                    │
   source  ───►  │ Capture ─► Landmarks ─► Per-frame  ─► Ring buffer ─► Window  ─►    │
 (OpenCV)        │ thread     (MediaPipe)   signals       (N seconds)    features     │
                 │                          (EAR, gaze,                    │           │
                 │                           head pose)                    ▼           │
                 │                                              Baseline normalizer ◄─┤◄── Calibration profile (per user)
                 │                                                         │          │
                 │                                                         ▼          │
                 │                                             Classifier (joblib)    │
                 │                                                         │          │
                 │                                              Smoothing (EMA/vote)  │
                 │                                                         ▼          │
                 │                                   Flask app ── SSE stream ──► Dashboard (browser)
                 └────────────────────────────────────────────────────────────────────┘

                 ┌────────────────────── OFFLINE (research) ──────────────────────────┐
 Session folders ─► Window features (SAME code) ─► Dataset (windows.parquet) ─► Train / LOSO eval ─► Reports
 (per-frame signals + events + NASA-TLX)
                 └────────────────────────────────────────────────────────────────────┘
```

**Key design decision:** the system records **per-frame low-level signals** (not just windowed features) during data collection. This lets the team change window length, features, and calibration strategy later **without re-recording participants**. The same `features/` code path serves both online and offline use.

---

## 5. Tech stack

| Area | Choice | Notes |
|---|---|---|
| Language | Python 3.10 to 3.12 | MediaPipe wheels support these |
| Video | OpenCV (`opencv-python`) | capture, video-file input, optional debug overlay |
| Landmarks | MediaPipe **Face Landmarker** (Tasks API) | 478 landmarks including 10 iris points; provides facial transformation matrix. Fallback: legacy `mp.solutions.face_mesh(refine_landmarks=True)` if the Tasks API is unavailable. |
| Numerics | NumPy, pandas, SciPy | |
| ML | scikit-learn, XGBoost, joblib | |
| Web | Flask, Server-Sent Events (SSE) for live updates | no Socket.IO needed |
| Frontend | Vanilla JS + Chart.js (vendored locally, no CDN dependency at demo time) | keep simple |
| Config | PyYAML | `config.yaml` |
| Storage | CSV (events, TLX) and Parquet (signals, windows) via `pyarrow` | |
| Testing | pytest | |
| Lint/format | ruff + black | |
| Plots | matplotlib, seaborn | evaluation figures |

The webcam is captured **server-side** by OpenCV (the app runs on the user's laptop). The browser is only for the dashboard and the task runner.

---

## 6. Functional requirements

### 6.1 Capture and landmarks
- **FR-CAP-1** Accept input from: webcam index, video file path, or replay of recorded signals.
- **FR-CAP-2** Run capture in its own thread with a bounded queue; drop frames (never block) when the pipeline is slower than the camera. Log dropped frames.
- **FR-CAP-3** Report effective FPS and timestamp every frame using a monotonic clock (video-file mode uses frame index / video FPS).
- **FR-LM-1** For each frame, return landmarks (normalized and pixel coordinates) or a `face_detected = False` flag.
- **FR-LM-2** Handle: no face, multiple faces (use the largest), partial occlusion (mark frame invalid).

### 6.2 Per-frame signals (Section 7.1)
- **FR-SIG-1** Compute EAR for each eye and their mean.
- **FR-SIG-2** Compute horizontal and vertical gaze ratios from iris and eye geometry.
- **FR-SIG-3** Compute head pose (yaw, pitch, roll in degrees).
- **FR-SIG-4** Emit one signal row per frame: `t, valid, ear_l, ear_r, ear, gaze_x_l, gaze_x_r, gaze_x, gaze_y_l, gaze_y_r, gaze_y, yaw, pitch, roll` (plus optional `brow`, `mouth` if S4 is built).

### 6.3 Windowed features (Section 7.2)
- **FR-FEAT-1** Detect blinks from the EAR time series.
- **FR-FEAT-2** Compute window features over a sliding window (default 10 s window, 1 s stride in live mode; 5 s stride for offline datasets).
- **FR-FEAT-3** Discard any window with less than 80% valid frames.
- **FR-FEAT-4** Same function must be used online and offline (`features.window.compute_window_features(signal_df, baseline_open_ear, cfg)`).

### 6.4 Calibration (Section 8)
- **FR-CAL-1** Guided calibration of about 90 s: 60 s relaxed fixation, 30 s easy low-load activity.
- **FR-CAL-2** Produce a `CalibrationProfile` (JSON) with: open-eye EAR baseline, per-feature baseline mean and std, gaze center offset, head-pose neutral, quality score.
- **FR-CAL-3** Reject calibration if quality is poor (too few valid frames, face lost, extreme motion) and tell the user why.
- **FR-CAL-4** Profiles are stored per participant ID and reusable within a session. Profile files contain no images.

### 6.4b Synthetic data
- **FR-SYN** A generator `scripts/make_synthetic.py` creates fake per-frame signals for N subjects with plausible class-dependent shifts and subject-specific offsets, used for tests and CI. It must be clearly labeled synthetic and never mixed with real data.

### 6.5 Data collection ([HUMAN] runs sessions, agent builds tooling)
- **FR-COL-1** Consent screen, required before any recording.
- **FR-COL-2** Session flow: consent, participant metadata, calibration, task blocks (counterbalanced), NASA-TLX after each block, debrief.
- **FR-COL-3** In-browser n-back task (letters, keypress response) at levels 1, 2, 3.
- **FR-COL-4** Log timestamped events (block start/stop, stimulus onset, response, correctness, reaction time) in the same clock domain as the signals.
- **FR-COL-5** Save each session to `data/sessions/<participant_id>/<session_id>/` (Section 9.2).
- **FR-COL-6** Per-participant metadata: pseudonymous ID, glasses (y/n), lighting tag (`bright`/`normal`/`dim`), camera name, approximate distance, notes. No names.
- **FR-COL-7** Command to delete all data for a participant ID.

### 6.6 ML
- **FR-ML-1** Build a windowed dataset from sessions with labels (Section 10.1).
- **FR-ML-2** Train baseline LR, Random Forest, XGBoost with a reproducible seed.
- **FR-ML-3** Evaluate with leave-one-subject-out (LOSO) and report metrics (Section 11).
- **FR-ML-4** Export the chosen model with `joblib` plus a `model_card.json` (feature list and order, classes, training data summary, window config, normalization scheme, date, metrics).

### 6.7 Real-time inference and dashboard
- **FR-RT-1** Load model, calibration profile; produce a load prediction every 1 s using the latest 10 s window.
- **FR-RT-2** Smooth output (EMA over probabilities, plus hysteresis to avoid flicker).
- **FR-RT-3** Stream results to the browser via SSE (Section 12).
- **FR-UI-1** Dashboard shows: current load level and confidence, a continuous load score over time, indicator tiles (blink rate, EAR, gaze dispersion, head motion), FPS, face-detected status, calibration status, and the proxy disclaimer.
- **FR-UI-2** Dashboard works when the model is missing (shows indicators only and a warning) so the vision pipeline can be demoed independently.

---

## 7. Signal and feature specification

### 7.1 Per-frame signals

**Landmark indices (MediaPipe Face Mesh / Face Landmarker, 478-point topology):**

| Item | Indices |
|---|---|
| Right eye (image-left), EAR points p1..p6 | `33, 160, 158, 133, 153, 144` |
| Left eye (image-right), EAR points p1..p6 | `362, 385, 387, 263, 373, 380` |
| Right iris (center, then 4 boundary) | `468` (center), `469, 470, 471, 472` |
| Left iris (center, then 4 boundary) | `473` (center), `474, 475, 476, 477` |
| Right eye corners (outer, inner) | `33, 133` |
| Left eye corners (inner, outer) | `362, 263` |
| Right eye upper/lower lid (for vertical gaze) | `159, 145` |
| Left eye upper/lower lid (for vertical gaze) | `386, 374` |
| Head-pose model points | nose tip `1`, chin `152`, right eye outer `33`, left eye outer `263`, mouth right `61`, mouth left `291` |

*Agent: verify these indices once against a rendered debug overlay (`scripts/debug_overlay.py`) on a sample face and record the verification in `docs/PROGRESS.md`. If any index is wrong for the installed MediaPipe version, fix it centrally in `vision/landmark_map.py`.*

**EAR (per eye):**
```
EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )
```
Use pixel coordinates (multiply normalized x by width and y by height) so aspect ratio is correct. `ear = (ear_l + ear_r) / 2`.

**Gaze ratio (per eye):**
- Horizontal: project `(iris_center - outer_corner)` onto the unit vector from outer corner to inner corner, divide by eye width. Result roughly 0..1 (0.5 = looking straight). Report centered as `gaze_x = ratio - 0.5`.
- Vertical: `(iris_center_y - upper_lid_y) / (lower_lid_y - upper_lid_y)`, centered the same way as `gaze_y`.
- Average left and right for `gaze_x`, `gaze_y`.
- Note: this measures iris position within the eye, which mixes in head rotation. That is why head pose is recorded separately and included as features.

**Head pose:**
- Use `cv2.solvePnP` with a generic 3D face model for the 6 points above and approximate camera intrinsics (focal length = image width, principal point = image center). Convert rotation to Euler angles (yaw, pitch, roll in degrees). Alternatively derive from the Face Landmarker transformation matrix. Pick one and document it; keep a unit test for each axis sign convention (yaw positive = turning to user's right, and so on, as decided in the docs).
- Unwrap angles to avoid ±180° jumps.

**Preprocessing per frame series:**
- Mark frames invalid if no face, or EAR is NaN or out of [0, 0.6].
- Linearly interpolate short invalid gaps (≤ 3 frames), otherwise leave NaN.
- Light smoothing of gaze (median filter, kernel 3) before velocity computation. **Do not smooth EAR** before blink detection.

### 7.2 Windowed features (target set, about 25 features)

All computed per window of `W` seconds (default 10 s).

| Group | Feature | Definition |
|---|---|---|
| **Eye openness** | `ear_mean`, `ear_std`, `ear_min` | stats of EAR in window |
| | `perclos` | fraction of valid frames with `ear < closure_ratio * baseline_open_ear` (default `closure_ratio` = 0.2, i.e. a P80-style closure; configurable) |
| **Blinks** | `blink_rate` | blinks per minute (count / window minutes) |
| | `blink_dur_mean`, `blink_dur_std` | blink durations (ms); 0 if no blinks |
| | `blink_count` | raw count |
| **Gaze** | `gaze_x_std`, `gaze_y_std` | dispersion |
| | `gaze_speed_mean`, `gaze_speed_p95` | magnitude of gaze velocity |
| | `saccade_rate` | count of velocity-threshold crossings per second (threshold configurable) |
| | `fixation_ratio` | fraction of time with gaze speed below fixation threshold |
| | `offscreen_ratio` | fraction of frames with \|gaze\| beyond a configurable bound (looking away) |
| **Head pose** | `yaw_std`, `pitch_std`, `roll_std` | variation |
| | `head_speed_mean` | mean angular speed (deg/s) |
| | `head_motion_energy` | sum of squared angular velocities, normalized by window length |
| **Quality** | `valid_ratio` | fraction of valid frames (used for filtering, not as a model feature) |

**Blink detection algorithm (default):**
1. Threshold `T_blink = blink_ratio * baseline_open_ear` (default `blink_ratio` = 0.75; configurable; fallback if no calibration: 0.21 absolute).
2. A blink is a run of consecutive frames with `ear < T_blink` lasting between `blink_min_ms` (default 60 ms) and `blink_max_ms` (default 600 ms). Longer runs are treated as eye closure, not blinks, and counted toward PERCLOS only.
3. Enforce a refractory period (default 100 ms) to avoid double counting.
4. Frame rate varies, so use timestamps, never frame counts, for durations.

Unit-test blink detection with synthetic EAR series (known blink counts and durations, different FPS values such as 15, 24, 30, 60).

---

## 8. Personalization and calibration

**Why:** blink rate, EAR level, gaze behavior, and head movement vary widely between people (glasses, eye shape, camera angle, habits). Absolute values are not comparable across people; deviation from one's own baseline is more informative.

### 8.1 Calibration protocol (about 90 s)
1. **Setup check (10 s):** face centered, distance, lighting (brightness histogram check), glasses glare warning.
2. **Phase A: Relaxed fixation (60 s):** look at a fixation cross at screen center, breathe normally, blink naturally. Gives `baseline_open_ear` (e.g., 90th percentile of EAR outside blinks), baseline blink rate, low gaze dispersion, neutral head pose.
3. **Phase B: Easy activity (30 s):** read a short, easy passage or do a trivial 0-back/tap task. Gives a "low load but active" baseline closer to everyday use.

### 8.2 Baseline profile
For windows computed within calibration (10 s windows, 5 s stride; about 10 windows), compute per-feature `mu_base` and `sigma_base`. Save:
```json
{
  "participant_id": "P001",
  "created_at": "...",
  "baseline_open_ear": 0.29,
  "gaze_center": {"x": 0.01, "y": -0.02},
  "head_neutral": {"yaw": 1.5, "pitch": -3.0, "roll": 0.4},
  "features": {"blink_rate": {"mu": 17.2, "sigma": 4.1}, "...": {}},
  "quality": {"valid_ratio": 0.97, "n_windows": 10, "passed": true}
}
```

### 8.3 Normalization schemes (all implemented; compared in ablation)
- **`raw`**: no normalization (uncalibrated baseline for comparison).
- **`zbase`** (default): `z = (f - mu_base) / max(sigma_base, sigma_floor)`. `sigma_floor` per feature is computed from the pooled training population (e.g., 25th percentile of per-subject sigma) to avoid blow-ups from tiny baseline variance.
- **`ratio`**: `f / max(mu_base, eps)` for strictly positive features.

Head-pose and gaze signals are also re-centered by `head_neutral` and `gaze_center` **before** feature computation when calibration is available.

### 8.4 Leakage rules
- The held-out subject's calibration windows may be used only to compute **their own** baseline (no labels, no model fitting). That mirrors real use.
- `sigma_floor` and any scaler are computed from training subjects only.

---

## 9. Data collection protocol and data schema

### 9.1 Experimental protocol ([HUMAN])
- **Target participants:** at least 15 (aim for 20+) volunteers (students). More participants matter far more than more windows per participant.
- **Ethics:** get faculty and institutional approval and written informed consent before collecting any data from anyone outside the team. Consent text template: `docs/consent_template.md` (agent drafts; humans/faculty review).
- **Session (about 30 min):**
  1. Consent and metadata (2 min)
  2. Calibration (about 1.5 min)
  3. Task blocks: **n-back levels 1, 2, 3**, 3 min each, **counterbalanced order** across participants (Latin square). Optional second family (S1) with matching three levels. A 60 s rest between blocks.
  4. After each block: **NASA-TLX** (raw/RTLX: 6 sliders, 0 to 100).
  5. Debrief.
- **Label definition (primary):** window label = difficulty level of the block it falls in (Low = 1-back, Medium = 2-back, High = 3-back). Rest and calibration windows are `rest` and excluded from the 3-class set (kept for analysis).
- **Label validation:** per participant, check that mean NASA-TLX rises with difficulty and that task accuracy drops or RT rises. Flag participants where this fails; report results with and without them (pre-registered rule in `docs/EVAL_PLAN.md`; do not remove subjects after seeing model results).
- Discard the first 15 s of each block when building windows (task-switch transient). Configurable.
- **Environmental variation:** record lighting tag, glasses, camera. Have at least a subset of participants repeat a short session in a different lighting condition or camera position for the robustness analysis.

### 9.2 On-disk layout
```
data/
  sessions/
    P001/
      S01/
        meta.json          # participant + environment metadata
        signals.parquet    # per-frame signals (Section 6.2 columns), time in seconds
        events.csv         # timestamp_s, event_type, block_id, task, level, payload_json
        nasa_tlx.csv       # block_id, mental, physical, temporal, performance, effort, frustration, raw_tlx
        calibration.json   # CalibrationProfile
  processed/
    windows_W10_S5.parquet # built dataset (one row per window)
  synthetic/               # synthetic only; never mixed with real
models/
  <model_name>/model.joblib, model_card.json
reports/
  figures/, tables/, report.md
```
`data/` is git-ignored. Never commit participant data.

### 9.3 `windows.parquet` columns
`participant_id, session_id, block_id, task, level_label (0/1/2 or 'rest'), t_start, t_end, valid_ratio, lighting, glasses, <feature columns raw>, <feature columns zbase>, nasa_tlx_block`

---

## 10. Machine learning specification

### 10.1 Problem formulation
- **Primary:** 3-class classification (Low / Medium / High).
- **Secondary:** binary (Low vs High, dropping Medium) which is usually the more realistic claim.
- **Output at runtime:** class probabilities, plus a scalar **load score** in [0, 1], computed as the probability-weighted expected level: `(P_med * 0.5 + P_high * 1.0)`.

### 10.2 Models
| Model | Role | Notes |
|---|---|---|
| Majority-class | sanity floor | |
| Logistic Regression (standardized) | simple baseline | |
| Random Forest | main candidate | `n_estimators` 300, class-balanced |
| XGBoost | main candidate | small depth (3 to 5), strong regularization |
| (S2) 1D-CNN / LSTM | stretch | |

Hyperparameters: tune with **nested, subject-wise** CV on training subjects only (e.g., `GroupKFold` inside each LOSO fold). Keep search grids small; this dataset is small.

### 10.3 Feature handling
- Feature order is fixed and stored in `model_card.json`; inference must reject mismatches.
- Impute NaN with training-set medians (stored in the model pipeline).
- Provide feature-group ablations: eye-only, gaze-only, head-only, all.

### 10.4 Runtime constraints
- Inference per window under 20 ms on a laptop CPU.
- Model artifact under 50 MB.

---

## 11. Evaluation plan

All metrics are computed per held-out subject and then averaged (macro over subjects), plus pooled; report mean and standard deviation across subjects, with bootstrap 95% confidence intervals over subjects.

### 11.1 Metrics
Accuracy, macro precision, macro recall, macro F1, confusion matrix (row-normalized and counts), per-class F1, and for binary: ROC-AUC. Also: per-subject score distribution (box plot) because averages hide failures.

### 11.2 Splits
| Setting | Purpose |
|---|---|
| **LOSO (cross-subject)** | primary result; realistic for new users |
| **Within-subject** (blockwise CV: train on some blocks of the same user, test on other blocks; never random window splits because adjacent windows are near-duplicates and inflate accuracy) | upper reference; shows personalization potential |
| **Cross-environment** | train on one lighting/camera condition, test on another (subset of subjects) |

### 11.3 Ablations (each produces a table or figure)
1. **Calibration:** `raw` vs `zbase` vs `ratio`. *This is the project's central claim; it must be tested rigorously.*
2. **Feature groups:** eye, gaze, head, all.
3. **Window length:** 5, 10, 20, 30 s.
4. **Model comparison:** LR vs RF vs XGBoost.
5. **Calibration duration:** 30, 60, 90 s (simulate by truncating calibration data).
6. **Label noise check:** correlation between predicted load score and NASA-TLX at the block level.
7. **Per-environment performance:** lighting, glasses.

### 11.4 Statistical rigor
- Paired comparisons across subjects (e.g., Wilcoxon signed-rank on per-subject F1) for calibration on vs off and model vs model.
- Report effect sizes, not only p-values. State N clearly. Do not over-claim with small N.

### 11.5 Success criteria (planning assumptions, not guarantees)
These are the team's planning targets, **not** promises about what webcam signals can deliver:
- Pipeline runs live at 20 FPS or more on a typical laptop CPU with end-to-end latency (frame to dashboard) under 1.5 s.
- LOSO macro-F1 for 3-class is clearly above chance (0.33) and above the majority baseline, with a CI that excludes it.
- `zbase` calibration improves LOSO macro-F1 over `raw` (report the result honestly even if it does not).
- Binary Low-vs-High performs better than 3-class.
Whatever the numbers are, report them with limitations. A well-evidenced modest result is publishable; an inflated one is not.

---

## 12. Dashboard, task runner, and API

### 12.1 Pages
| Route | Purpose |
|---|---|
| `/` | Live dashboard |
| `/calibrate` | Guided calibration (fixation cross, instructions, progress, quality result) |
| `/collect` | Experiment runner: consent, metadata, calibration, n-back blocks, NASA-TLX |
| `/report` | (optional) renders `reports/report.md` figures |

### 12.2 Live dashboard layout
- Header: app name, participant ID, status chips (Camera OK, Face detected, Calibrated, Model loaded), FPS.
- **Hero:** load level badge (LOW / MEDIUM / HIGH) + confidence + gauge for the load score.
- **Timeline chart:** load score over last 5 minutes (Chart.js), with block/event markers during data collection.
- **Indicator tiles** with mini sparklines: blink rate (per min), EAR, gaze dispersion, head motion, each shown with delta vs baseline (z-score) when calibrated.
- **Disclaimer strip (always visible):** "Estimates are based on behavioral signals from the webcam and are not a measurement of brain activity."
- Controls: Start/Stop, Recalibrate, Toggle debug overlay (MJPEG preview of landmarks; **off by default**, preview only, nothing saved).

### 12.3 API
| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | camera, fps, face_detected, calibrated, model info |
| GET | `/api/stream` | SSE; one JSON message per second (below) |
| POST | `/api/session/start` | body: participant metadata; starts recording signals |
| POST | `/api/session/stop` | finalizes and saves session files |
| POST | `/api/calibration/start` | begins calibration |
| GET | `/api/calibration/status` | progress, quality |
| POST | `/api/events` | task events from browser (type, block, payload) with server-side timestamp |
| POST | `/api/nasa-tlx` | TLX responses for a block |
| GET | `/api/model` | model card summary |
| DELETE | `/api/participant/<id>` | delete a participant's data (requires confirmation param) |

**SSE message:**
```json
{
  "t": 1731990000.12,
  "face_detected": true,
  "fps": 27.4,
  "calibrated": true,
  "load": {"level": "MEDIUM", "probs": {"low": 0.2, "medium": 0.55, "high": 0.25}, "score": 0.52, "confidence": 0.55},
  "indicators": {
    "blink_rate": 14.0, "ear": 0.27, "gaze_dispersion": 0.04, "head_motion": 1.8,
    "z": {"blink_rate": -0.7, "gaze_dispersion": 0.9, "head_motion": 0.3}
  },
  "warnings": []
}
```
Warnings include: `no_face`, `low_light`, `high_head_motion`, `not_calibrated`, `model_missing`.

### 12.4 n-back task spec
- Letter stimuli (consonants), 500 ms on, 2000 ms inter-stimulus interval (configurable), 30% target rate, key `Space` for match.
- Levels 1, 2, 3; 3 min per block; practice round (30 s) per level, excluded from analysis.
- Fixation cross between stimuli; large centered letters so the gaze stays near center (reduces gaze artifacts from reading position changes).
- Log every stimulus onset, response, correctness, and RT as events.

---

## 13. Non-functional requirements

### 13.1 Performance and reliability
- 20+ FPS processing on a mid-range laptop CPU (no GPU required).
- Graceful degradation: lost face does not crash the pipeline; UI shows the warning; windows with low validity are skipped.
- Camera disconnect handled with a clear UI error and clean shutdown.
- Deterministic offline processing: same inputs and seed give the same dataset and metrics.

### 13.2 Privacy, consent, and responsible use
- **No raw video or images are ever written to disk or sent over the network.** Frames are processed in memory. Only numeric signals and features are stored.
- The Flask server binds to `127.0.0.1` by default.
- Explicit consent screen; participants can stop at any time; data deletion command (FR-COL-7).
- Pseudonymous participant IDs; no names, no USN, no emails in data files. The ID mapping (if any) is kept by the experimenter outside the repo.
- Facial landmark time series are still biometric-adjacent data; state this in the consent text and README.
- Document intended use (research, education, demo) and **inappropriate uses** (employee surveillance, exam proctoring decisions, hiring, medical diagnosis). Include a "Responsible Use" section in the README and the paper.
- The UI must always show the proxy disclaimer.

### 13.3 Maintainability
- Type hints, docstrings on public functions, `ruff`/`black` clean.
- One-command setup and run (Section 17).
- README with architecture, setup, usage, and the limitations.

---

## 14. Repository structure

```
cogload/
├── PRD.md                      # this file
├── README.md
├── AGENTS.md                   # short rules for coding agents (copy of Section 0)
├── config.yaml
├── requirements.txt
├── pyproject.toml              # ruff/black/pytest config
├── Makefile                    # make setup | test | run | collect | train | eval
├── docs/
│   ├── PROGRESS.md
│   ├── QUESTIONS.md
│   ├── IDEAS.md
│   ├── EVAL_PLAN.md
│   ├── consent_template.md
│   └── architecture.md
├── src/cogload/
│   ├── config.py               # typed config loader
│   ├── capture/
│   │   ├── source.py           # webcam / video file / replay
│   │   └── thread.py           # threaded capture, bounded queue
│   ├── vision/
│   │   ├── landmarker.py       # MediaPipe wrapper
│   │   └── landmark_map.py     # all landmark indices
│   ├── signals/
│   │   ├── ear.py
│   │   ├── gaze.py
│   │   ├── head_pose.py
│   │   └── frame_signals.py    # landmarks -> per-frame signal row
│   ├── features/
│   │   ├── blinks.py
│   │   ├── window.py           # compute_window_features(...)
│   │   └── names.py            # canonical feature list and order
│   ├── calibration/
│   │   ├── protocol.py
│   │   ├── profile.py          # CalibrationProfile dataclass + IO
│   │   └── normalize.py        # raw / zbase / ratio
│   ├── data/
│   │   ├── recorder.py         # writes signals/events
│   │   ├── schema.py
│   │   └── build_dataset.py
│   ├── ml/
│   │   ├── models.py
│   │   ├── train.py
│   │   ├── evaluate.py         # LOSO, within-subject, ablations
│   │   └── export.py           # model + model_card
│   ├── realtime/
│   │   ├── engine.py           # pipeline orchestration
│   │   └── smoothing.py
│   └── app/
│       ├── server.py           # Flask app factory
│       ├── routes.py
│       ├── templates/ (index.html, calibrate.html, collect.html)
│       └── static/ (css, js, vendor/chart.min.js, nback.js)
├── scripts/
│   ├── make_synthetic.py
│   ├── debug_overlay.py
│   ├── process_video.py        # video -> signals.parquet
│   ├── run_eval.py
│   └── make_report.py
├── tests/
├── data/                       # git-ignored
├── models/
└── reports/
```

---

## 15. Configuration (`config.yaml` defaults)

```yaml
camera:
  index: 0
  width: 640
  height: 480
  target_fps: 30
processing:
  max_queue: 2
  interpolate_gap_frames: 3
features:
  window_sec: 10
  stride_sec_live: 1
  stride_sec_offline: 5
  min_valid_ratio: 0.8
  drop_block_start_sec: 15
  blink_ratio: 0.75
  blink_min_ms: 60
  blink_max_ms: 600
  blink_refractory_ms: 100
  closure_ratio: 0.2
  saccade_speed_thresh: 0.6      # gaze-units/sec; tune on data
  fixation_speed_thresh: 0.15
  offscreen_gaze_bound: 0.35
calibration:
  phase_a_sec: 60
  phase_b_sec: 30
  min_valid_ratio: 0.85
  normalization: zbase           # raw | zbase | ratio
model:
  path: models/best
  smoothing_alpha: 0.4
  hysteresis_margin: 0.1
server:
  host: 127.0.0.1
  port: 5000
  debug_preview: false
random_seed: 42
```
Thresholds marked "tune on data" must be tuned on **training data only** and recorded in `docs/EVAL_PLAN.md`.

---

## 16. Implementation plan (phases, tasks, acceptance criteria)

Effort tags: S (about 0.5 day), M (1 to 2 days), L (3 to 5 days). Adjust to the team's calendar.

### Phase 0: Project scaffold (S)
- Create repo structure, `requirements.txt` (pinned), `pyproject.toml`, `Makefile`, `config.yaml`, typed config loader, `AGENTS.md`, `docs/` stubs, `.gitignore` (data, models, `__pycache__`, `.venv`).
- **Accept:** `make setup && make test` passes on a clean machine; `python -c "import cogload"` works.

### Phase 1: Capture and landmarks (M)
- Implement `capture/source.py` (webcam, video file) and threaded capture with bounded queue and FPS counter.
- Implement MediaPipe Face Landmarker wrapper; handle no-face and multi-face.
- `scripts/debug_overlay.py`: draw eye contours, iris, and landmark indices on a video or webcam; save screenshots for verification (screenshots of synthetic or consenting-team faces only).
- **Accept:** processes a sample video at 20+ FPS on CPU; returns landmarks for at least 95% of frames of a clean frontal video; index table in 7.1 verified visually and noted in PROGRESS.md; tests cover no-face and dropped-frame behavior.

### Phase 2: Per-frame signals (M)
- Implement EAR, gaze ratios, head pose, and `frame_signals.py` producing the row in FR-SIG-4.
- Preprocessing (validity, interpolation, unwrap).
- `scripts/process_video.py`: video to `signals.parquet`.
- **Accept:** unit tests on synthetic landmark geometries: EAR matches hand-computed values; gaze ratio is 0.5 for centered iris and monotonic as the iris moves; head-pose axis signs match documented convention on a synthetic rotation; end-to-end run on a sample video shows an EAR trace with visible blinks.

### Phase 3: Blink detection and windowed features (M)
- Implement blink detector, all features in 7.2, and `compute_window_features`.
- Feature names and order in `features/names.py`.
- **Accept:** tests with synthetic EAR at 15/24/30/60 FPS recover blink count within ±0 for clean signals and durations within ±1 frame; window filter drops windows with under 80% valid frames; identical output when run on the same signals via offline and streaming paths (test).

### Phase 4: Calibration (M)
- Implement `CalibrationProfile`, protocol logic, quality checks, and three normalization schemes.
- Calibration page UI (`/calibrate`) with fixation cross, countdown, progress, and pass/fail message.
- **Accept:** calibration on a sample or synthetic stream produces a valid JSON profile; poor-quality input (face lost for 30%) is rejected with a clear reason; normalization unit tests (z-score math, sigma floor, ratio, no NaN/inf).

### Phase 5: Recording and task runner (L)
- `data/recorder.py` writing per-session files; `/collect` flow (consent, metadata, calibration, counterbalanced n-back blocks, TLX), `/api/events`, `/api/nasa-tlx`.
- n-back implementation per 12.4, with a clock-sync approach (browser sends events; server timestamps on receipt and stores both timestamps; document the expected skew).
- Participant deletion command.
- **Accept:** a full simulated session (using video input) produces all files in 9.2 with correct columns; event timestamps align with signal timestamps within 50 ms in a sync test; block order follows the Latin-square assignment; deletion removes every file for that ID.
- **[HUMAN]** Pilot with 2 to 3 team members to check task difficulty, timing, and webcam positioning. Fix issues before real collection.

### Phase 6: Dataset builder (M)
- `build_dataset.py`: windows from sessions with labels, block-start trimming, calibration-based normalization columns (`raw`, `zbase`, `ratio`), metadata columns.
- `scripts/make_synthetic.py` generating a synthetic multi-subject dataset (clearly marked).
- **Accept:** on synthetic data, produces `windows.parquet` matching 9.3; no window crosses a block boundary; counts per class/subject printed; leakage test: per-subject baselines use only that subject's calibration data.

### Phase 7: Training and evaluation (L)
- Models, nested subject-wise tuning, LOSO and within-subject (blockwise) evaluation, metrics, confusion matrices, figures, all ablations from 11.3, statistical tests from 11.4.
- `export.py` to produce model + `model_card.json`.
- **Accept:** on synthetic data, the pipeline runs end-to-end and recovers the planted signal (F1 far above chance, with `zbase` beating `raw` when subject offsets are planted, which validates the evaluation machinery); a test confirms that no subject appears in both train and test of any fold; results are reproducible with the fixed seed; figures and tables saved to `reports/`.
- **[HUMAN]** Real data collection (Section 9.1) must be done before real results exist. Agent prepares the checklist `docs/COLLECTION_CHECKLIST.md`.

### Phase 8: Real-time engine (M)
- `realtime/engine.py`: capture, signals, ring buffer, window features every 1 s, normalization, model, smoothing and hysteresis, warnings.
- Replay mode: feed a recorded `signals.parquet` or video at real-time speed.
- **Accept:** in video-replay mode produces one prediction per second after the first full window; end-to-end latency measured and logged (target under 1.5 s); handles face-lost segments without crashing; engine uses the same `compute_window_features` as offline (test).

### Phase 9: Dashboard (M)
- Flask app, SSE stream, live dashboard per 12.2, status chips, disclaimer, warnings, indicator tiles, timeline, debug preview toggle.
- Works without a model (FR-UI-2) and without calibration (shows `not_calibrated`).
- **Accept:** `make run` serves the dashboard on localhost; with a video source, the UI updates every second; killing the camera source shows an error state, not a crash; UI usable at 1280×720 and 1920×1080; Lighthouse/console shows no JS errors.

### Phase 10: Final evaluation and report (M)
- **[HUMAN]** After real data is collected, run `make eval` on real data; generate `reports/report.md` with all tables and figures, plus a limitations section.
- Agent: `scripts/make_report.py` assembles the report with consistent figures, the exact config and git commit hash used, and dataset summary (N subjects, windows per class).
- **Accept:** one command regenerates all results from `data/`; the report includes every ablation in 11.3 and the honest-limitations section.

### Phase 11: Hardening and documentation (S to M)
- README (setup, usage, architecture diagram, screenshots, responsible use), `docs/architecture.md`, demo script (`docs/DEMO.md` with a 3-minute walkthrough), cleanup, test coverage report, final lint.
- **Accept:** a new person can clone, run `make setup && make test && make run`, and demo from the README alone; CI-style `make test` passes with no webcam.

---

## 17. Commands (target developer experience)

```bash
make setup      # create venv, install requirements, download face_landmarker.task
make test       # pytest (no webcam needed)
make run        # start Flask dashboard (live)
make run-replay VIDEO=path/to/video.mp4
make collect    # start Flask in collection mode (/collect)
make synth      # generate synthetic dataset
make dataset    # build windows.parquet from data/sessions
make train      # train models and export best
make eval       # LOSO + ablations + figures
make report     # assemble reports/report.md
```

---

## 18. Testing strategy

- **Unit:** EAR, gaze, head pose, blink detector, window features, normalization, config.
- **Property/synthetic:** known-ground-truth signals (blink counts, gaze oscillation, head rotation).
- **Integration:** video file to signals to features to prediction to SSE message.
- **ML integrity:** subject-disjoint splits, no leakage, determinism, feature-order enforcement.
- **UI smoke:** Flask test client for each route and the SSE message schema.
- **Performance:** a benchmark script that reports FPS and per-stage latency (capture, landmark, features, model).

---

## 19. Risks and honest limitations

The team prefers direct assessments, so here they are.

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Weak signal, high noise** | Blink and gaze changes under load are real but small and vary by person and by task. Expect modest cross-subject accuracy, especially for the 3-class problem. | Personalization, binary framing, within-subject reference, honest reporting. |
| **Small N** | 15 to 25 participants limits generalization claims and statistical power. | Subject-wise CV, CIs, effect sizes, state limitations, recruit as many as ethically feasible. |
| **Labels are proxies** | Task difficulty is not the same as experienced load; people differ in skill. | NASA-TLX and performance checks, pre-registered exclusion rules. |
| **Confounds** | Screen position, reading, fatigue, lighting, glasses, and camera angle change EAR, gaze, and blink behavior independently of load. Time-on-task fatigue can mimic load. | Counterbalanced block order, environment tags, cross-environment tests, include order-effect analysis. |
| **Task-specific results** | A model trained on n-back may not transfer to real work. | Second task family (S1) and explicit generalization caveat. |
| **Window leakage** | Random window splits inflate scores massively. | Subject-wise and block-wise splits only. |
| **Ethics and privacy** | Face-derived biometric data and potential misuse (surveillance). | Consent, no video storage, local-only, responsible-use policy, faculty approval. |
| **Webcam variability** | Low FPS, auto-exposure, and compression degrade blink detection (short blinks at 15 FPS are barely sampled). | Timestamp-based durations, FPS warning below 20, minimum quality gates. |

---

## 20. Open decisions (humans, please confirm; agent defaults in brackets)

1. Faculty/ethics approval path for human-subject data collection. [Assume required; draft consent template.]
2. Number of participants and recruitment plan. [Target 20.]
3. Task set: n-back only, or n-back plus arithmetic (S1)? [n-back MVP; arithmetic as stretch.]
4. Window length final value. [10 s; ablation decides.]
5. Primary claim: 3-class or binary? [Report both; lead with whichever the data support, honestly.]
6. Publication target (conference, IEEE student track, workshop) and its page and format constraints. [Unspecified.]
7. Hardware for the demo: laptop model and webcam resolution. [Assume 720p laptop webcam, CPU only.]

---

## 21. Mapping to the project abstract (traceability)

| Abstract claim | Where it is specified |
|---|---|
| Facial landmarks, EAR, blink frequency, gaze movement, head-pose variation | 7.1, 7.2 |
| Personalized calibration baseline and deviation | 8, FR-CAL |
| Random Forest and XGBoost for classification of load states | 10 |
| Real-time Flask dashboard | 12, FR-RT, FR-UI |
| Labeled data from controlled tasks, NASA-TLX | 9.1 |
| Accuracy, precision, recall, F1, confusion matrices | 11.1 |
| Cross-subject and environmental variability | 11.2, 11.3 |
| End-to-end system, pipeline, ML framework, dashboard, quantitative evaluation | Phases 1 to 10 |
| Behavioral proxies, privacy, consent, responsible deployment | 1.3, 13.2 |

---

## Appendix A: Kickoff prompt for the AI agent

Paste this into Antigravity (or any coding agent) after placing `PRD.md` at the repo root:

> Read `PRD.md` fully, starting with Section 0. You are implementing the `cogload` project. First create the Phase 0 scaffold and write `AGENTS.md` from Section 0. Then work through the phases in order. For each phase: list your task plan, implement, write tests, run `make test`, verify every acceptance criterion, and update `docs/PROGRESS.md` before moving on. Everything must be testable without a webcam (use video files and the synthetic generator). Do not add features outside the PRD. If a decision in Section 20 blocks you, log it in `docs/QUESTIONS.md`, pick the stated default, and continue. Start with Phase 0 and Phase 1 now.

## Appendix B: Suggested working pattern with an agent

- Give the agent **one phase at a time** (copy the phase section into the prompt) rather than the entire project at once; review the diff and the test results between phases.
- Commit after every accepted phase (`git tag phase-N-done`).
- Keep a human in the loop for: landmark index verification (Phase 1), the n-back timing pilot (Phase 5), and all data collection and ethics steps.
- Run the real-data evaluation yourself and read the numbers before the agent writes any narrative about them; do not let generated text outrun the evidence.

## Appendix C: References to read before the paper (verify details yourself)

- NASA Task Load Index (Hart and Staveland, 1988), and the unweighted "Raw TLX" variant.
- Soukupova and Cech (2016), "Real-Time Eye Blink Detection using Facial Landmarks" (EAR).
- Kirchner (1958) on the n-back paradigm.
- MediaPipe Face Landmarker documentation (Google AI Edge).
- Literature on blink rate, PERCLOS, and gaze dynamics as workload and fatigue indicators (search IEEE Xplore and Google Scholar for "webcam cognitive load estimation", "remote photoplethysmography and eye-based workload"; collect 15 to 20 papers for the related-work section).
