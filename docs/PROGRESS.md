# Project Progress Log

## Phase Overview & Completion Status

### Phase 0: Project Scaffold ✅
- Repo structure created per PRD Section 14.
- `requirements.txt`, `pyproject.toml`, `Makefile`, `config.yaml`, `AGENTS.md` initialized.
- Typed configuration loader `src/cogload/config.py` implemented.
- Model asset `models/face_landmarker.task` downloaded and verified.

### Phase 1: Capture & Facial Landmarks ✅
- Video capture sources implemented: `WebcamSource`, `FileVideoSource`, `ReplaySource`.
- Threaded non-blocking capture queue with frame-dropping and monotonic timestamping.
- MediaPipe Tasks API `FaceLandmarkerWrapper` implemented with 478-point topology, bounding-box selection, and occlusion handling.
- Visual debug overlay script `scripts/debug_overlay.py` built.

### Phase 2: Per-Frame Behavioral Signals ✅
- Per-frame signal extraction: Eye Aspect Ratio (EAR), horizontal and vertical iris-based gaze ratios, 3D head pose estimation with phase unwrapping.
- Optional S4 stretch features implemented: brow furrow and mouth tension proxies.
- Signal preprocessing: invalid frame handling, short-gap linear interpolation (≤ 3 frames), rolling median smoothing for gaze.
- Video processor script `scripts/process_video.py` built.

### Phase 3: Blink Detection & Window Features ✅
- Timestamp-based blink detection algorithm with refractory period, min/max duration filters, and PERCLOS.
- Window feature extractor (`compute_window_features`) returning 24 canonical features across eye openness, blink frequency, gaze dispersion, head motion, and tension.
- Identical code used across live streaming and offline dataset processing.
- Window validity filter strictly enforcing ≥ 80% valid frame threshold.

### Phase 4: Personal Baseline Calibration ✅
- Guided calibration routine measuring relaxed baseline open EAR, neutral head pose, gaze center, and feature statistics.
- `CalibrationProfile` JSON serialization and deserialization.
- Three normalization schemes implemented and compared: `raw`, `zbase` (with training population sigma floors), and `ratio`.

### Phase 5: Experiment Runner & Data Collection ✅
- In-browser cognitive task runner (/collect) featuring Latin-square counterbalanced 1-back, 2-back, and 3-back blocks.
- NASA-TLX 6-dimensional workload survey collection.
- `SessionRecorder` generating structured datasets (`signals.parquet`, `events.csv`, `nasa_tlx.csv`, `meta.json`).
- Participant data deletion API endpoint for ethical compliance (FR-COL-7).

### Phase 6: Dataset Builder & Synthetic Generator ✅
- Dataset builder `src/cogload/data/build_dataset.py` segmenting sessions into sliding windows with block-start transient trimming and subject-specific normalization columns.
- Multi-subject synthetic dataset generator `scripts/make_synthetic.py` (FR-SYN) supporting camera-less CI and automated verification.

### Phase 7: ML Classification & Cross-Validation ✅
- Estimators implemented: Majority Baseline, Standardized Logistic Regression, Balanced Random Forest, and XGBoost.
- Rigorous Leave-One-Subject-Out (LOSO) cross-validation ensuring strict zero data leakage across participant identities.
- Model export producing `model.joblib` and `model_card.json` with training metadata and sigma floors.

### Phase 8: Real-Time Inference Engine ✅
- Ring-buffer signal processing pipeline executing 10 s sliding window feature extraction every 1 s.
- Probability smoothing via Exponential Moving Average (EMA) and state hysteresis margin.
- Continuous cognitive load score [0.0, 1.0] and confidence estimation.

### Phase 9: Web Application & Live Dashboard ✅
- Flask server running live Server-Sent Events (SSE) stream at 1 Hz.
- Interactive web dashboard displaying live cognitive load badge (LOW / MEDIUM / HIGH), continuous load gauge, 5-minute rolling workload timeline (Chart.js), and 4 behavioral indicator tiles with baseline z-score deltas.
- Guided calibration interface (/calibrate) with center fixation cross and countdown timer.
- Prominent behavioral proxy disclaimer banners displayed across all pages.

### Phase 10 & 11: Evaluation, Reporting, & Hardening ✅
- Full test suite passing across all modules (25 unit tests).
- Robust oral activity (eating / chewing / speaking) artifact detection and suppression filter:
  - Outer lip vermilion landmarks (0, 17) used to avoid resting closed-lip zero division.
  - Inner lip mastication detector (`inner_gap / ipd > 0.04`) bounds mouth tension during open chewing.
  - Mastication filter dampens jaw-coupled head pose pitch oscillations, chewing tremor, and neutralizes mouth tension when eyebrows remain relaxed (`brow_furrow <= 0.28`).
  - Workload probability capping suppresses false HIGH cognitive load alarms while snacking, eating, or speaking.
  - Informational warning pill (`ℹ️ Oral activity (eating / speaking) — jaw tension artifacts filtered`) surfaces dynamically in the live dashboard.
- Safari/macOS canvas video rendering with continuous single-frame fallback (`/api/frame`) delivering stable 30 FPS video with MediaPipe mesh debug overlay.
- Full evaluation runner `scripts/run_eval.py` generating confusion matrices and ablation charts.
- Empirical evaluation report `reports/report.md` generated.
- Complete unit and integration test suite (24 tests) passing headless without webcam.

