# Cogload: Real-Time Cognitive Load Estimation via Facial & Ocular Signals

> **Important Framing:** Webcam-derived behavioral signals (Eye Aspect Ratio, blink frequency, gaze dispersion, and head movement) are **behavioral proxies** for cognitive workload, **not** direct measurements of neural activity. The system estimates an operational workload level, not a person's intelligence or capability, and is not a medical device.

---

## Architecture Overview

```
[Webcam / Video / Replay Source]
              │
              ▼
   [Threaded Capture Queue]
              │
              ▼
[MediaPipe Face Landmarker (478 pts)]
              │
              ▼
    [Per-Frame Signals]
    - Eye Aspect Ratio (EAR)
    - Iris Gaze Ratios (x, y)
    - Head Pose (yaw, pitch, roll)
    - Brow / Mouth tension proxies (S4)
              │
              ▼
    [Sliding Ring Buffer (W=10s)]
              │
              ▼
   [Window Feature Extractor]
    - Blink statistics & PERCLOS
    - Gaze dispersion & saccade rates
    - Head motion velocity & energy
              │
              ▼
   [Personal Baseline Normalizer]
    (raw, zbase, ratio against CalibrationProfile)
              │
              ▼
      [ML Classifier]
    (Random Forest / XGBoost / Logistic Regression)
              │
              ▼
   [Probability Smoothing & Hysteresis]
              │
              ▼
   [SSE Stream -> Flask Web UI]
```

---

## Quick Start

### 1. Setup & Tests
```bash
# Setup environment, download FaceLandmarker task asset, install editable package
make setup

# Run complete test suite (24 headless unit & integration tests, no camera needed)
make test
```

### 2. Run the Live Web Dashboard
```bash
# Start live dashboard on http://127.0.0.1:5000
make run

# Or run with a sample video file replay
make run-replay VIDEO=path/to/video.mp4
```

### 3. Experiment Data Collection Runner
Visit `http://127.0.0.1:5000/collect` or run:
```bash
make collect
```
- Guided consent screen
- Counterbalanced **n-back memory tasks** (1-back, 2-back, 3-back)
- Synchronized event logging
- Post-block **NASA-TLX** 6-dimensional workload survey

### 4. Synthetic Data Generation & Offline Model Training
```bash
# Generate multi-subject synthetic data (FR-SYN)
make synth

# Build windowed dataset (windows_W10_S5.parquet)
make dataset

# Train models and run full LOSO ablation suite
make eval

# Generate summary report (reports/report.md)
make report
```

---

## Web Dashboard Features (`/`)
- **Live State Badge:** Displays real-time cognitive workload level (`LOW`, `MEDIUM`, `HIGH`) with smoothed state hysteresis.
- **Continuous Workload Gauge:** Real-time continuous workload score in `[0.0, 1.0]`.
- **Workload Timeline:** 5-minute rolling workload history chart rendered with local Chart.js.
- **4 Key Behavioral Indicator Tiles:**
  - Blink Rate (`bpm` with $\Delta$ z-score vs baseline)
  - Eye Aspect Ratio (`EAR` openness proxy)
  - Gaze Dispersion ($\Delta$ z-score)
  - Head Motion Speed (`deg/s` with $\Delta$ z-score)
- **Guided Personal Calibration (`/calibrate`):** 90-second neutral fixation cross to establish personalized ocular and head baselines.

---

## Project Structure
- `src/cogload/capture/`: Video capture abstractions and non-blocking queue thread.
- `src/cogload/vision/`: MediaPipe Tasks API face landmarker wrapper and 478-point topology mappings.
- `src/cogload/signals/`: Per-frame EAR, gaze ratio, head pose (PnP), and facial tension extraction.
- `src/cogload/features/`: Timestamp-based blink detection, PERCLOS, and sliding window features.
- `src/cogload/calibration/`: Guided calibration protocol and `zbase` / `ratio` normalization.
- `src/cogload/data/`: Session recording, event synchronization, and dataset builder.
- `src/cogload/ml/`: Model definitions (LR, RF, XGBoost), LOSO cross-validation, and model export.
- `src/cogload/realtime/`: Real-time ring buffer engine, probability EMA smoothing, and hysteresis.
- `src/cogload/app/`: Flask web application, SSE streaming endpoints, and HTML5 templates.
- `reports/`: Confusion matrices, ablation comparison plots, and `report.md`.

---

## Responsible Use & Ethical Guidelines

### Intended Use
This project is intended strictly for:
- Academic research in human-computer interaction (HCI), ergonomics, and cognitive ergonomics.
- Educational demonstrations of real-time computer vision and bio-inspired signal processing.
- Opt-in self-monitoring to assist with work pacing and break reminders.

### Inappropriate & Prohibited Uses
- **No Workplace Surveillance:** Must not be used for employee monitoring, productivity evaluation, or disciplinary actions.
- **No Exam Proctoring or Assessment:** Cognitive load estimation does not indicate honesty, attentiveness, or academic integrity.
- **No Clinical or Medical Diagnosis:** The system measures behavioral proxies from a webcam and cannot diagnose ADHD, cognitive impairment, or neurological disorders.
- **No Hiring or Performance Scoring:** Behavioral proxies vary significantly across individuals, neurotypes, lighting, and camera setups.

### Privacy by Design
- **Zero Raw Image Retention:** No video frames or face images are ever stored on disk or transmitted over networks.
- Frames are processed transiently in volatile memory and discarded immediately after landmark extraction.
- Local execution by default on `127.0.0.1`.
- Built-in data deletion endpoints for participant ethical rights.
