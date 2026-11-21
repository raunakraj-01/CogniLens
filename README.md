# Cogload: Real-Time Cognitive Load Estimation from Commodity Webcams

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 25 Passed](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)]()
[![Inference: ~30 FPS CPU](https://img.shields.io/badge/inference-~30%20FPS%20(CPU)-success.svg)]()

> **TL;DR:** An end-to-end computer vision and machine learning pipeline that estimates cognitive workload in real time using an ordinary laptop camera. It extracts 24 bio-inspired ocular, head pose, and facial tension features at ~30 FPS on CPU, normalizes against personalized baselines, and runs an artifact-filtered classifier without storing any raw video on disk.

---

## Why This Project?

Measuring mental workload traditionally requires expensive, intrusive hardware—EEG headbands ($1,000+), medical-grade pupillometry rigs, or research eye-trackers ($15,000+). 

**Cogload** explores how far commodity webcams can go by tracking subtle behavioral proxies:
- **Blink Dynamics & PERCLOS:** Cognitive suppression during intense focus vs. resting blink rates.
- **2D Gaze Fixation & Saccade Rates:** Normalized iris-to-eye projections capturing gaze velocity and focal dispersion.
- **3D Head Pose Dynamics:** Pitch, yaw, and roll tracking via `cv2.solvePnP` with phase unwrapping to prevent gimbal flips.
- **Facial Muscle Tension:** Corrugator supercilii (inner eyebrow furrow) and vermilion lip compression as physical strain markers.

---

## System Architecture

```
[Webcam / Video File / Replay Stream]
                  │
                  ▼
   [Threaded Non-Blocking Capture Queue]   (Drops stale frames, monotonic timestamps)
                  │
                  ▼
  [MediaPipe Face Landmarker (478 pts)]    (Sub-millisecond landmark extraction)
                  │
                  ▼
        [Per-Frame Signal Extraction]
        ├── Eye Aspect Ratio (EAR)
        ├── Iris Gaze Coordinates (gaze_x, gaze_y)
        ├── 3D Head Pose (solvePnP + Rodrigues)
        └── Facial Tension (Brow Furrow & Lip Ratio)
                  │
                  ▼
      [10-Second Sliding Ring Buffer]
                  │
                  ▼
    [Temporal Window Feature Extractor (24 Features)]
        ├── Blinks: Rate, mean duration, variance, PERCLOS
        ├── Gaze: Mean speed, 95th percentile, saccades, fixations
        ├── Head Motion: Mean velocity, energy, angle variance
        └── Tension: Brow furrow & mouth compression statistics
                  │
                  ▼
      [Personalized Baseline Normalizer]   (z_base against 30s calibration profile)
                  │
                  ▼
    [Mastication & Motion Artifact Filter] (Suppresses eating/chewing false alarms)
                  │
                  ▼
         [ML Inference Engine]             (Balanced Random Forest / XGBoost)
                  │
                  ▼
     [EMA Smoothing & State Hysteresis]    (Prevents flickering between LOW/MED/HIGH)
                  │
                  ▼
      [SSE Stream -> Live Web Dashboard]   (1 Hz updates, continuous score, Chart.js)
```

---

## Key Engineering Highlights

### 1. Real-Time CPU Pipeline (<18ms per frame)
Operates at a sustained **30 FPS on a standard laptop CPU** with zero GPU requirements. Frame capture and inference are decoupled across a threaded ring buffer to guarantee non-blocking UI rendering and prevent memory bloat.

### 2. Personalized Baseline Calibration ($z_{base}$)
Human eye openness (EAR), resting head posture, and facial proportions differ vastly across individuals and camera angles. Instead of naive global thresholds, Cogload includes an in-browser 30-second resting calibration protocol that establishes individual $\mu$ and $\sigma$ parameters with training population sigma floors:
$$z = \frac{f - \mu_{\text{baseline}}}{\max(\sigma_{\text{baseline}}, \sigma_{\text{floor}})}$$

### 3. Mastication & Oral Activity Rejection Filter
*The Challenge:* Normal activities like eating a snack or talking while working move the jaw (chin landmark 152), which `solvePnP` misinterprets as rapid 50–160 deg/s head pitch oscillation, falsely triggering high cognitive load alarms.  
*The Fix:* A dedicated oral activity detector tracks rhythmic mouth aperture variation. When jaw oscillation is detected while eyebrows remain relaxed ($\text{brow\_furrow} \le 0.28$), the system actively neutralizes mastication artifacts, flags an informational banner, and dampens workload spikes to keep the estimate accurate.

### 4. Zero Data Leakage (Strict LOSO Evaluation)
Evaluated strictly using **Leave-One-Subject-Out (LOSO) Cross-Validation**. In every fold, all windows from a participant are held out entirely from training to prove true generalization across new faces, rather than memorizing individual facial structure.

### 5. Privacy by Design
- **No raw video or image frames ever touch the disk or network.**
- Frames are processed transiently in volatile memory and immediately discarded.
- Built-in data deletion protocol conforming to research ethics standards.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Computer Vision** | OpenCV 4.10, MediaPipe Face Landmarker (478 dense landmarks) |
| **Data & Signals** | NumPy, SciPy (Gaussian & median filters, PnP pose), Pandas, PyArrow Parquet |
| **Machine Learning** | Scikit-Learn (Random Forest, Logistic Regression), XGBoost |
| **Backend & Serving** | Python 3.10+, Flask, Server-Sent Events (SSE), Threading |
| **Frontend UI** | HTML5 Canvas (30 FPS video preview), CSS3, Chart.js, Vanilla ES6 JS |
| **Testing & CI** | Pytest (25 headless tests, mock replay, synthetic signal generators) |

---

## Live Dashboard & Ground Truth Task Runner

The web application runs locally at `http://127.0.0.1:5050`:

1. **Live Monitor (`/`):**
   - Real-time cognitive load state badge (`LOW`, `MEDIUM`, `HIGH`) with hysteresis smoothing.
   - Continuous workload progress gauge `[0.0 - 1.0]` and confidence readout.
   - 60-second rolling workload history timeline via Chart.js.
   - 4 live behavioral indicator cards (Blink Rate, EAR, Gaze Dispersion, Head Speed) with delta baseline z-scores.
   - Real-time MediaPipe facial mesh debug canvas overlay.

2. **Calibration Suite (`/calibrate`):**
   - Guided fixation cross routine computing individual resting baseline parameters.

3. **In-Browser Experiment Runner (`/collect`):**
   - Counterbalanced **n-back cognitive memory tasks** (1-back, 2-back, 3-back) for controlled workload induction.
   - Post-block **NASA-TLX 6-dimensional survey** collection for ground-truth validation.

---

## Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- A built-in or USB webcam (or a video file for headless replay)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cogload.git
cd cogload

# Create virtual environment and install dependencies
make setup
```

### 2. Run Test Suite
```bash
# Runs 25 automated tests (fully headless; no camera required)
make test
```

### 3. Launch the Application
```bash
# Start the live dashboard on http://127.0.0.1:5050
make run
```

### 4. Headless Video Replay (No Webcam Needed)
```bash
# Replay any prerecorded video through the full live pipeline
make run-replay VIDEO=path/to/test_video.mp4
```

---

## Offline ML Pipeline & Reproducibility

To re-run the synthetic data generation, window segmentation, and LOSO evaluation benchmark:

```bash
# 1. Generate multi-subject synthetic behavioral signals (FR-SYN)
make synth

# 2. Build 10-second sliding window dataset (data/processed/windows_W10_S5.parquet)
make dataset

# 3. Run full Leave-One-Subject-Out evaluation and export best model
make eval

# 4. Generate ablation and metrics markdown summary (reports/report.md)
make report
```

---

## Responsible Use & Ethical Framing

- **Behavioral Proxy Disclaimer:** Facial expressions, blinks, and gaze shifts are behavioral proxies correlated with cognitive workload. They do **not** directly read brain activity or assess intellectual capacity.
- **Prohibited Uses:** This software is designed exclusively for research, accessibility, and ergonomic break-reminder systems. It is strictly not intended for employee surveillance, workplace productivity scoring, high-stakes exam proctoring, or medical/clinical diagnosis.

---

## License

This project is licensed under the [MIT License](LICENSE).
