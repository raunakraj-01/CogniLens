# Cogload Architecture Specification

## Dataflow Pipeline
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
    - Brow / Mouth tension (optional S4)
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
   [Baseline Normalizer]
    (raw, zbase, ratio against CalibrationProfile)
              │
              ▼
      [ML Classifier]
    (LR / Random Forest / XGBoost)
              │
              ▼
   [Probability Smoothing & Hysteresis]
              │
              ▼
   [SSE Stream -> Flask Web UI]
```
