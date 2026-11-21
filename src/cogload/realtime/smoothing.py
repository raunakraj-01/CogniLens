"""Probability smoothing (EMA) and state hysteresis (PRD Section 6.7 & 10.1)."""

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np


@dataclass
class SmoothedPrediction:
    level: str             # "LOW" | "MEDIUM" | "HIGH"
    probs: Dict[str, float] # {"low": float, "medium": float, "high": float}
    score: float           # Continuous load score in [0.0, 1.0]
    confidence: float      # Maximum smoothed class probability


class OutputSmoother:
    """Applies Exponential Moving Average (EMA) and hysteresis margin to prevent flicker."""

    def __init__(self, alpha: float = 0.4, hysteresis_margin: float = 0.10):
        self.alpha = float(alpha)
        self.hysteresis_margin = float(hysteresis_margin)
        self.prev_probs: Optional[np.ndarray] = None
        self.current_level: str = "LOW"
        self.classes = ["LOW", "MEDIUM", "HIGH"]

    def update(self, raw_probs: np.ndarray) -> SmoothedPrediction:
        """Update smoothed probabilities from raw model probabilities.

        Args:
            raw_probs: array of shape (3,) corresponding to [LOW, MEDIUM, HIGH]
        """
        raw_probs = np.array(raw_probs, dtype=np.float64)
        if raw_probs.ndim > 1:
            raw_probs = raw_probs.ravel()

        # Handle binary or 3-class
        if len(raw_probs) == 2:
            # Binary (Low, High) -> pad medium as 0
            p_low = float(raw_probs[0])
            p_high = float(raw_probs[1])
            raw_probs = np.array([p_low, 0.0, p_high])
        elif len(raw_probs) < 3:
            raw_probs = np.array([0.33, 0.33, 0.34])

        # Normalize sum to 1.0
        prob_sum = np.sum(raw_probs)
        if prob_sum > 0:
            raw_probs = raw_probs / prob_sum
        else:
            raw_probs = np.array([0.33, 0.33, 0.34])

        # Exponential Moving Average
        if self.prev_probs is None:
            smoothed = raw_probs
        else:
            smoothed = self.alpha * raw_probs + (1.0 - self.alpha) * self.prev_probs

        self.prev_probs = smoothed

        # Continuous load score (PRD 10.1): score = P_med * 0.5 + P_high * 1.0
        p_low = float(smoothed[0])
        p_med = float(smoothed[1])
        p_high = float(smoothed[2])
        score = float(np.clip(p_med * 0.5 + p_high * 1.0, 0.0, 1.0))

        # Hysteresis for discrete label change (prevents flickering)
        curr_idx = self.classes.index(self.current_level)
        curr_prob = smoothed[curr_idx]

        candidate_idx = int(np.argmax(smoothed))
        candidate_prob = smoothed[candidate_idx]

        if candidate_idx != curr_idx:
            # Only switch if candidate probability exceeds current by hysteresis_margin
            if candidate_prob > (curr_prob + self.hysteresis_margin):
                self.current_level = self.classes[candidate_idx]

        confidence = float(smoothed[self.classes.index(self.current_level)])

        return SmoothedPrediction(
            level=self.current_level,
            probs={"low": round(p_low, 3), "medium": round(p_med, 3), "high": round(p_high, 3)},
            score=round(score, 3),
            confidence=round(confidence, 3),
        )

    def reset(self) -> None:
        self.prev_probs = None
        self.current_level = "LOW"
