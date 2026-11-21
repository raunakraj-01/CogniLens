"""Blink detection and eye closure statistics (PRD Section 7.2)."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class BlinkEvent:
    start_t: float
    end_t: float
    duration_ms: float
    min_ear: float


def detect_blinks(
    timestamps: np.ndarray,
    ear_series: np.ndarray,
    valid_mask: np.ndarray,
    baseline_open_ear: float = 0.28,
    blink_ratio: float = 0.75,
    blink_min_ms: float = 60.0,
    blink_max_ms: float = 600.0,
    blink_refractory_ms: float = 100.0,
) -> List[BlinkEvent]:
    """Detect blinks from timestamped EAR series.

    Algorithm (PRD Section 7.2):
    1. Threshold T_blink = blink_ratio * baseline_open_ear (fallback 0.21 if no baseline).
    2. Runs of consecutive frames with ear < T_blink.
    3. Runs lasting between blink_min_ms and blink_max_ms are blinks.
    4. Refractory period enforces separation to prevent double-counting.
    """
    if len(timestamps) < 2 or len(ear_series) < 2:
        return []

    t_thresh = blink_ratio * baseline_open_ear if baseline_open_ear > 0.05 else 0.21

    # Boolean array of closed frames among valid frames
    is_closed = np.where(valid_mask, ear_series < t_thresh, False)

    blinks: List[BlinkEvent] = []
    in_blink = False
    start_idx = 0
    last_blink_end_t = -1e9

    for i in range(len(is_closed)):
        if is_closed[i]:
            if not in_blink:
                in_blink = True
                start_idx = i
        else:
            if in_blink:
                in_blink = False
                end_idx = i - 1
                t_start = timestamps[start_idx]
                t_end = timestamps[end_idx]
                dur_ms = (t_end - t_start) * 1000.0

                # Duration must be within blink bounds
                if blink_min_ms <= dur_ms <= blink_max_ms:
                    # Enforce refractory period
                    if (t_start - last_blink_end_t) * 1000.0 >= blink_refractory_ms:
                        min_val = float(np.min(ear_series[start_idx : end_idx + 1]))
                        blinks.append(
                            BlinkEvent(
                                start_t=t_start,
                                end_t=t_end,
                                duration_ms=dur_ms,
                                min_ear=min_val,
                            )
                        )
                        last_blink_end_t = t_end

    # Handle run reaching end of window
    if in_blink:
        end_idx = len(is_closed) - 1
        t_start = timestamps[start_idx]
        t_end = timestamps[end_idx]
        dur_ms = (t_end - t_start) * 1000.0
        if blink_min_ms <= dur_ms <= blink_max_ms:
            if (t_start - last_blink_end_t) * 1000.0 >= blink_refractory_ms:
                min_val = float(np.min(ear_series[start_idx : end_idx + 1]))
                blinks.append(
                    BlinkEvent(
                        start_t=t_start,
                        end_t=t_end,
                        duration_ms=dur_ms,
                        min_ear=min_val,
                    )
                )

    return blinks


def compute_perclos(
    ear_series: np.ndarray,
    valid_mask: np.ndarray,
    baseline_open_ear: float = 0.28,
    closure_ratio: float = 0.20,
) -> float:
    """Compute PERCLOS (percentage of eye closure).

    Fraction of valid frames with ear < closure_ratio * baseline_open_ear.
    """
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return 0.0

    closure_thresh = closure_ratio * baseline_open_ear if baseline_open_ear > 0.05 else 0.06
    closed_frames = np.sum((ear_series < closure_thresh) & valid_mask)
    return float(closed_frames / valid_count)
