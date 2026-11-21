"""Dataset builder: creates windowed dataset from session files (PRD Section 6.6 & 9.3)."""

from pathlib import Path
from typing import Optional, List, Dict
import json
import numpy as np
import pandas as pd
from cogload.config import AppConfig, load_config
from cogload.features.window import compute_window_features
from cogload.features.names import FEATURE_NAMES
from cogload.calibration.profile import CalibrationProfile
from cogload.calibration.normalize import normalize_features, recenter_signals


def build_dataset_from_sessions(
    sessions_dir: str | Path = "data/sessions",
    output_path: str | Path = "data/processed/windows_W10_S5.parquet",
    cfg: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """Builds windows.parquet from raw session recordings.

    Rules (PRD 9.1 & 9.3):
    - Sliding windows (W=10s, stride=5s offline).
    - Discard first 15s of each block (drop_block_start_sec) to avoid task switch transients.
    - No window crosses block boundaries.
    - Stores raw, zbase, and ratio normalized features.
    """
    if cfg is None:
        cfg = load_config()

    sessions_dir = Path(sessions_dir)
    rows: List[Dict[str, any]] = []

    if not sessions_dir.exists():
        print(f"No sessions directory found at {sessions_dir}")
        return pd.DataFrame()

    session_paths = list(sessions_dir.glob("*/*"))
    for s_path in session_paths:
        if not s_path.is_dir():
            continue

        meta_file = s_path / "meta.json"
        sig_file = s_path / "signals.parquet"
        events_file = s_path / "events.csv"
        tlx_file = s_path / "nasa_tlx.csv"
        calib_file = s_path / "calibration.json"

        if not (meta_file.exists() and sig_file.exists() and events_file.exists()):
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sig_df = pd.read_parquet(sig_file)
        events_df = pd.read_csv(events_file)
        tlx_df = pd.read_csv(tlx_file) if tlx_file.exists() else pd.DataFrame()

        tlx_map: Dict[str, float] = {}
        if not tlx_df.empty and "block_id" in tlx_df.columns and "raw_tlx" in tlx_df.columns:
            tlx_map = dict(zip(tlx_df["block_id"], tlx_df["raw_tlx"]))

        # Load calibration profile if present
        calib_profile: Optional[CalibrationProfile] = None
        baseline_open_ear = 0.28
        if calib_file.exists():
            calib_profile = CalibrationProfile.load_json(calib_file)
            baseline_open_ear = calib_profile.baseline_open_ear

        # Extract blocks from events
        # Events have block_start and block_stop
        block_starts = events_df[events_df["event_type"] == "block_start"]
        for _, start_ev in block_starts.iterrows():
            block_id = str(start_ev["block_id"])
            task = str(start_ev["task"])
            level = int(start_ev["level"])
            t_block_start = float(start_ev["timestamp_s"])

            # Find matching stop event
            stop_ev = events_df[
                (events_df["event_type"] == "block_stop") & (events_df["block_id"] == block_id)
            ]
            if stop_ev.empty:
                continue
            t_block_stop = float(stop_ev.iloc[0]["timestamp_s"])

            # Trim first 15 seconds transient (drop_block_start_sec)
            t_valid_start = t_block_start + cfg.features.drop_block_start_sec
            if t_valid_start >= t_block_stop:
                continue

            # Slice signals within this block boundary
            block_signals = sig_df[
                (sig_df["t"] >= t_valid_start) & (sig_df["t"] <= t_block_stop)
            ].copy()
            if block_signals.empty or len(block_signals) < 10:
                continue

            # Re-center signals using participant's calibration
            block_signals = recenter_signals(block_signals, calib_profile)

            # Generate sliding windows
            w_sec = cfg.features.window_sec
            stride_sec = cfg.features.stride_sec_offline
            curr_w_start = t_valid_start

            while curr_w_start + w_sec <= t_block_stop + 1e-4:
                curr_w_end = curr_w_start + w_sec
                w_slice = block_signals[
                    (block_signals["t"] >= curr_w_start) & (block_signals["t"] < curr_w_end)
                ]

                # Compute window features
                raw_feats = compute_window_features(
                    w_slice, baseline_open_ear=baseline_open_ear, cfg=cfg.features
                )
                if raw_feats is not None:
                    # Map level (1->0: Low, 2->1: Medium, 3->2: High)
                    # or 'rest' if level is 0
                    if level == 0:
                        level_label = "rest"
                    elif level == 1:
                        level_label = 0
                    elif level == 2:
                        level_label = 1
                    elif level == 3:
                        level_label = 2
                    else:
                        level_label = level

                    # Compute normalized versions
                    zbase_feats = normalize_features(raw_feats, calib_profile, scheme="zbase")
                    ratio_feats = normalize_features(raw_feats, calib_profile, scheme="ratio")

                    row: Dict[str, Any] = {
                        "participant_id": meta.get("participant_id", "P000"),
                        "session_id": meta.get("session_id", "S01"),
                        "block_id": block_id,
                        "task": task,
                        "level_label": level_label,
                        "t_start": curr_w_start,
                        "t_end": curr_w_end,
                        "valid_ratio": raw_feats.get("valid_ratio", 1.0),
                        "lighting": meta.get("lighting", "normal"),
                        "glasses": meta.get("glasses", False),
                        "nasa_tlx_block": tlx_map.get(block_id, np.nan),
                    }

                    # Add raw features
                    for k in FEATURE_NAMES:
                        row[f"{k}_raw"] = raw_feats.get(k, 0.0)
                        row[f"{k}_zbase"] = zbase_feats.get(k, 0.0)
                        row[f"{k}_ratio"] = ratio_feats.get(k, 0.0)

                    rows.append(row)

                curr_w_start += stride_sec

    dataset_df = pd.DataFrame(rows)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_df.empty:
        dataset_df.to_parquet(out_path, index=False)
        print(f"Dataset saved to {out_path} with {len(dataset_df)} windows across {dataset_df['participant_id'].nunique()} participants.")
    else:
        print("No windows generated. Check sessions and validity.")

    return dataset_df


if __name__ == "__main__":
    build_dataset_from_sessions()
