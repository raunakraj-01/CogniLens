"""Synthetic dataset generator for test, validation, and CI (PRD Section 6.4b FR-SYN)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cogload.data.schema import SessionMeta, TaskEvent, NasaTlxResponse
from cogload.data.recorder import SessionRecorder
from cogload.calibration.protocol import run_calibration_from_signals


def generate_synthetic_session(
    participant_id: str,
    base_dir: str | Path = "data/sessions",
    seed: int = 42,
    fps: float = 30.0,
    block_duration_sec: float = 60.0,
) -> Path:
    """Generate a realistic synthetic session with planted cognitive load signatures."""
    subj_num = int(participant_id.replace("P", "").replace("S", ""))
    rng = np.random.RandomState(seed + subj_num * 101)

    # Subject-specific idiosyncratic baselines
    subj_ear_base = rng.uniform(0.26, 0.32)
    subj_blink_rate = rng.uniform(16.0, 22.0)  # Blinks per min at rest
    subj_gaze_cx = rng.uniform(-0.02, 0.02)
    subj_gaze_cy = rng.uniform(-0.02, 0.02)
    subj_head_yaw = rng.uniform(-3.0, 3.0)
    subj_head_pitch = rng.uniform(-3.0, 3.0)
    subj_head_roll = rng.uniform(-1.5, 1.5)
    subj_brow_neutral = rng.uniform(0.12, 0.18)
    subj_mouth_neutral = rng.uniform(3.2, 3.8)

    # Subject behavioural response archetype (ensures models learn each modality's signal)
    archetypes = ["all", "tension_dominant", "all", "tension_dominant", "fidget_dominant", "ocular_dominant"]
    archetype = archetypes[subj_num % len(archetypes)]

    tension_mult = 1.5 if archetype in ["tension_dominant", "all"] else 0.0
    fidget_mult = 1.5 if archetype in ["fidget_dominant", "all"] else 0.0
    ocular_mult = 1.5 if archetype in ["ocular_dominant", "all"] else 0.0

    recorder = SessionRecorder(base_dir=base_dir)
    meta = SessionMeta(
        participant_id=participant_id,
        session_id="S01",
        created_at=datetime.now(timezone.utc).isoformat(),
        glasses=bool(rng.choice([True, False])),
        lighting=rng.choice(["bright", "normal", "dim"]),
        camera_name="SyntheticCamera",
        notes=f"Synthetic session (archetype: {archetype})",
    )
    recorder.start_session(meta)

    # Simulation state variables for continuous smooth dynamics
    curr_yaw = subj_head_yaw
    curr_pitch = subj_head_pitch
    curr_roll = subj_head_roll
    curr_gx = subj_gaze_cx
    curr_gy = subj_gaze_cy
    target_gx = subj_gaze_cx
    target_gy = subj_gaze_cy
    blink_frames_remaining = 0
    t_curr = 0.0

    def step_frame(
        effective_blink_rate: float,
        gaze_saccade_p: float,
        gaze_saccade_amp: float,
        head_drift_noise: float,
        brow_target: float,
        mouth_target: float,
    ) -> dict:
        nonlocal curr_yaw, curr_pitch, curr_roll
        nonlocal curr_gx, curr_gy, target_gx, target_gy
        nonlocal blink_frames_remaining, t_curr

        # 1. Blink dynamics (each blink lasts 4-7 frames, PRD Section 7.2)
        if blink_frames_remaining > 0:
            blink_frames_remaining -= 1
            ear_val = float(0.07 + rng.normal(0, 0.008))
        else:
            p_blink_start = (effective_blink_rate / 60.0) / fps
            if rng.rand() < p_blink_start:
                blink_frames_remaining = rng.randint(4, 8)
                ear_val = float(0.07 + rng.normal(0, 0.008))
            else:
                ear_val = float(subj_ear_base + rng.normal(0, 0.008))
        ear_val = float(np.clip(ear_val, 0.05, 0.45))

        # 2. Gaze dynamics (fixations with occasional saccades, PRD Section 7.3)
        if rng.rand() < gaze_saccade_p:
            target_gx = subj_gaze_cx + rng.normal(0, gaze_saccade_amp)
            target_gy = subj_gaze_cy + rng.normal(0, gaze_saccade_amp)
        curr_gx += 0.25 * (target_gx - curr_gx) + rng.normal(0, 0.002)
        curr_gy += 0.25 * (target_gy - curr_gy) + rng.normal(0, 0.002)

        # 3. Head pose dynamics (mean-reverting continuous walk, PRD Section 7.4)
        curr_yaw += 0.04 * (subj_head_yaw - curr_yaw) + rng.normal(0, head_drift_noise)
        curr_pitch += 0.04 * (subj_head_pitch - curr_pitch) + rng.normal(0, head_drift_noise)
        curr_roll += 0.04 * (subj_head_roll - curr_roll) + rng.normal(0, head_drift_noise * 0.5)

        # 4. Facial tension proxies (PRD Section 3.2 S4)
        brow_val = float(np.clip(brow_target + rng.normal(0, 0.02), 0.0, 1.0))
        mouth_val = float(np.clip(mouth_target + rng.normal(0, 0.15), 1.0, 15.0))

        row = {
            "t": t_curr,
            "valid": True,
            "ear_l": ear_val,
            "ear_r": ear_val,
            "ear": ear_val,
            "gaze_x_l": float(curr_gx),
            "gaze_x_r": float(curr_gx),
            "gaze_x": float(curr_gx),
            "gaze_y_l": float(curr_gy),
            "gaze_y_r": float(curr_gy),
            "gaze_y": float(curr_gy),
            "yaw": float(curr_yaw),
            "pitch": float(curr_pitch),
            "roll": float(curr_roll),
            "brow_furrow": brow_val,
            "mouth_tension": mouth_val,
        }
        t_curr += 1.0 / fps
        return row

    # 1. Calibration phase (45 seconds of resting baseline fixation)
    calib_sec = 45.0
    calib_frames = int(calib_sec * fps)
    calib_rows = []

    for _ in range(calib_frames):
        row = step_frame(
            effective_blink_rate=subj_blink_rate,
            gaze_saccade_p=0.03,
            gaze_saccade_amp=0.02,
            head_drift_noise=0.06,
            brow_target=subj_brow_neutral,
            mouth_target=subj_mouth_neutral,
        )
        calib_rows.append(row)
        recorder.log_signal_row(row)

    # Compute calibration profile
    calib_df = pd.DataFrame(calib_rows)
    profile, _ = run_calibration_from_signals(calib_df, participant_id=participant_id)
    if profile:
        recorder.set_calibration(profile)

    # 2. Cognitive Task Blocks (Levels 1, 2, 3 counterbalanced)
    block_levels = [1, 2, 3]
    rng.shuffle(block_levels)

    for block_num, level in enumerate(block_levels, start=1):
        block_id = f"B{block_num}_L{level}"

        recorder.log_event(
            TaskEvent(
                timestamp_s=t_curr,
                event_type="block_start",
                block_id=block_id,
                task="n-back",
                level=level,
                payload_json=json.dumps({"level": level, "archetype": archetype}),
            )
        )

        if level == 1:
            eff_blink = subj_blink_rate * 1.0
            gaze_p = 0.03
            gaze_amp = 0.02
            head_noise = 0.06
            brow_t = subj_brow_neutral
            mouth_t = subj_mouth_neutral
            tlx_score = rng.uniform(18.0, 32.0)
        elif level == 2:
            eff_blink = subj_blink_rate * max(0.4, (1.0 - 0.28 * ocular_mult))
            gaze_p = 0.04 + 0.04 * ocular_mult
            gaze_amp = 0.02 + 0.03 * ocular_mult
            head_noise = 0.06 + 0.08 * fidget_mult
            brow_t = subj_brow_neutral + 0.12 * tension_mult
            mouth_t = subj_mouth_neutral + 1.1 * tension_mult
            tlx_score = rng.uniform(45.0, 58.0)
        else:
            eff_blink = subj_blink_rate * max(0.25, (1.0 - 0.58 * ocular_mult))
            gaze_p = 0.05 + 0.10 * ocular_mult
            gaze_amp = 0.03 + 0.07 * ocular_mult
            head_noise = 0.06 + 0.22 * fidget_mult
            brow_t = subj_brow_neutral + 0.28 * tension_mult
            mouth_t = subj_mouth_neutral + 2.5 * tension_mult
            tlx_score = rng.uniform(72.0, 90.0)

        frames_in_block = int(block_duration_sec * fps)
        for _ in range(frames_in_block):
            row = step_frame(
                effective_blink_rate=eff_blink,
                gaze_saccade_p=gaze_p,
                gaze_saccade_amp=gaze_amp,
                head_drift_noise=head_noise,
                brow_target=brow_t,
                mouth_target=mouth_t,
            )
            recorder.log_signal_row(row)

        recorder.log_event(
            TaskEvent(
                timestamp_s=t_curr,
                event_type="block_stop",
                block_id=block_id,
                task="n-back",
                level=level,
            )
        )

        # NASA-TLX
        recorder.log_nasa_tlx(
            NasaTlxResponse(
                block_id=block_id,
                mental=tlx_score + rng.normal(0, 2),
                physical=rng.uniform(10, 25),
                temporal=tlx_score - rng.uniform(5, 10),
                performance=100.0 - tlx_score,
                effort=tlx_score + rng.normal(0, 2),
                frustration=tlx_score - rng.uniform(5, 10),
                raw_tlx=float(tlx_score),
            )
        )

        t_curr += 10.0  # Rest break

    session_dir = recorder.finalize_session()
    return session_dir


def generate_multi_subject_dataset(
    n_subjects: int = 15,
    base_dir: str | Path = "data/sessions",
    seed: int = 42,
    block_duration_sec: float = 60.0,
) -> None:
    """Generate multi-subject synthetic dataset."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic sessions for {n_subjects} subjects in {base_dir}...")

    for i in range(1, n_subjects + 1):
        pid = f"P{i:03d}"
        generate_synthetic_session(
            participant_id=pid,
            base_dir=base_dir,
            seed=seed,
            block_duration_sec=block_duration_sec,
        )
    print("Synthetic data generation complete.")


def main():
    parser = argparse.ArgumentParser(description="Synthetic dataset generator (FR-SYN)")
    parser.add_argument("--subjects", type=int, default=15, help="Number of participants")
    parser.add_argument("--dir", type=str, default="data/sessions", help="Target sessions dir")
    parser.add_argument("--block-sec", type=float, default=60.0, help="Duration of each block")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_multi_subject_dataset(
        n_subjects=args.subjects,
        base_dir=args.dir,
        seed=args.seed,
        block_duration_sec=args.block_sec,
    )


if __name__ == "__main__":
    main()
