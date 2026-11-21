"""Unit tests for SessionRecorder and participant deletion (FR-COL-7)."""

import json
from pathlib import Path
import pandas as pd
from cogload.data.schema import SessionMeta, TaskEvent, NasaTlxResponse
from cogload.data.recorder import SessionRecorder
from cogload.calibration.profile import CalibrationProfile


def test_session_recorder_lifecycle(tmp_path: Path):
    recorder = SessionRecorder(base_dir=tmp_path)
    meta = SessionMeta(
        participant_id="P999",
        session_id="S01",
        created_at="2026-11-21T00:00:00Z",
        glasses=True,
        lighting="bright",
        camera_name="TestCam",
    )

    # 1. Start session
    s_dir = recorder.start_session(meta)
    assert s_dir.exists()
    assert (s_dir / "meta.json").exists()

    # 2. Log signals and events
    recorder.log_signal_row({
        "t": 0.0,
        "valid": True,
        "ear": 0.3,
        "gaze_x": 0.0,
        "gaze_y": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
    })
    recorder.log_event(
        TaskEvent(
            timestamp_s=0.0,
            event_type="block_start",
            block_id="B1",
            task="n-back",
            level=1,
        )
    )
    recorder.log_nasa_tlx(
        NasaTlxResponse(
            block_id="B1",
            mental=40.0,
            physical=20.0,
            temporal=30.0,
            performance=80.0,
            effort=45.0,
            frustration=20.0,
            raw_tlx=39.1,
        )
    )

    # 3. Finalize
    final_dir = recorder.finalize_session()
    assert (final_dir / "signals.parquet").exists()
    assert (final_dir / "events.csv").exists()
    assert (final_dir / "nasa_tlx.csv").exists()

    # Verify contents
    sig_df = pd.read_parquet(final_dir / "signals.parquet")
    assert len(sig_df) == 1
    events_df = pd.read_csv(final_dir / "events.csv")
    assert len(events_df) == 1

    # 4. Participant deletion (FR-COL-7)
    deleted = SessionRecorder.delete_participant_data("P999", base_dir=tmp_path)
    assert deleted is True
    assert not (tmp_path / "P999").exists()
