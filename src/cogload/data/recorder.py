"""Session data recorder: manages signals, events, NASA-TLX, and deletion (PRD Section 9.2)."""

import json
from pathlib import Path
import shutil
from typing import Optional, List, Dict, Any
import pandas as pd
from cogload.data.schema import SessionMeta, TaskEvent, NasaTlxResponse
from cogload.calibration.profile import CalibrationProfile


class SessionRecorder:
    """Records session data to data/sessions/<participant_id>/<session_id>/."""

    def __init__(self, base_dir: str | Path = "data/sessions"):
        self.base_dir = Path(base_dir)
        self.current_meta: Optional[SessionMeta] = None
        self.session_dir: Optional[Path] = None
        self.signal_rows: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.tlx_records: List[Dict[str, Any]] = []
        self.calibration: Optional[CalibrationProfile] = None

    def start_session(self, meta: SessionMeta) -> Path:
        self.current_meta = meta
        self.session_dir = self.base_dir / meta.participant_id / meta.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.signal_rows.clear()
        self.events.clear()
        self.tlx_records.clear()

        # Write meta.json
        meta_dict = {
            "participant_id": meta.participant_id,
            "session_id": meta.session_id,
            "created_at": meta.created_at,
            "glasses": meta.glasses,
            "lighting": meta.lighting,
            "camera_name": meta.camera_name,
            "approx_distance_cm": meta.approx_distance_cm,
            "notes": meta.notes,
        }
        with open(self.session_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        return self.session_dir

    def log_signal_row(self, row_dict: Dict[str, Any]) -> None:
        self.signal_rows.append(row_dict)

    def log_event(self, event: TaskEvent) -> None:
        self.events.append({
            "timestamp_s": event.timestamp_s,
            "event_type": event.event_type,
            "block_id": event.block_id,
            "task": event.task,
            "level": event.level,
            "payload_json": event.payload_json,
        })

    def log_nasa_tlx(self, tlx: NasaTlxResponse) -> None:
        self.tlx_records.append({
            "block_id": tlx.block_id,
            "mental": tlx.mental,
            "physical": tlx.physical,
            "temporal": tlx.temporal,
            "performance": tlx.performance,
            "effort": tlx.effort,
            "frustration": tlx.frustration,
            "raw_tlx": tlx.raw_tlx,
        })

    def set_calibration(self, profile: CalibrationProfile) -> None:
        self.calibration = profile
        if self.session_dir:
            profile.save_json(self.session_dir / "calibration.json")

    def finalize_session(self) -> Path:
        """Flush in-memory signals and event tables to disk."""
        if self.session_dir is None:
            raise RuntimeError("No active session to finalize.")

        # Save signals.parquet
        if self.signal_rows:
            signals_df = pd.DataFrame(self.signal_rows)
            signals_df.to_parquet(self.session_dir / "signals.parquet", index=False)
        else:
            empty_df = pd.DataFrame(columns=["t", "valid", "ear", "gaze_x", "gaze_y", "yaw", "pitch", "roll"])
            empty_df.to_parquet(self.session_dir / "signals.parquet", index=False)

        # Save events.csv
        events_df = pd.DataFrame(self.events) if self.events else pd.DataFrame(
            columns=["timestamp_s", "event_type", "block_id", "task", "level", "payload_json"]
        )
        events_df.to_csv(self.session_dir / "events.csv", index=False)

        # Save nasa_tlx.csv
        tlx_df = pd.DataFrame(self.tlx_records) if self.tlx_records else pd.DataFrame(
            columns=["block_id", "mental", "physical", "temporal", "performance", "effort", "frustration", "raw_tlx"]
        )
        tlx_df.to_csv(self.session_dir / "nasa_tlx.csv", index=False)

        # Save calibration profile if present
        if self.calibration is not None:
            self.calibration.save_json(self.session_dir / "calibration.json")

        final_path = self.session_dir
        self.session_dir = None
        self.current_meta = None
        return final_path

    @staticmethod
    def delete_participant_data(participant_id: str, base_dir: str | Path = "data/sessions") -> bool:
        """GDPR/Ethics compliance: completely delete all data for a participant (FR-COL-7)."""
        target = Path(base_dir) / participant_id
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            return True
        return False
