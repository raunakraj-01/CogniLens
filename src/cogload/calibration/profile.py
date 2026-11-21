"""CalibrationProfile dataclass and JSON serialization (PRD Section 8.2)."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class CalibrationProfile:
    participant_id: str
    created_at: str
    baseline_open_ear: float
    gaze_center: Dict[str, float]
    head_neutral: Dict[str, float]
    features: Dict[str, Dict[str, float]]  # feature_name -> {"mu": float, "sigma": float}
    quality: Dict[str, Any]                # {"valid_ratio": float, "n_windows": int, "passed": bool, ...}

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "CalibrationProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def is_valid(self) -> bool:
        return bool(self.quality.get("passed", False))
