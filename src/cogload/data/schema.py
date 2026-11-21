"""Data storage schemas, validators, and serialization contracts (PRD Section 9)."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass
class SessionMeta:
    participant_id: str
    session_id: str
    created_at: str
    glasses: bool
    lighting: str  # 'bright' | 'normal' | 'dim'
    camera_name: str
    approx_distance_cm: float = 60.0
    notes: str = ""


@dataclass
class TaskEvent:
    timestamp_s: float
    event_type: str  # 'block_start' | 'block_stop' | 'stimulus' | 'response' | 'rest_start' | 'rest_stop'
    block_id: str
    task: str        # 'n-back' | 'arithmetic'
    level: int       # 1, 2, 3 or 0 for rest
    payload_json: str = "{}"


@dataclass
class NasaTlxResponse:
    block_id: str
    mental: float       # 0..100
    physical: float     # 0..100
    temporal: float     # 0..100
    performance: float  # 0..100
    effort: float       # 0..100
    frustration: float  # 0..100
    raw_tlx: float      # Mean of 6 dimensions
