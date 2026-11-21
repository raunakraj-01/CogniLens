"""Offline video processing script: video -> signals.parquet (PRD Section 16 Phase 2)."""

import argparse
from pathlib import Path
import cv2
import pandas as pd
from cogload.capture.source import FileVideoSource
from cogload.vision.landmarker import FaceLandmarkerWrapper
from cogload.signals.frame_signals import (
    FrameSignalExtractor,
    interpolate_short_gaps,
    smooth_gaze_signals,
)


def process_video_to_signals(
    video_path: str | Path,
    output_parquet: str | Path,
    max_gap_interpolate: int = 3,
) -> pd.DataFrame:
    """Process video file and save extracted per-frame signals to Parquet."""
    source = FileVideoSource(video_path)
    landmarker = FaceLandmarkerWrapper()
    extractor = FrameSignalExtractor()

    rows = []
    frame_idx = 0

    while True:
        success, frame, ts = source.read()
        if not success or frame is None:
            break

        res = landmarker.process(frame)
        row = extractor.process_frame(t=ts, landmark_result=res, image_shape=frame.shape[:2])
        rows.append(row.to_dict())
        frame_idx += 1

    source.release()
    landmarker.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = interpolate_short_gaps(df, max_gap=max_gap_interpolate)
        df = smooth_gaze_signals(df)

    out_p = Path(output_parquet)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    print(f"Processed {frame_idx} frames. Output saved to {out_p}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Process video file to signals.parquet")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    process_video_to_signals(args.video, args.output)


if __name__ == "__main__":
    main()
