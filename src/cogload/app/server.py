"""Flask application server and live background inference worker (PRD Section 12)."""

import argparse
from pathlib import Path
import threading
import time
from typing import Optional
from flask import Flask
from cogload.config import AppConfig, load_config
from cogload.capture.source import WebcamSource, FileVideoSource, BaseVideoSource
from cogload.capture.thread import CaptureThread
from cogload.realtime.engine import RealtimeInferenceEngine, InferenceResult
from cogload.calibration.protocol import CalibrationRunner
from cogload.data.recorder import SessionRecorder


class AppState:
    """Shared state between background video worker and Flask routes."""

    def __init__(
        self,
        cfg: AppConfig,
        video_source_path: Optional[str] = None,
        testing: bool = False,
    ):
        self.cfg = cfg
        self.testing = testing
        self.lock = threading.Lock()
        self.latest_result: Optional[InferenceResult] = None
        self.latest_frame = None
        self.latest_landmark_res = None

        # Sources
        if testing or video_source_path == "synthetic":
            from cogload.capture.source import ReplaySource
            import numpy as np
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60)]
            self.source: BaseVideoSource = ReplaySource(frames=frames, fps=30.0)
        elif video_source_path:
            self.source = FileVideoSource(video_source_path)
        else:
            try:
                self.source = WebcamSource(
                    index=cfg.camera.index,
                    width=cfg.camera.width,
                    height=cfg.camera.height,
                    target_fps=cfg.camera.target_fps,
                )
            except Exception as e:
                print(f"Warning: Webcam {cfg.camera.index} unavailable ({e}). Using synthetic mock.")
                from cogload.capture.source import ReplaySource
                import numpy as np
                frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60)]
                self.source = ReplaySource(frames=frames, fps=30.0)

        self.capture_thread = CaptureThread(self.source, max_queue=cfg.processing.max_queue)
        self.engine = RealtimeInferenceEngine(cfg=cfg)
        self.recorder = SessionRecorder()
        self.calibration_runner: Optional[CalibrationRunner] = None
        self.is_recording = False
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def start_worker(self) -> None:
        if self._running:
            return
        self._running = True
        self.capture_thread.start()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="InferenceWorker")
        self._worker_thread.start()

    def _run_loop(self) -> None:
        while self._running:
            pkt = self.capture_thread.get_latest(timeout=0.05)
            if pkt is None:
                continue

            # Process frame through inference engine
            inf_res, landmark_res = self.engine.process_frame(pkt.frame, timestamp_sec=pkt.timestamp_sec)

            with self.lock:
                self.latest_frame = pkt.frame
                self.latest_landmark_res = landmark_res
                if inf_res is not None:
                    self.latest_result = inf_res

                # Record frame signals if session active
                if self.is_recording and landmark_res is not None:
                    row = self.engine.signal_extractor.process_frame(
                        t=pkt.timestamp_sec,
                        landmark_result=landmark_res,
                        image_shape=pkt.frame.shape[:2],
                    )
                    self.recorder.log_signal_row(row.to_dict())

                # Accumulate calibration frames if calibrating
                if self.calibration_runner is not None and landmark_res is not None:
                    row = self.engine.signal_extractor.process_frame(
                        t=pkt.timestamp_sec,
                        landmark_result=landmark_res,
                        image_shape=pkt.frame.shape[:2],
                    )
                    self.calibration_runner.add_signal_row(row.to_dict())

    def stop_worker(self) -> None:
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self.capture_thread.stop()
        self.engine.close()


def create_app(
    cfg: Optional[AppConfig] = None,
    video_source: Optional[str] = None,
    testing: bool = False,
) -> Flask:
    """Flask application factory."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["TESTING"] = testing

    cfg = cfg or load_config()
    state = AppState(cfg=cfg, video_source_path=video_source, testing=testing)
    if not testing:
        state.start_worker()

    app.state = state  # type: ignore

    from cogload.app.routes import register_routes
    register_routes(app)

    return app


def main():
    parser = argparse.ArgumentParser(description="Run Cogload Web Dashboard")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for replay")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5050, help="Server port")
    args = parser.parse_args()

    cfg = load_config()
    cfg.server.host = args.host
    cfg.server.port = args.port

    app = create_app(cfg=cfg, video_source=args.video)
    print(f"\n========================================================")
    print(f"  Cogload Cognitive Load Dashboard running on:")
    print(f"  http://{args.host}:{args.port}")
    print(f"========================================================\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
