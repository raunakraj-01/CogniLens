"""Flask route handlers and REST / SSE API definitions (PRD Section 12)."""

import json
import time
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    Response,
    send_from_directory,
)
from cogload.calibration.protocol import CalibrationRunner
from cogload.data.schema import SessionMeta, TaskEvent, NasaTlxResponse
from cogload.data.recorder import SessionRecorder


def register_routes(app: Flask) -> None:
    state = app.state  # type: ignore

    # 1. HTML Pages
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/calibrate")
    def calibrate_page():
        return render_template("calibrate.html")

    @app.route("/collect")
    def collect_page():
        return render_template("collect.html")

    # 2. SSE Live Stream (PRD Section 12.3)
    @app.route("/api/stream")
    def api_stream():
        def generate():
            while True:
                with state.lock:
                    res = state.latest_result
                if res is not None:
                    data = res.to_sse_dict()
                else:
                    data = {
                        "t": time.time(),
                        "face_detected": False,
                        "fps": round(state.capture_thread.effective_fps, 1),
                        "calibrated": bool(state.engine.calibration and state.engine.calibration.is_valid()),
                        "load": {"level": "LOW", "probs": {"low": 1.0, "medium": 0.0, "high": 0.0}, "score": 0.0, "confidence": 0.0},
                        "indicators": {
                            "blink_rate": 0.0,
                            "ear": 0.0,
                            "gaze_dispersion": 0.0,
                            "head_motion": 0.0,
                            "z": {"blink_rate": 0.0, "gaze_dispersion": 0.0, "head_motion": 0.0},
                        },
                        "warnings": ["no_face"] if not getattr(state, "latest_frame", None) is not None else [],
                    }
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1.0)

        return Response(generate(), mimetype="text/event-stream")

    # 3. Live Video Preview Feed (MJPEG with standard WebKit Content-Length headers, PRD 12.2)
    @app.route("/api/video_feed")
    def video_feed():
        import cv2
        from scripts.debug_overlay import draw_debug_overlay
        show_overlay = request.args.get("overlay", "true").lower() == "true"
        is_testing = bool(app.config.get("TESTING", False))

        def gen_frames():
            while True:
                with state.lock:
                    frame = state.latest_frame
                    res = state.latest_landmark_res
                if frame is not None:
                    display = frame.copy()
                    if show_overlay and res is not None:
                        display = draw_debug_overlay(display, res)
                    ret, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ret:
                        payload = buf.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
                            + payload + b"\r\n"
                        )
                if is_testing:
                    break
                time.sleep(0.04)

        resp = Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    # 3b. Universal Single-Frame Snapshot (Works on 100% of browsers including Safari)
    @app.route("/api/frame")
    def single_frame():
        import cv2
        import numpy as np
        from scripts.debug_overlay import draw_debug_overlay
        show_overlay = request.args.get("overlay", "true").lower() == "true"
        with state.lock:
            frame = state.latest_frame
            res = state.latest_landmark_res
        if frame is None:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buf = cv2.imencode(".jpg", empty)
            return Response(buf.tobytes(), mimetype="image/jpeg")

        display = frame.copy()
        if show_overlay and res is not None:
            display = draw_debug_overlay(display, res)
        ret, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        resp = Response(buf.tobytes(), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp

    # 4. System Status
    @app.route("/api/status")
    def api_status():
        with state.lock:
            res = state.latest_result
        return jsonify({
            "camera_ok": state.capture_thread._running,
            "fps": round(state.capture_thread.effective_fps, 1),
            "face_detected": res.face_detected if res else False,
            "calibrated": bool(state.engine.calibration and state.engine.calibration.is_valid()),
            "model_loaded": state.engine.model is not None,
            "is_recording": state.is_recording,
        })

    # 5. Calibration API
    @app.route("/api/calibration/start", methods=["POST"])
    def api_calibration_start():
        data = request.get_json(silent=True) or {}
        participant_id = data.get("participant_id", "P_DEMO")
        dur_sec = float(data.get("duration_sec", 30.0))
        from cogload.config import CalibrationConfig
        calib_cfg = CalibrationConfig(
            phase_a_sec=dur_sec * 0.67,
            phase_b_sec=dur_sec * 0.33,
            min_valid_ratio=0.75,
        )
        runner = CalibrationRunner(
            participant_id=participant_id,
            calib_cfg=calib_cfg,
            feat_cfg=state.cfg.features,
        )
        with state.lock:
            state.calibration_runner = runner
        return jsonify({"status": "started", "participant_id": participant_id, "duration_sec": dur_sec})

    @app.route("/api/calibration/status")
    def api_calibration_status():
        with state.lock:
            runner = state.calibration_runner
        if runner is None:
            return jsonify({"status": "idle", "calibrated": False})

        elapsed = runner.elapsed_seconds
        target_sec = runner.calib_cfg.phase_a_sec + runner.calib_cfg.phase_b_sec
        progress = min(1.0, elapsed / target_sec) if target_sec > 0 else 1.0

        if progress >= 1.0:
            profile, msg = runner.finish()
            with state.lock:
                state.calibration_runner = None
                if profile:
                    state.engine.set_calibration(profile)
            return jsonify({
                "status": "completed" if profile else "failed",
                "progress": 1.0,
                "passed": bool(profile and profile.is_valid()),
                "message": msg,
            })

        return jsonify({
            "status": "in_progress",
            "progress": round(progress, 2),
            "elapsed_seconds": round(elapsed, 1),
            "target_seconds": round(target_sec, 1),
        })

    # 5. Session Data Collection API
    @app.route("/api/session/start", methods=["POST"])
    def api_session_start():
        body = request.get_json(silent=True) or {}
        pid = body.get("participant_id", "P001")
        sid = body.get("session_id", "S01")
        meta = SessionMeta(
            participant_id=pid,
            session_id=sid,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            glasses=bool(body.get("glasses", False)),
            lighting=body.get("lighting", "normal"),
            camera_name=body.get("camera_name", "webcam"),
            notes=body.get("notes", ""),
        )
        with state.lock:
            state.recorder.start_session(meta)
            if state.engine.calibration:
                state.recorder.set_calibration(state.engine.calibration)
            state.is_recording = True
        return jsonify({"status": "recording_started", "session_dir": str(state.recorder.session_dir)})

    @app.route("/api/session/stop", methods=["POST"])
    def api_session_stop():
        with state.lock:
            if not state.is_recording:
                return jsonify({"status": "error", "message": "No active recording"}), 400
            final_p = state.recorder.finalize_session()
            state.is_recording = False
        return jsonify({"status": "recording_finalized", "saved_path": str(final_p)})

    @app.route("/api/events", methods=["POST"])
    def api_events():
        body = request.get_json(silent=True) or {}
        ev = TaskEvent(
            timestamp_s=time.monotonic(),  # Server-side monotonic clock domain
            event_type=body.get("event_type", "stimulus"),
            block_id=body.get("block_id", "B1"),
            task=body.get("task", "n-back"),
            level=int(body.get("level", 1)),
            payload_json=json.dumps(body.get("payload", {})),
        )
        with state.lock:
            if state.is_recording:
                state.recorder.log_event(ev)
        return jsonify({"status": "logged", "timestamp_s": ev.timestamp_s})

    @app.route("/api/nasa-tlx", methods=["POST"])
    def api_nasa_tlx():
        body = request.get_json(silent=True) or {}
        tlx = NasaTlxResponse(
            block_id=body.get("block_id", "B1"),
            mental=float(body.get("mental", 50)),
            physical=float(body.get("physical", 20)),
            temporal=float(body.get("temporal", 30)),
            performance=float(body.get("performance", 80)),
            effort=float(body.get("effort", 50)),
            frustration=float(body.get("frustration", 30)),
            raw_tlx=float(body.get("raw_tlx", 50.0)),
        )
        with state.lock:
            if state.is_recording:
                state.recorder.log_nasa_tlx(tlx)
        return jsonify({"status": "tlx_saved"})

    # 6. Model Card Summary
    @app.route("/api/model")
    def api_model():
        card = state.engine.model_card or {"error": "No model loaded"}
        return jsonify(card)

    # 7. Participant Deletion (FR-COL-7)
    @app.route("/api/participant/<pid>", methods=["DELETE"])
    def api_delete_participant(pid):
        confirm = request.args.get("confirm", "false").lower() == "true"
        if not confirm:
            return jsonify({"error": "Confirmation parameter ?confirm=true required"}), 400
        deleted = SessionRecorder.delete_participant_data(pid)
        return jsonify({"deleted": deleted, "participant_id": pid})
