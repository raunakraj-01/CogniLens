"""Flask test client suite testing all routes, API endpoints, and schemas."""

import json
from cogload.app.server import create_app


def test_flask_routes_and_api():
    app = create_app(testing=True)
    client = app.test_client()

    # 1. Page routes
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"behavioral proxies from the webcam" in resp.data

    resp_calib = client.get("/calibrate")
    assert resp_calib.status_code == 200

    resp_collect = client.get("/collect")
    assert resp_collect.status_code == 200

    # 2. Status API
    resp_video = client.get("/api/video_feed")
    assert resp_video.status_code == 200

    resp_status = client.get("/api/status")
    assert resp_status.status_code == 200
    status_data = resp_status.get_json()
    assert "face_detected" in status_data
    assert "calibrated" in status_data

    # 3. Calibration API
    resp_calib_start = client.post(
        "/api/calibration/start",
        data=json.dumps({"participant_id": "P_TEST"}),
        content_type="application/json",
    )
    assert resp_calib_start.status_code == 200

    resp_calib_status = client.get("/api/calibration/status")
    assert resp_calib_status.status_code == 200

    # 4. Session Start & Stop
    resp_sess_start = client.post(
        "/api/session/start",
        data=json.dumps({"participant_id": "P_TEST_APP", "lighting": "normal"}),
        content_type="application/json",
    )
    assert resp_sess_start.status_code == 200

    # Events API
    resp_ev = client.post(
        "/api/events",
        data=json.dumps({"event_type": "stimulus", "block_id": "B1", "level": 1}),
        content_type="application/json",
    )
    assert resp_ev.status_code == 200

    # TLX API
    resp_tlx = client.post(
        "/api/nasa-tlx",
        data=json.dumps({"block_id": "B1", "mental": 50, "raw_tlx": 45.0}),
        content_type="application/json",
    )
    assert resp_tlx.status_code == 200

    resp_sess_stop = client.post("/api/session/stop")
    assert resp_sess_stop.status_code == 200

    # Participant deletion
    resp_del = client.delete("/api/participant/P_TEST_APP?confirm=true")
    assert resp_del.status_code == 200
