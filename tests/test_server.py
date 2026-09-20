from unittest.mock import patch

from fastapi.testclient import TestClient

from fleet_dashboard import core
from fleet_dashboard.server import app

client = TestClient(app)


def test_fleet_endpoint_shape():
    activity = [core.ToolActivity(tool_name="my-tool", total=5, failures=1, last_call="2026-09-20", tables=["processing_log"])]
    services = [core.ServiceStatus(label="com.localfirst.my-tool", running=True, pid=1, keep_alive=False, last_exit_code=0)]
    with patch("fleet_dashboard.server.core.get_fleet_activity", return_value=activity), \
         patch("fleet_dashboard.server.core.get_launch_agents", return_value=services):
        resp = client.get("/api/fleet")
    assert resp.status_code == 200
    body = resp.json()
    assert body["activity"][0]["tool_name"] == "my-tool"
    assert body["activity"][0]["failure_rate"] == 0.2
    assert body["services"][0]["running"] is True


def test_models_endpoint_shape():
    usage = [
        core.ModelUsage(tool_name="japanese-tutor", model="phi4-mini", total=8, failures=2),
        core.ModelUsage(tool_name="japanese-tutor", model="deepseek-chat", total=2, failures=0),
    ]
    with patch("fleet_dashboard.server.core.get_model_usage", return_value=usage):
        resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"][0]["tool_name"] == "japanese-tutor"
    assert body["usage"][0]["model"] == "phi4-mini"
    assert body["usage"][0]["failure_rate"] == 0.25
    assert body["usage"][1]["model"] == "deepseek-chat"


def test_tensions_endpoint_shape():
    summary = core.TensionSummary(pending_count=3, active_count=1, recent_titles=["A tension"])
    with patch("fleet_dashboard.server.core.get_tension_summary", return_value=summary):
        resp = client.get("/api/tensions")
    assert resp.status_code == 200
    assert resp.json() == {"pending_count": 3, "active_count": 1, "recent_titles": ["A tension"]}


def test_art_endpoint_no_art_available():
    with patch("fleet_dashboard.server.core.get_latest_art", return_value=None):
        resp = client.get("/api/art")
    assert resp.status_code == 200
    assert resp.json() == {"available": False}


def test_art_image_404_when_missing():
    with patch("fleet_dashboard.server.core.get_latest_art", return_value=None):
        resp = client.get("/api/art/image")
    assert resp.status_code == 404


def test_japanese_tutor_endpoint_unreachable():
    summary = core.JapaneseTutorSummary(reachable=False)
    with patch("fleet_dashboard.server.core.get_japanese_tutor_summary", return_value=summary):
        resp = client.get("/api/japanese-tutor")
    assert resp.status_code == 200
    assert resp.json() == {"reachable": False, "cards_due": 0, "mastery": []}
