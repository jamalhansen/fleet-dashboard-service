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
        core.ModelUsage(tool_name="japanese-tutor", model="phi4-mini", provider="ollama", total=8, failures=2),
        core.ModelUsage(tool_name="japanese-tutor", model="deepseek-chat", provider="deepseek", total=2, failures=0),
    ]
    by_provider = [
        core.ProviderUsage(provider="ollama", total=8, failures=2, tool_count=1, model_count=1),
        core.ProviderUsage(provider="deepseek", total=2, failures=0, tool_count=1, model_count=1),
    ]
    with patch("fleet_dashboard.server.core.get_model_usage", return_value=usage), \
         patch("fleet_dashboard.server.core.get_provider_usage", return_value=by_provider):
        resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"][0]["tool_name"] == "japanese-tutor"
    assert body["usage"][0]["provider"] == "ollama"
    assert body["usage"][0]["model"] == "phi4-mini"
    assert body["usage"][0]["failure_rate"] == 0.25
    assert body["usage"][1]["model"] == "deepseek-chat"
    assert body["by_provider"][0]["provider"] == "ollama"
    assert body["by_provider"][0]["tool_count"] == 1
    assert body["by_provider"][1]["provider"] == "deepseek"


def test_tensions_endpoint_shape():
    summary = core.TensionSummary(pending_count=3, active_count=1, recent_titles=["A tension"])
    with patch("fleet_dashboard.server.core.get_tension_summary", return_value=summary):
        resp = client.get("/api/tensions")
    assert resp.status_code == 200
    assert resp.json() == {"pending_count": 3, "active_count": 1, "recent_titles": ["A tension"]}


def test_vault_health_endpoint_shape():
    health = core.VaultHealth(observations_pending=7, inbox_count=153, inbox_oldest_days=12.345, last_health_check="2026-09-06")
    with patch("fleet_dashboard.server.core.get_vault_health", return_value=health):
        resp = client.get("/api/vault-health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["observations_pending"] == 7
    assert body["inbox_count"] == 153
    assert body["inbox_oldest_days"] == 12.3
    assert body["last_health_check"] == "2026-09-06"


def test_vault_health_endpoint_handles_no_inbox_files():
    health = core.VaultHealth(observations_pending=0, inbox_count=0, inbox_oldest_days=None, last_health_check=None)
    with patch("fleet_dashboard.server.core.get_vault_health", return_value=health):
        resp = client.get("/api/vault-health")
    assert resp.json()["inbox_oldest_days"] is None


def test_frontmatter_validation_endpoint_unavailable():
    summary = core.FrontmatterValidationSummary(available=False)
    with patch("fleet_dashboard.server.core.get_frontmatter_validation", return_value=summary):
        resp = client.get("/api/frontmatter-validation")
    assert resp.status_code == 200
    assert resp.json() == {"available": False}


def test_frontmatter_validation_endpoint_available():
    summary = core.FrontmatterValidationSummary(
        available=True, generated_at="2026-09-20T12:00:00Z", total=10, invalid_count=2,
        invalid_files=[{"file": "a.md", "errors": ["x"]}],
        error_summary=[{"error": "x", "count": 2, "files": ["a.md", "b.md"], "more": 0}],
    )
    with patch("fleet_dashboard.server.core.get_frontmatter_validation", return_value=summary):
        resp = client.get("/api/frontmatter-validation")
    body = resp.json()
    assert body["available"] is True
    assert body["total"] == 10
    assert body["invalid_count"] == 2
    assert body["invalid_files"] == [{"file": "a.md", "errors": ["x"]}]
    assert body["error_summary"] == [{"error": "x", "count": 2, "files": ["a.md", "b.md"], "more": 0}]


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
