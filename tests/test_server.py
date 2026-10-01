from unittest.mock import patch

from fastapi.testclient import TestClient

from fleet_dashboard import core, server
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
        core.ModelUsage(tool_name="japanese-tutor", model="phi4-mini", provider="ollama", total=8, failures=2, via_gateway=True),
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
    assert body["usage"][0]["via_gateway"] is True
    assert body["usage"][1]["model"] == "deepseek-chat"
    assert body["usage"][1]["via_gateway"] is False
    assert body["by_provider"][0]["provider"] == "ollama"
    assert body["by_provider"][0]["tool_count"] == 1
    assert body["by_provider"][1]["provider"] == "deepseek"


def test_fetches_endpoint_shape():
    usage = [
        core.FetchUsage(tool_name="http-retriever-service", domain="arxiv.org", total=12, failures=1, avg_duration_ms=250.4, last_call="2026-09-21 07:07:42"),
    ]
    with patch("fleet_dashboard.server.core.get_fetch_usage", return_value=usage):
        resp = client.get("/api/fetches")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"][0]["domain"] == "arxiv.org"
    assert body["usage"][0]["total"] == 12
    assert body["usage"][0]["failure_rate"] == round(1 / 12, 4)
    assert body["usage"][0]["avg_duration_ms"] == 250


def test_api_calls_endpoint_shape():
    usage = [
        core.ApiCallUsage(tool_name="content-discovery-agent", service="readwise", operation="list_highlights", total=5, failures=0, last_call="2026-09-21 07:00:00"),
    ]
    with patch("fleet_dashboard.server.core.get_api_call_usage", return_value=usage):
        resp = client.get("/api/api-calls")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"][0]["service"] == "readwise"
    assert body["usage"][0]["operation"] == "list_highlights"
    assert body["usage"][0]["total"] == 5


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


def test_vault_health_endpoint_accepts_keysix():
    """Regression 2026-09-20: vault-health/tensions were hardcoded to
    Contexta only -- Jamal's other vault (KeySix) was invisible."""
    health = core.VaultHealth(observations_pending=1, inbox_count=0, inbox_oldest_days=None, last_health_check=None)
    with patch("fleet_dashboard.server.core.get_vault_health", return_value=health) as mock_get:
        resp = client.get("/api/vault-health?vault=KeySix")
    assert resp.status_code == 200
    mock_get.assert_called_with(server.VAULTS["KeySix"])


def test_tensions_endpoint_accepts_keysix():
    summary = core.TensionSummary(pending_count=0, active_count=0, recent_titles=[])
    with patch("fleet_dashboard.server.core.get_tension_summary", return_value=summary) as mock_get:
        resp = client.get("/api/tensions?vault=KeySix")
    assert resp.status_code == 200
    mock_get.assert_called_with(server.VAULTS["KeySix"])


def test_vault_health_endpoint_rejects_unknown_vault():
    resp = client.get("/api/vault-health?vault=NotAVault")
    assert resp.status_code == 404


def test_repo_health_endpoint_unavailable():
    summary = core.RepoHealthSummary(available=False)
    with patch("fleet_dashboard.server.core.get_repo_health", return_value=summary):
        resp = client.get("/api/repo-health")
    assert resp.status_code == 200
    assert resp.json() == {"available": False}


def test_repo_health_endpoint_available():
    summary = core.RepoHealthSummary(
        available=True,
        generated_at="2026-09-20T12:00:00Z",
        total=2,
        healthy=1,
        repos=[{"name": "bad-repo", "ok": False, "lint_ok": False, "lint_errors": 3,
                "tests_ok": True, "tests_passed": 1, "tests_failed": 0,
                "hooks_ok": True, "dirty": False, "unpushed": 0}],
    )
    with patch("fleet_dashboard.server.core.get_repo_health", return_value=summary):
        resp = client.get("/api/repo-health")
    body = resp.json()
    assert body["available"] is True
    assert body["healthy"] == 1
    assert body["repos"][0]["name"] == "bad-repo"


def test_gateway_routing_endpoint_available():
    summary = core.GatewayRoutingSummary(
        available=True,
        generated_at="2026-09-21T06:00:00Z",
        entries=[core.GatewayRoutingEntry(tool_name="pebble", category="direct_unclassified", status="review")],
    )
    with patch("fleet_dashboard.server.core.get_gateway_routing_audit", return_value=summary):
        resp = client.get("/api/gateway-routing")
    body = resp.json()
    assert body["available"] is True
    assert body["entries"][0]["tool_name"] == "pebble"
    assert body["entries"][0]["status"] == "review"


def test_gateway_routing_endpoint_unavailable():
    summary = core.GatewayRoutingSummary(available=False)
    with patch("fleet_dashboard.server.core.get_gateway_routing_audit", return_value=summary):
        resp = client.get("/api/gateway-routing")
    assert resp.json() == {"available": False}


def test_writing_endpoint_unavailable():
    with patch("fleet_dashboard.server.core.get_writing_cadence", return_value=core.WritingCadence(available=False)):
        resp = client.get("/api/writing")
    assert resp.json() == {"available": False}


def test_writing_endpoint_available():
    cadence = core.WritingCadence(
        available=True, last_published="2026-07-24", last_published_title="Go Hybrid", days_since_last=62,
        weeks=[{"week_of": "2026-09-21", "posts": 0}], weeks_on_target=0,
        pipeline={"draft": 1, "outline": 33, "idea": 12, "brainstorm": 25},
        freshest_draft={"name": "my-test-suite-tried-to-brew-install", "modified": "2026-09-24"},
    )
    with patch("fleet_dashboard.server.core.get_writing_cadence", return_value=cadence):
        body = client.get("/api/writing").json()
    assert body["available"] is True
    assert body["days_since_last"] == 62
    assert body["pipeline"]["outline"] == 33
    assert body["freshest_draft"]["name"] == "my-test-suite-tried-to-brew-install"


def test_blog_validation_endpoint_unavailable():
    summary = core.BlogValidationSummary(available=False)
    with patch("fleet_dashboard.server.core.get_blog_validation", return_value=summary):
        resp = client.get("/api/blog-validation")
    assert resp.status_code == 200
    assert resp.json() == {"available": False}


def test_blog_validation_endpoint_available():
    target = {
        "name": "vault", "posts": 137, "posts_passed": 97, "posts_failed": 40,
        "blocks_passed": 245, "blocks_failed": 85, "blocks_skipped": 121,
        "fully_covered": 8, "needs_attention": 97, "assertion_pct": 13,
        "failed_posts": [{"slug": "03a-find-errors-with-grep", "errors": ["block 0 (bash): exit 1"]}],
        "more_failed": 0,
    }
    summary = core.BlogValidationSummary(
        available=True, generated_at="2026-09-23T12:00:00Z", targets=[target],
        posts=137, posts_failed=40, blocks_failed=85,
    )
    with patch("fleet_dashboard.server.core.get_blog_validation", return_value=summary):
        resp = client.get("/api/blog-validation")
    body = resp.json()
    assert body["available"] is True
    assert body["posts"] == 137
    assert body["posts_failed"] == 40
    assert body["blocks_failed"] == 85
    assert body["targets"] == [target]


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
        content_todo=[{"key": "needs_status", "label": "Needs a status decision", "count": 1, "files": ["a.md"], "more": 0}],
    )
    with patch("fleet_dashboard.server.core.get_frontmatter_validation", return_value=summary):
        resp = client.get("/api/frontmatter-validation")
    body = resp.json()
    assert body["available"] is True
    assert body["total"] == 10
    assert body["invalid_count"] == 2
    assert body["invalid_files"] == [{"file": "a.md", "errors": ["x"]}]
    assert body["error_summary"] == [{"error": "x", "count": 2, "files": ["a.md", "b.md"], "more": 0}]
    assert body["content_todo"] == [{"key": "needs_status", "label": "Needs a status decision", "count": 1, "files": ["a.md"], "more": 0}]


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
    assert resp.json() == {
        "reachable": False,
        "cards_due": 0,
        "new_count": 0,
        "review_count": 0,
        "mastery": [],
        "reviews_today_attempts": 0,
        "reviews_today_distinct_cards": 0,
    }


def test_japanese_tutor_endpoint_includes_reviews_today():
    summary = core.JapaneseTutorSummary(
        reachable=True,
        cards_due=3,
        new_count=2,
        review_count=1,
        mastery=[],
        reviews_today_attempts=7,
        reviews_today_distinct_cards=5,
    )
    with patch("fleet_dashboard.server.core.get_japanese_tutor_summary", return_value=summary):
        resp = client.get("/api/japanese-tutor")
    assert resp.status_code == 200
    body = resp.json()
    assert body["new_count"] == 2
    assert body["review_count"] == 1
    assert body["reviews_today_attempts"] == 7
    assert body["reviews_today_distinct_cards"] == 5


def test_art_is_blind_until_rated():
    from pathlib import Path

    from fleet_dashboard.core import ArtItem

    unrated = ArtItem("t", 0.7, "X", "2026-10-02", Path("/x.png"), human_score=None, artist="mentored")
    with patch("fleet_dashboard.server.core.get_latest_art", return_value=unrated):
        body = client.get("/api/art").json()
    assert body["rated"] is False and body["artist"] is None and body["self_score"] is None
    rated = ArtItem("t", 0.7, "X", "2026-10-02", Path("/x.png"), human_score=0.75, artist="mentored")
    with patch("fleet_dashboard.server.core.get_latest_art", return_value=rated):
        body = client.get("/api/art").json()
    assert (body["artist"], body["self_score"], body["human_score"]) == ("mentored", 0.7, 0.75)
