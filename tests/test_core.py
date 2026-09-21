import json
import plistlib
from unittest.mock import MagicMock, patch

import duckdb

from fleet_dashboard import core


def _seed_db(db_path, processing_rows=(), fetch_rows=(), api_call_rows=()):
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, success BOOLEAN, "
            "via_gateway BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("CREATE TABLE tools (id INTEGER, name VARCHAR)")
        conn.execute(
            "CREATE TABLE fetch_log (tool_id INTEGER, success BOOLEAN, "
            "attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE api_call_log (tool_id INTEGER, success BOOLEAN, "
            "attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        for name, success in processing_rows:
            conn.execute("INSERT INTO processing_log (tool_name, success) VALUES (?, ?)", [name, success])

        tool_ids, next_id = {}, 1
        for name, _ in list(fetch_rows) + list(api_call_rows):
            if name not in tool_ids:
                tool_ids[name] = next_id
                conn.execute("INSERT INTO tools (id, name) VALUES (?, ?)", [next_id, name])
                next_id += 1
        for name, success in fetch_rows:
            conn.execute("INSERT INTO fetch_log (tool_id, success) VALUES (?, ?)", [tool_ids[name], success])
        for name, success in api_call_rows:
            conn.execute("INSERT INTO api_call_log (tool_id, success) VALUES (?, ?)", [tool_ids[name], success])
    finally:
        conn.close()


class TestGetFleetActivity:
    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(tmp_path / "nope.duckdb"))
        assert core.get_fleet_activity() == []

    def test_merges_across_all_three_tables(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        _seed_db(
            db,
            processing_rows=[("content-discovery-agent", True), ("content-discovery-agent", False)],
            fetch_rows=[("content-discovery-agent", True)],
            api_call_rows=[("content-discovery-agent", True)],
        )
        activity = core.get_fleet_activity()
        assert len(activity) == 1
        a = activity[0]
        assert a.tool_name == "content-discovery-agent"
        assert a.total == 4
        assert a.failures == 1
        assert set(a.tables) == {"processing_log", "fetch_log", "api_call_log"}

    def test_sorted_most_recent_first(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute("CREATE TABLE processing_log (tool_name VARCHAR, success BOOLEAN, via_gateway BOOLEAN, created_at TIMESTAMP)")
        conn.execute("CREATE TABLE tools (id INTEGER, name VARCHAR)")
        conn.execute("CREATE TABLE fetch_log (tool_id INTEGER, success BOOLEAN, attempted_at TIMESTAMP)")
        conn.execute("CREATE TABLE api_call_log (tool_id INTEGER, success BOOLEAN, attempted_at TIMESTAMP)")
        conn.execute(
            "INSERT INTO processing_log (tool_name, success, created_at) "
            "VALUES ('older-tool', true, CURRENT_TIMESTAMP - INTERVAL 1 HOUR)"
        )
        conn.execute(
            "INSERT INTO processing_log (tool_name, success, created_at) "
            "VALUES ('newer-tool', true, CURRENT_TIMESTAMP)"
        )
        conn.close()
        activity = core.get_fleet_activity()
        assert [a.tool_name for a in activity] == ["newer-tool", "older-tool"]

    def test_lock_conflict_returns_empty_not_raises(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        _seed_db(db, processing_rows=[("my-tool", True)])
        with patch("duckdb.connect", side_effect=RuntimeError("could not set lock on file")):
            assert core.get_fleet_activity() == []

    def test_gateway_echo_row_excluded_from_total(self, tmp_path, monkeypatch):
        """Regression 2026-09-20: llm-gateway-service's own row for a
        gateway-routed call (via_gateway=True) is a second record of the
        same call the tool's own row already counted -- summing both
        doubled every gateway-routed tool's call total."""
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, success BOOLEAN, "
            "via_gateway BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("CREATE TABLE tools (id INTEGER, name VARCHAR)")
        conn.execute("CREATE TABLE fetch_log (tool_id INTEGER, success BOOLEAN, attempted_at TIMESTAMP)")
        conn.execute("CREATE TABLE api_call_log (tool_id INTEGER, success BOOLEAN, attempted_at TIMESTAMP)")
        conn.execute(
            "INSERT INTO processing_log (tool_name, success, via_gateway) VALUES "
            "('my-tool', true, NULL), ('my-tool', true, true)"
        )
        conn.close()

        activity = core.get_fleet_activity()
        assert len(activity) == 1
        assert activity[0].total == 1


class TestGetLaunchAgents:
    def test_filters_to_localfirst_and_jamalhansen_prefixes(self):
        fake = MagicMock()
        fake.stdout = (
            "-\t0\tcom.localfirst.artist-agent\n"
            "1234\t0\tcom.apple.something\n"
            "5678\t0\tcom.jamalhansen.discovery-loop\n"
        )
        with patch("fleet_dashboard.core.subprocess.run", return_value=fake):
            statuses = core.get_launch_agents()
        labels = {s.label for s in statuses}
        assert labels == {"com.localfirst.artist-agent", "com.jamalhansen.discovery-loop"}

    def test_running_state_and_pid_parsed(self):
        fake = MagicMock()
        fake.stdout = "4242\t0\tcom.localfirst.llm-gateway-service\n"
        with patch("fleet_dashboard.core.subprocess.run", return_value=fake):
            statuses = core.get_launch_agents()
        assert statuses[0].running is True
        assert statuses[0].pid == 4242

    def test_launchctl_failure_returns_empty_not_raises(self):
        with patch("fleet_dashboard.core.subprocess.run", side_effect=RuntimeError("boom")):
            assert core.get_launch_agents() == []

    def test_keep_alive_read_from_real_plist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core, "_launch_agents_dir", lambda: tmp_path)
        path = tmp_path / "com.localfirst.llm-gateway-service.plist"
        with path.open("wb") as f:
            plistlib.dump({"KeepAlive": True}, f)
        fake = MagicMock()
        fake.stdout = "99\t0\tcom.localfirst.llm-gateway-service\n"
        with patch("fleet_dashboard.core.subprocess.run", return_value=fake):
            statuses = core.get_launch_agents()
        assert statuses[0].keep_alive is True


class TestGetModelUsage:
    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(tmp_path / "nope.duckdb"))
        assert core.get_model_usage() == []

    def test_groups_by_tool_and_model(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        # A FallbackProvider failure (primary's real model, success=False) and the
        # fallback's own successful row land as two distinct (tool, model) pairs --
        # exactly the "real cost/reliability per model" comparison this exists for.
        conn.execute(
            "INSERT INTO processing_log (tool_name, model, success) VALUES "
            "('japanese-tutor', 'phi4-mini', false), "
            "('japanese-tutor', 'phi4-mini', false), "
            "('japanese-tutor', 'phi4-mini', true), "
            "('japanese-tutor', 'deepseek-chat', true)"
        )
        conn.close()

        usage = core.get_model_usage()
        by_model = {u.model: u for u in usage}
        assert by_model["phi4-mini"].total == 3
        assert by_model["phi4-mini"].failures == 2
        assert by_model["phi4-mini"].failure_rate == 2 / 3
        assert by_model["phi4-mini"].provider == "ollama"
        assert by_model["deepseek-chat"].total == 1
        assert by_model["deepseek-chat"].failures == 0
        assert by_model["deepseek-chat"].provider == "deepseek"

    def test_null_model_reported_as_unset(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO processing_log (tool_name, model, success) VALUES ('some-tool', NULL, true)")
        conn.close()

        usage = core.get_model_usage()
        assert usage[0].model == "(unset)"
        assert usage[0].provider == "(unset)"

    def test_provider_prefixed_and_bare_model_collapse_into_one_row(self, tmp_path, monkeypatch):
        """Real 2026-09-20 production data has both "phi4-mini" and
        "ollama:phi4-mini" logged for the same tool -- same logical model,
        should not double-count as two rows."""
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO processing_log (tool_name, model, success) VALUES "
            "('my-tool', 'phi4-mini', true), "
            "('my-tool', 'ollama:phi4-mini', true)"
        )
        conn.close()

        usage = core.get_model_usage()
        assert len(usage) == 1
        assert usage[0].provider == "ollama"
        assert usage[0].model == "phi4-mini"
        assert usage[0].total == 2

    def test_lock_conflict_returns_empty_not_raises(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.close()
        with patch("duckdb.connect", side_effect=RuntimeError("could not set lock on file")):
            assert core.get_model_usage() == []

    def test_gateway_routed_call_is_not_double_counted(self, tmp_path, monkeypatch):
        """Regression 2026-09-20: a gateway-routed call produces two rows --
        the calling tool's own timed_run() row (provider usually NULL) and
        llm-gateway-service's own row for the same request (via_gateway=True,
        provider populated for real) -- and both used to land in the same
        guessed bucket, doubling the count. Real numbers: obsidian-vault-
        auto-tagger's true ~81 qwen2.5:7b calls showed as ~152."""
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO processing_log (tool_name, model, provider, via_gateway, success) VALUES "
            "('obsidian-vault-auto-tagger', 'qwen2.5:7b', NULL, NULL, true), "
            "('obsidian-vault-auto-tagger', 'qwen2.5:7b', 'ollama', true, true)"
        )
        conn.close()

        usage = core.get_model_usage()
        assert len(usage) == 1
        assert usage[0].total == 1
        assert usage[0].provider == "ollama"

    def test_non_gateway_null_provider_row_still_counted(self, tmp_path, monkeypatch):
        """A tool that never routes through the gateway (or a legacy row from
        before the provider column existed) has no via_gateway row to defer
        to -- its NULL-provider rows must still be counted via the
        classify_provider() heuristic, not silently dropped."""
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO processing_log (tool_name, model, success) VALUES ('old-tool', 'phi4-mini', true)"
        )
        conn.close()

        usage = core.get_model_usage()
        assert len(usage) == 1
        assert usage[0].total == 1

    def test_real_provider_column_takes_priority_over_heuristic(self, tmp_path, monkeypatch):
        """local_first_common now writes a real provider column for
        gateway-routed calls (2026-09-20) -- when it's populated, use it
        directly instead of guessing from the model string. A bare
        "phi4-mini" with no real provider recorded would otherwise guess
        "ollama" correctly by luck; this proves it's reading the real column,
        not just getting lucky, by using a model name the heuristic would
        get wrong on its own."""
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO processing_log (tool_name, model, provider, success) VALUES "
            "('my-tool', 'a-custom-finetune', 'groq', true)"
        )
        conn.close()

        usage = core.get_model_usage()
        assert usage[0].provider == "groq"
        assert usage[0].model == "a-custom-finetune"
        # the heuristic alone would have guessed "ollama" for this model string
        assert core.classify_provider("a-custom-finetune")[0] == "ollama"


class TestClassifyProvider:
    def test_unset_model(self):
        assert core.classify_provider(None) == ("(unset)", "(unset)")
        assert core.classify_provider("") == ("(unset)", "(unset)")

    def test_known_prefix_splits_provider_and_model(self):
        assert core.classify_provider("ollama:phi4-mini") == ("ollama", "phi4-mini")
        assert core.classify_provider("anthropic:claude-haiku-4-5-20251001") == ("anthropic", "claude-haiku-4-5-20251001")
        assert core.classify_provider("mock:test-model") == ("mock", "test-model")

    def test_unrecognized_prefix_is_not_split(self):
        """"llama3.2:3b" has a colon but "llama3.2" isn't a known provider
        name -- it's an Ollama tag, must not be misread as provider "llama3.2"."""
        assert core.classify_provider("llama3.2:3b") == ("ollama", "llama3.2:3b")

    def test_claude_and_gemini_and_deepseek_prefixes(self):
        assert core.classify_provider("claude-sonnet-5")[0] == "anthropic"
        assert core.classify_provider("gemini-2.0-flash")[0] == "gemini"
        assert core.classify_provider("deepseek-chat")[0] == "deepseek"

    def test_known_groq_default_model(self):
        assert core.classify_provider("llama-3.3-70b-versatile")[0] == "groq"

    def test_mock_repr_leaked_from_a_test(self):
        assert core.classify_provider("<MagicMock name='mock.model' id='123'>")[0] == "mock"

    def test_unrecognized_model_defaults_to_ollama(self):
        assert core.classify_provider("qwen2.5:7b")[0] == "ollama"
        assert core.classify_provider("nomic-embed-text")[0] == "ollama"


class TestGetProviderUsage:
    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(tmp_path / "nope.duckdb"))
        assert core.get_provider_usage() == []

    def test_rolls_up_across_tools_and_models(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, provider VARCHAR, "
            "via_gateway BOOLEAN, success BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO processing_log (tool_name, model, success) VALUES "
            "('japanese-tutor', 'phi4-mini', false), "
            "('japanese-tutor', 'phi4-mini', true), "
            "('photo-renamer', 'qwen2.5:7b', true), "
            "('japanese-tutor', 'deepseek-chat', true)"
        )
        conn.close()

        usage = core.get_provider_usage()
        by_provider = {p.provider: p for p in usage}
        assert by_provider["ollama"].total == 3
        assert by_provider["ollama"].failures == 1
        assert by_provider["ollama"].tool_count == 2
        assert by_provider["ollama"].model_count == 2
        assert by_provider["deepseek"].total == 1
        assert by_provider["deepseek"].tool_count == 1


class TestGetTensionSummary:
    def test_missing_vault_returns_zeroes(self, tmp_path):
        result = core.get_tension_summary(tmp_path / "does-not-exist")
        assert result.pending_count == 0
        assert result.recent_titles == []

    def test_counts_pending_and_active_separately(self, tmp_path):
        tensions_dir = tmp_path / "ops" / "tensions"
        tensions_dir.mkdir(parents=True)
        (tensions_dir / "t1.md").write_text(
            "---\ntitle: First tension\nstatus: pending\ncreated: 2026-09-01\n---\nbody"
        )
        (tensions_dir / "t2.md").write_text(
            "---\ntitle: Second tension\nstatus: active\ncreated: 2026-09-02\n---\nbody"
        )
        result = core.get_tension_summary(tmp_path)
        assert result.pending_count == 1
        assert result.active_count == 1
        assert "Second tension" in result.recent_titles


class TestGetLatestArt:
    def test_missing_dir_returns_none(self, tmp_path):
        assert core.get_latest_art(tmp_path / "nope") is None

    def test_no_items_returns_none(self, tmp_path):
        (tmp_path / "items").mkdir()
        assert core.get_latest_art(tmp_path / "items") is None

    def test_picks_most_recent_by_filename_and_finds_image(self, tmp_path):
        items_dir = tmp_path / "items"
        images_dir = tmp_path / "images"
        items_dir.mkdir()
        images_dir.mkdir()
        (items_dir / "2026-09-19-old.md").write_text(
            "---\ntitle: Old piece\nself_score: 0.5\n---\nbody"
        )
        (items_dir / "2026-09-20-new.md").write_text(
            "---\ntitle: New piece\nself_score: 0.8\ninterest: Testing\n---\nbody"
        )
        (images_dir / "2026-09-20-new.png").write_bytes(b"fake-png-bytes")

        item = core.get_latest_art(items_dir)
        assert item is not None
        assert item.title == "New piece"
        assert item.self_score == 0.8
        assert item.image_path == images_dir / "2026-09-20-new.png"

    def test_no_matching_image_returns_none(self, tmp_path):
        items_dir = tmp_path / "items"
        (tmp_path / "images").mkdir()
        items_dir.mkdir()
        (items_dir / "2026-09-20-new.md").write_text("---\ntitle: New piece\n---\nbody")
        assert core.get_latest_art(items_dir) is None


class TestGetJapaneseTutorSummary:
    def test_unreachable_server_returns_not_reachable(self):
        result = core.get_japanese_tutor_summary("http://127.0.0.1:1")
        assert result.reachable is False
        assert result.cards_due == 0

    def test_reachable_server_parses_response(self):
        """Regression 2026-09-20: this used to read len(/api/cards/due),
        which always returns a fixed-size, backfilled list regardless of
        real review progress. Must read /api/cards/due/count instead."""
        due_resp = MagicMock()
        due_resp.json.return_value = {"due_count": 3, "new_count": 2, "review_count": 1}
        mastery_resp = MagicMock()
        mastery_resp.json.return_value = [{"stage": "hiragana", "mastered": 5}]
        reviews_resp = MagicMock()
        reviews_resp.json.return_value = {"attempts": 7, "distinct_cards": 5}

        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False
        fake_client.get.side_effect = [due_resp, mastery_resp, reviews_resp]

        with patch("fleet_dashboard.core.httpx.Client", return_value=fake_client):
            result = core.get_japanese_tutor_summary()

        assert result.reachable is True
        assert result.cards_due == 3
        assert result.new_count == 2
        assert result.review_count == 1
        assert result.mastery == [{"stage": "hiragana", "mastered": 5}]
        assert result.reviews_today_attempts == 7
        assert result.reviews_today_distinct_cards == 5
        fake_client.get.assert_any_call("http://127.0.0.1:8421/api/cards/due/count")
        fake_client.get.assert_any_call("http://127.0.0.1:8421/api/reviews/today")


class TestGetVaultHealth:
    def test_missing_vault_returns_zeroes(self, tmp_path):
        result = core.get_vault_health(tmp_path / "does-not-exist")
        assert result.observations_pending == 0
        assert result.inbox_count == 0
        assert result.inbox_oldest_days is None
        assert result.last_health_check is None

    def test_counts_only_pending_observations(self, tmp_path):
        obs_dir = tmp_path / "ops" / "observations"
        obs_dir.mkdir(parents=True)
        (obs_dir / "o1.md").write_text("---\ntype: observation\nstatus: pending\n---\nbody")
        (obs_dir / "o2.md").write_text("---\ntype: observation\nstatus: pending\n---\nbody")
        (obs_dir / "o3.md").write_text("---\ntype: observation\nstatus: promoted\n---\nbody")
        result = core.get_vault_health(tmp_path)
        assert result.observations_pending == 2

    def test_inbox_count_and_oldest_age(self, tmp_path):
        import os
        import time

        inbox_dir = tmp_path / "inbox"
        inbox_dir.mkdir()
        old_file = inbox_dir / "old.md"
        old_file.write_text("old")
        two_days_ago = time.time() - (2 * 86400)
        os.utime(old_file, (two_days_ago, two_days_ago))
        (inbox_dir / "new.md").write_text("new")

        result = core.get_vault_health(tmp_path)
        assert result.inbox_count == 2
        assert result.inbox_oldest_days >= 1.9

    def test_last_health_check_is_most_recent_report(self, tmp_path):
        health_dir = tmp_path / "ops" / "health"
        health_dir.mkdir(parents=True)
        (health_dir / "2026-08-01-report.md").write_text("old report")
        (health_dir / "2026-09-06-report.md").write_text("newest report")
        result = core.get_vault_health(tmp_path)
        assert result.last_health_check == "2026-09-06"

    def test_malformed_observation_note_is_skipped_not_fatal(self, tmp_path):
        obs_dir = tmp_path / "ops" / "observations"
        obs_dir.mkdir(parents=True)
        (obs_dir / "broken.md").write_text("not valid frontmatter at all {{{")
        (obs_dir / "good.md").write_text("---\nstatus: pending\n---\nbody")
        result = core.get_vault_health(tmp_path)
        assert result.observations_pending == 1


class TestGetRepoHealth:
    def test_missing_snapshot_returns_unavailable(self, tmp_path):
        result = core.get_repo_health(tmp_path / "nope.json")
        assert result.available is False

    def test_reads_real_snapshot(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps({
            "generated_at": "2026-09-20T12:00:00Z",
            "total": 2,
            "healthy": 1,
            "repos": {
                "good-repo": {
                    "lint": {"ok": True, "error_count": 0},
                    "tests": {"ok": True, "passed": 10, "failed": 0},
                    "git": {"dirty": False, "unpushed": 0, "has_remote": True},
                    "hooks": {"ok": True, "installed": True},
                },
                "bad-repo": {
                    "lint": {"ok": False, "error_count": 3},
                    "tests": {"ok": False, "passed": 0, "failed": 1},
                    "git": {"dirty": True, "unpushed": 2, "has_remote": True},
                    "hooks": {"ok": True, "installed": True},
                },
            },
        }))
        result = core.get_repo_health(snapshot)
        assert result.available is True
        assert result.total == 2
        assert result.healthy == 1
        assert len(result.repos) == 2

    def test_unhealthy_repos_sorted_first(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps({
            "generated_at": "x",
            "total": 2,
            "healthy": 1,
            "repos": {
                "a-good-repo": {
                    "lint": {"ok": True, "error_count": 0},
                    "tests": {"ok": True, "passed": 1, "failed": 0},
                    "git": {"dirty": False, "unpushed": 0, "has_remote": True},
                    "hooks": {"ok": True, "installed": True},
                },
                "z-bad-repo": {
                    "lint": {"ok": False, "error_count": 1},
                    "tests": {"ok": True, "passed": 1, "failed": 0},
                    "git": {"dirty": False, "unpushed": 0, "has_remote": True},
                    "hooks": {"ok": True, "installed": True},
                },
            },
        }))
        result = core.get_repo_health(snapshot)
        assert result.repos[0]["name"] == "z-bad-repo"
        assert result.repos[0]["ok"] is False

    def test_malformed_snapshot_returns_unavailable_not_raises(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text("not valid json {{{")
        result = core.get_repo_health(snapshot)
        assert result.available is False

    def test_no_remote_counts_as_unhealthy(self, tmp_path):
        """Jamal: no remote should be called out on the dashboard -- a repo
        with no remote is unbacked, at risk of being lost, regardless of
        how clean its lint/tests/hooks are."""
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps({
            "generated_at": "x",
            "total": 1,
            "healthy": 0,
            "repos": {
                "local-only-repo": {
                    "lint": {"ok": True, "error_count": 0},
                    "tests": {"ok": True, "passed": 5, "failed": 0},
                    "git": {"dirty": False, "unpushed": 0, "has_remote": False},
                    "hooks": {"ok": True, "installed": True},
                },
            },
        }))
        result = core.get_repo_health(snapshot)
        assert result.repos[0]["ok"] is False
        assert result.repos[0]["has_remote"] is False


class TestGetFrontmatterValidation:
    def test_missing_snapshot_returns_unavailable(self, tmp_path):
        result = core.get_frontmatter_validation(tmp_path / "nope.json")
        assert result.available is False

    def test_reads_real_snapshot(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(
            '{"generated_at": "2026-09-20T12:00:00Z", "total": 10, "invalid_count": 2, '
            '"invalid_files": [{"file": "a.md", "errors": ["x"]}, {"file": "b.md", "errors": ["y"]}]}'
        )
        result = core.get_frontmatter_validation(snapshot)
        assert result.available is True
        assert result.total == 10
        assert result.invalid_count == 2
        assert len(result.invalid_files) == 2

    def test_invalid_files_truncated_to_ten(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [{"file": f"{i}.md", "errors": ["x"]} for i in range(25)]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 25, "invalid_count": 25, "invalid_files": files}))
        result = core.get_frontmatter_validation(snapshot)
        assert len(result.invalid_files) == 10

    def test_malformed_snapshot_returns_unavailable_not_raises(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text("not valid json {{{")
        result = core.get_frontmatter_validation(snapshot)
        assert result.available is False

    def test_error_summary_groups_files_by_error_text(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [
            {"file": "a.md", "errors": ["Missing universal field: 'tags'"]},
            {"file": "b.md", "errors": ["Missing universal field: 'tags'"]},
            {"file": "c.md", "errors": ["Missing universal field: 'canonical_url'"]},
        ]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 3, "invalid_count": 3, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        by_error = {e["error"]: e for e in result.error_summary}
        assert by_error["Missing universal field: 'tags'"]["count"] == 2
        assert set(by_error["Missing universal field: 'tags'"]["files"]) == {"a.md", "b.md"}
        assert by_error["Missing universal field: 'canonical_url'"]["count"] == 1

    def test_error_summary_sorted_by_count_descending(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [
            {"file": "a.md", "errors": ["rare error"]},
            {"file": "b.md", "errors": ["common error"]},
            {"file": "c.md", "errors": ["common error"]},
            {"file": "d.md", "errors": ["common error"]},
        ]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 4, "invalid_count": 4, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        assert result.error_summary[0]["error"] == "common error"
        assert result.error_summary[0]["count"] == 3
        assert result.error_summary[1]["error"] == "rare error"

    def test_error_summary_covers_all_invalid_files_not_just_the_truncated_ten(self, tmp_path):
        """error_summary must reflect all 25 invalid files, even though
        invalid_files itself is truncated to 10 for the per-file view."""
        snapshot = tmp_path / "snapshot.json"
        files = [{"file": f"{i}.md", "errors": ["shared error"]} for i in range(25)]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 25, "invalid_count": 25, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        assert len(result.invalid_files) == 10
        assert result.error_summary[0]["count"] == 25

    def test_error_summary_files_capped_with_a_more_count(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [{"file": f"{i}.md", "errors": ["shared error"]} for i in range(25)]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 25, "invalid_count": 25, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        entry = result.error_summary[0]
        assert len(entry["files"]) == 20
        assert entry["more"] == 5

    def test_error_summary_a_file_with_multiple_errors_counts_under_each(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [{"file": "a.md", "errors": ["error one", "error two"]}]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 1, "invalid_count": 1, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        errors = {e["error"] for e in result.error_summary}
        assert errors == {"error one", "error two"}

    def test_content_todo_buckets_files_by_action_needed(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [
            {"file": "a.md", "errors": ["Missing universal field: 'tags'"]},
            {"file": "b.md", "errors": ["Missing universal field: 'status'"]},
            {"file": "c.md", "errors": ["Missing 'category' field"]},
            {"file": "d.md", "errors": ["Field 'canonical_url' is required when 'status' is 'published'"]},
            {"file": "e.md", "errors": ["Failed to parse frontmatter: bad yaml"]},
        ]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 5, "invalid_count": 5, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        by_key = {b["key"]: b for b in result.content_todo}
        assert by_key["auto_fixable"]["files"] == ["a.md"]
        assert by_key["needs_status"]["files"] == ["b.md"]
        assert by_key["needs_category"]["files"] == ["c.md"]
        assert by_key["published_incomplete"]["files"] == ["d.md"]
        assert by_key["parse_error"]["files"] == ["e.md"]

    def test_content_todo_most_urgent_bucket_listed_first(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [
            {"file": "a.md", "errors": ["Missing universal field: 'tags'"]},
            {"file": "b.md", "errors": ["Field 'canonical_url' is required when 'status' is 'published'"]},
        ]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 2, "invalid_count": 2, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        assert result.content_todo[0]["key"] == "published_incomplete"

    def test_content_todo_a_file_can_land_in_more_than_one_bucket(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [
            {"file": "a.md", "errors": ["Missing universal field: 'tags'", "Missing universal field: 'status'"]},
        ]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 1, "invalid_count": 1, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        by_key = {b["key"]: b for b in result.content_todo}
        assert by_key["auto_fixable"]["files"] == ["a.md"]
        assert by_key["needs_status"]["files"] == ["a.md"]

    def test_content_todo_empty_buckets_are_omitted(self, tmp_path):
        snapshot = tmp_path / "snapshot.json"
        files = [{"file": "a.md", "errors": ["Missing universal field: 'tags'"]}]
        snapshot.write_text(json.dumps({"generated_at": "x", "total": 1, "invalid_count": 1, "invalid_files": files}))

        result = core.get_frontmatter_validation(snapshot)

        keys = {b["key"] for b in result.content_todo}
        assert keys == {"auto_fixable"}
