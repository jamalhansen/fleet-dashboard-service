import plistlib
from unittest.mock import MagicMock, patch

import duckdb

from fleet_dashboard import core


def _seed_db(db_path, processing_rows=(), fetch_rows=(), api_call_rows=()):
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, success BOOLEAN, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
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
        conn.execute("CREATE TABLE processing_log (tool_name VARCHAR, success BOOLEAN, created_at TIMESTAMP)")
        conn.execute("CREATE TABLE tools (id INTEGER, name VARCHAR)")
        conn.execute("CREATE TABLE fetch_log (tool_id INTEGER, success BOOLEAN, attempted_at TIMESTAMP)")
        conn.execute("CREATE TABLE api_call_log (tool_id INTEGER, success BOOLEAN, attempted_at TIMESTAMP)")
        conn.execute(
            "INSERT INTO processing_log VALUES ('older-tool', true, CURRENT_TIMESTAMP - INTERVAL 1 HOUR)"
        )
        conn.execute("INSERT INTO processing_log VALUES ('newer-tool', true, CURRENT_TIMESTAMP)")
        conn.close()
        activity = core.get_fleet_activity()
        assert [a.tool_name for a in activity] == ["newer-tool", "older-tool"]

    def test_lock_conflict_returns_empty_not_raises(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        _seed_db(db, processing_rows=[("my-tool", True)])
        with patch("duckdb.connect", side_effect=RuntimeError("could not set lock on file")):
            assert core.get_fleet_activity() == []


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
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, success BOOLEAN, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
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
        assert by_model["deepseek-chat"].total == 1
        assert by_model["deepseek-chat"].failures == 0

    def test_null_model_reported_as_unset(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, success BOOLEAN, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT INTO processing_log (tool_name, model, success) VALUES ('some-tool', NULL, true)")
        conn.close()

        usage = core.get_model_usage()
        assert usage[0].model == "(unset)"

    def test_lock_conflict_returns_empty_not_raises(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(db))
        conn = duckdb.connect(str(db))
        conn.execute(
            "CREATE TABLE processing_log (tool_name VARCHAR, model VARCHAR, success BOOLEAN, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.close()
        with patch("duckdb.connect", side_effect=RuntimeError("could not set lock on file")):
            assert core.get_model_usage() == []


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
        due_resp = MagicMock()
        due_resp.json.return_value = [{"card_id": 1}, {"card_id": 2}]
        mastery_resp = MagicMock()
        mastery_resp.json.return_value = [{"stage": "hiragana", "mastered": 5}]

        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False
        fake_client.get.side_effect = [due_resp, mastery_resp]

        with patch("fleet_dashboard.core.httpx.Client", return_value=fake_client):
            result = core.get_japanese_tutor_summary()

        assert result.reachable is True
        assert result.cards_due == 2
        assert result.mastery == [{"stage": "hiragana", "mastered": 5}]
