import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fleet_dashboard import snapshot

INDEX_HTML = snapshot.INDEX.read_text(encoding="utf-8")


def test_lookback_options_come_from_the_page():
    assert snapshot.lookback_options(INDEX_HTML) == ["24", "168", "720", "87600"]


def test_every_api_path_the_page_fetches_is_baked():
    fetched = set(re.findall(r"fetch\([`'](/api/[a-z-]+)", INDEX_HTML))
    baked = set(snapshot._routes())
    assert fetched - {snapshot.ART_IMAGE_PATH} <= baked


def test_collect_keys_match_the_urls_the_page_builds(monkeypatch):
    def fleet(lookback_hours: float = 168):
        return {"h": lookback_hours}

    def tensions(vault: str = "Contexta"):
        return {"v": vault}

    def writing():
        return {"ok": True}

    monkeypatch.setattr(snapshot, "_routes", lambda: {
        "/api/fleet": fleet, "/api/tensions": tensions, "/api/writing": writing,
    })
    data = snapshot.collect(["24", "168"], ["Contexta", "KeySix"])
    assert data == {
        "/api/fleet?lookback_hours=24": {"h": 24.0},
        "/api/fleet?lookback_hours=168": {"h": 168.0},
        "/api/tensions?vault=Contexta": {"v": "Contexta"},
        "/api/tensions?vault=KeySix": {"v": "KeySix"},
        "/api/writing": {"ok": True},
    }


def test_render_injects_data_note_and_hides_live_controls():
    page = snapshot.render(INDEX_HTML, {"/api/writing": {"title": "</script>"}}, datetime(2026, 9, 25, 7, 14, tzinfo=UTC))
    assert "Snapshot from Fri Sep 25, 7:14 AM" in page
    assert "#updated-at, #refresh-btn { display: none; }" in page
    shim = page.index("window.__SNAPSHOT__")
    assert shim < page.index("function lookbackHours")
    assert "<\\/script>" in page  # data can't close the script tag early
    payload = page[shim:].split("= ", 1)[1].split(";\nwindow.fetch", 1)[0]
    assert json.loads(payload.replace("<\\/", "</")) == {"/api/writing": {"title": "</script>"}}


def test_write_snapshot_overwrites_atomically(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(snapshot, "collect", lambda lookbacks, vaults: {"/api/art": {"available": False}})
    monkeypatch.setattr(snapshot, "prerender", lambda page: page)
    out = tmp_path / "Dashboards" / "fleet.html"
    snapshot.write_snapshot(out, now=datetime(2026, 9, 25, 8, 0, tzinfo=UTC))
    first = out.read_text()
    snapshot.write_snapshot(out, now=datetime(2026, 9, 25, 8, 30, tzinfo=UTC))
    assert out.read_text() != first
    assert "8:30 AM" in out.read_text()
    assert list(out.parent.iterdir()) == [out]


def test_strip_scripts_removes_js_and_hides_the_picker():
    html = '<html><head></head><body><script>x()</script><SCRIPT src="a.js"></SCRIPT><p>ok</p></body></html>'
    out = snapshot.strip_scripts(html)
    assert "script" not in out.lower().replace("#lookback-select", "")
    assert "<p>ok</p>" in out
    assert "#lookback-select { display: none; }" in out


def test_prerender_fills_every_panel_without_javascript(monkeypatch):
    pytest.importorskip("playwright")
    monkeypatch.setattr(snapshot, "_routes", dict)
    data = {path: {} for path in set(re.findall(r"fetch\([`'](/api/[a-z-]+)", INDEX_HTML))}
    page = snapshot.render(INDEX_HTML, data, datetime(2026, 9, 25, 7, 0, tzinfo=UTC))
    try:
        out = snapshot.prerender(page, timeout_ms=10_000)
    except Exception as e:
        if "Executable doesn't exist" in str(e):
            pytest.skip("playwright chromium not installed")
        raise
    assert 'class="skeleton"' not in out
    assert "<script" not in out.lower()


def test_publish_copies_with_scp_and_reports_success(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(snapshot.subprocess, "run", fake_run)
    page = tmp_path / "fleet.html"
    page.write_text("x")
    assert snapshot.publish(page, "clifford:dashboard/index.html") is True
    assert calls[0][0] == "scp" and calls[0][-2:] == [str(page), "clifford:dashboard/index.html"]
    assert "BatchMode=yes" in calls[0]


def test_publish_failure_is_reported_not_raised(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        snapshot.subprocess, "run",
        lambda cmd, **kw: type("R", (), {"returncode": 255, "stderr": "no route"})(),
    )
    assert snapshot.publish(tmp_path / "f.html", "clifford:x") is False
    assert "no route" in capsys.readouterr().out

    def boom(cmd, **kw):
        raise snapshot.subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(snapshot.subprocess, "run", boom)
    assert snapshot.publish(tmp_path / "f.html", "clifford:x") is False


def test_main_publishes_only_when_target_is_set(monkeypatch, tmp_path):
    out = tmp_path / "fleet.html"
    monkeypatch.setenv("FLEET_DASHBOARD_SNAPSHOT_PATH", str(out))
    monkeypatch.setattr(snapshot, "write_snapshot", lambda p: (p.write_text("x"), p)[1])
    pushed = []
    monkeypatch.setattr(snapshot, "publish", lambda p, t: pushed.append(t) or True)

    monkeypatch.delenv("FLEET_DASHBOARD_PUBLISH_TO", raising=False)
    snapshot.main()
    assert pushed == []

    monkeypatch.setenv("FLEET_DASHBOARD_PUBLISH_TO", "clifford:dashboard/index.html")
    snapshot.main()
    assert pushed == ["clifford:dashboard/index.html"]
