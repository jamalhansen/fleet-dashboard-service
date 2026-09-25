import json
import re
from datetime import UTC, datetime
from pathlib import Path

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
    out = tmp_path / "Dashboards" / "fleet.html"
    snapshot.write_snapshot(out, now=datetime(2026, 9, 25, 8, 0, tzinfo=UTC))
    first = out.read_text()
    snapshot.write_snapshot(out, now=datetime(2026, 9, 25, 8, 30, tzinfo=UTC))
    assert out.read_text() != first
    assert "8:30 AM" in out.read_text()
    assert list(out.parent.iterdir()) == [out]
