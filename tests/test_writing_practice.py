import json
from datetime import datetime

from fleet_dashboard import core


def test_missing_file_is_unavailable(tmp_path):
    assert core.get_writing_practice(tmp_path / "nope.json") == {"available": False}


def test_reads_status_and_flags_stale(tmp_path):
    path = tmp_path / "w.json"
    today = datetime.now().astimezone().date().isoformat()
    path.write_text(json.dumps({"generated": f"{today}T20:00-05:00", "today": {"done": False}}))
    got = core.get_writing_practice(path)
    assert got["available"] and not got["stale"] and got["today"] == {"done": False}
    path.write_text(json.dumps({"generated": "2000-01-01T03:30-05:00"}))
    assert core.get_writing_practice(path)["stale"]


def test_corrupt_file_degrades(tmp_path):
    path = tmp_path / "w.json"
    path.write_text("{not json")
    assert core.get_writing_practice(path) == {"available": False}
