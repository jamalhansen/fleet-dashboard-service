import pytest


@pytest.fixture(autouse=True)
def _isolate_tracking_db(tmp_path_factory, monkeypatch):
    tracking_db = tmp_path_factory.mktemp("tracking") / "test_tracking.duckdb"
    monkeypatch.setenv("LOCAL_FIRST_TRACKING_DB", str(tracking_db))
