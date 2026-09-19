from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import threading
import urllib.error

import pytest
import requests

from paper_atlas.ai_interpreter import AIInterpreter
from paper_atlas.conference_web import create_server
from paper_atlas.weekly_limit import WeeklyLimit, WeeklyLimitReached
from test_ai_interpreter import response
from test_conference_web import engine, request


def seed(quota, count):
    with quota.connect() as db:
        db.execute("INSERT OR REPLACE INTO ai_weekly_usage VALUES (?,?)", (quota.status()["week_start"], count))


def test_limit_1000_and_persistence(tmp_path):
    quota = WeeklyLimit(tmp_path / "usage.sqlite3")
    assert quota.status()["remaining"] == 1000
    for i in range(1000):
        assert quota.reserve()["used"] == i+1
    quota = WeeklyLimit(quota.path)
    with pytest.raises(WeeklyLimitReached) as error:
        quota.reserve()
    assert error.value.quota["remaining"] == 0
    assert quota.status()["used"] == 1000


def test_concurrent_reservations_cannot_exceed_limit(tmp_path):
    quota = WeeklyLimit(tmp_path / "usage.sqlite3")
    seed(quota, 995)
    def reserve(_):
        try:
            WeeklyLimit(quota.path).reserve()
            return True
        except WeeklyLimitReached:
            return False
    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(reserve, range(30))) == 5
    assert quota.status()["used"] == 1000


@pytest.mark.parametrize("before,after", [
    ("2026-09-20T23:59:59-05:00", "2026-09-21T00:00:00-05:00"),
    ("2026-03-08T23:59:59-05:00", "2026-03-09T00:00:00-05:00"),
    ("2026-11-01T23:59:59-06:00", "2026-11-02T00:00:00-06:00"),
    ("2027-01-03T23:59:59-06:00", "2027-01-04T00:00:00-06:00"),
])
def test_chicago_monday_reset_and_dst(tmp_path, before, after):
    now = [datetime.fromisoformat(before)]
    quota = WeeklyLimit(tmp_path / "usage.sqlite3", clock=lambda: now[0])
    seed(quota, 1000)
    assert quota.status()["resets_at"] == after
    with pytest.raises(WeeklyLimitReached):
        quota.reserve()
    now[0] = datetime.fromisoformat(after)
    assert quota.status()["used"] == 0
    assert quota.reserve()["remaining"] == 999


def test_ai_cache_does_not_consume_quota_and_works_at_limit(tmp_path):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return response()
    interpreter = AIInterpreter("fake", tmp_path / "cache", post)
    first = interpreter.interpret("graphs")
    assert first["weekly_quota"]["used"] == 1
    seed(interpreter.quota, 1000)
    assert interpreter.interpret("graphs")["cached"]
    with pytest.raises(WeeklyLimitReached):
        interpreter.interpret("new graphs")
    assert len(calls) == 1
    with pytest.raises(ValueError):
        interpreter.interpret("")
    assert interpreter.quota.status()["used"] == 1000


def test_failures_count_and_retry_is_blocked_at_cap(tmp_path):
    calls = []
    def incomplete(*args, **kwargs):
        calls.append(kwargs)
        return response(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    interpreter = AIInterpreter("fake", tmp_path / "cache", incomplete)
    seed(interpreter.quota, 999)
    with pytest.raises(WeeklyLimitReached):
        interpreter.interpret("graphs")
    assert len(calls) == 1
    assert interpreter.quota.status()["used"] == 1000


def test_transport_failure_counts(tmp_path):
    def post(*args, **kwargs):
        raise requests.Timeout()
    interpreter = AIInterpreter("fake", tmp_path / "cache", post)
    with pytest.raises(ValueError, match="could not be reached"):
        interpreter.interpret("graphs")
    assert interpreter.quota.status()["used"] == 1


def test_different_state_directories_share_counter(tmp_path):
    quota_path = tmp_path / "shared_usage"
    first = AIInterpreter("fake", tmp_path / "state1/cache", lambda *a, **k: response(), quota_path=quota_path)
    first.interpret("graphs")
    second = AIInterpreter("fake", tmp_path / "state2/cache", lambda *a, **k: response(), quota_path=quota_path)
    assert second.quota.status()["used"] == 1


def test_http_429_at_limit_and_local_search_available(engine, tmp_path):
    def no_network(*a, **k):
        pytest.fail("Quota must block before network access")
    interpreter = AIInterpreter("fake", tmp_path / "cache", no_network)
    seed(interpreter.quota, 1000)
    server = create_server(engine, 0, interpreter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        meta = json.loads(request(base + "/api/meta")[1])
        assert meta["ai"]["weekly_quota"]["remaining"] == 0
        auth = {"X-Local-Token": meta["csrf_token"]}
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base + "/api/interpret", {"prompt": "graphs"}, auth)
        assert error.value.code == 429
        assert json.loads(error.value.read())["weekly_quota"]["used"] == 1000
        result = json.loads(request(base + "/api/search", {"prompt": "graph neural networks"}, auth)[1])
        assert result["total"] > 0
        assert interpreter.quota.status()["used"] == 1000
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
