"""No paid API calls: strict output, privacy, cache, and HTTP integration."""
import json
import threading
from types import SimpleNamespace
import urllib.error

import pytest
import requests

from paper_atlas.ai_interpreter import AIInterpreter, InterpretationError, MODEL, load_api_key, validate_output
from paper_atlas.conference_web import create_server
from test_conference_web import engine, request


def answer():
    return {"groups": [{"label": "Graph models", "terms": ["graph neural networks", "graph transformers"]}],
            "supporting_groups": [],
            "exclusions": [], "needs_clarification": False, "explanation": "Search for graph models."}


def response(value=None, **overrides):
    body = {"status": "completed", "model": MODEL, "usage": {"input_tokens": 100, "output_tokens": 50},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value or answer())}]}]}
    body.update(overrides)
    return SimpleNamespace(status_code=200, json=lambda: body)


def test_private_key_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "key.env"
    assert load_api_key(path) is None
    path.write_text("# comment\nexport OPENAI_API_KEY='test-secret'\n")
    path.chmod(0o600)
    assert load_api_key(path) == "test-secret"
    path.chmod(0o644)
    with pytest.raises(InterpretationError, match="permissions"):
        load_api_key(path)
    monkeypatch.setenv("OPENAI_API_KEY", "env-test-secret")
    assert load_api_key(path) == "env-test-secret"


def test_key_file_is_not_executed(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "key.env"
    path.write_text("echo unsafe\nTOKEN=missing\n")
    path.chmod(0o600)
    with pytest.raises(InterpretationError, match="assignment"):
        load_api_key(path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(InterpretationError, match="regular file"):
        load_api_key(link)


def test_strict_request_and_memory_only_cache(tmp_path):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response()
    path = tmp_path / "cache.sqlite3"
    interpreter = AIInterpreter("test-secret", path, post)
    result = interpreter.interpret("Graph models, not just any learning algorithm")
    assert not result["cached"]
    payload = calls[0][1]["json"]
    assert payload["model"] == MODEL
    assert payload["text"]["format"]["strict"] is True
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "none"}
    assert calls[0][1]["allow_redirects"] is False
    assert "test-secret" not in json.dumps(result)
    assert not path.exists()
    assert interpreter.interpret(result["prompt"])["cached"]
    assert len(calls) == 1
    assert not AIInterpreter("test-secret", path, post).interpret(result["prompt"])["cached"]
    assert len(calls) == 2
    with pytest.raises(InterpretationError, match="description changed"):
        interpreter.load(result["id"], "Different prompt")


@pytest.mark.parametrize("change", [
    {"extra": "bad"}, {"needs_clarification": "false"}, {"groups": []},
    {"groups": [{"label": "bad", "terms": [123]}]}, {"exclusions": [123]},
    {"explanation": " "}, {"groups": [{"label": "x", "terms": ["x"], "extra": "x"}]},
])
def test_reject_invalid_structure(change):
    with pytest.raises(ValueError):
        validate_output({**answer(), **change})


@pytest.mark.parametrize("exclusions", [[], ["diffusion"], ["robotics", "image generation"]])
def test_ai_exclusions_and_rationale_survive_interpretation_and_cache(tmp_path, exclusions):
    generated = {**answer(), "exclusions": exclusions,
                 "explanation": "Model-provided rationale for this scope."}
    interpreter = AIInterpreter("test-secret", tmp_path / "cache", lambda *a, **k: response(generated))
    prompt = "Reinforcement learning for autoregressive language models"
    result = interpreter.interpret(prompt)
    assert result["exclusions"] == exclusions
    assert result["generated_concepts"]["exclusions"] == exclusions
    assert result["explanation"] == generated["explanation"]
    cached = interpreter.interpret(prompt)
    assert cached["cached"] and cached["exclusions"] == exclusions


def test_clarification_may_have_empty_groups():
    assert validate_output({**answer(), "needs_clarification": True, "groups": []})["needs_clarification"]


def test_bad_json_retried_once_never_extracted(tmp_path):
    calls = []
    def post(*a, **k):
        calls.append(k)
        return response(output=[{"type": "message", "content": [{"type": "output_text", "text": "```json\n" + json.dumps(answer()) + "\n```"}]}])
    interpreter = AIInterpreter("secret", tmp_path / "cache", post)
    with pytest.raises(InterpretationError, match="two attempts"):
        interpreter.interpret("graphs")
    assert len(calls) == 2


def test_incomplete_retry_then_success(tmp_path):
    calls = []
    def post(*a, **k):
        calls.append(k)
        return response(status="incomplete", incomplete_details={"reason": "max_output_tokens"}) if len(calls) == 1 else response()
    assert AIInterpreter("secret", tmp_path / "cache", post).interpret("graphs")["groups"]
    assert [c["json"]["max_output_tokens"] for c in calls] == [1800, 3000]


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_http_errors_sanitized_no_retry(tmp_path, status):
    calls = []
    def post(*a, **k):
        calls.append(k)
        return SimpleNamespace(status_code=status, json=lambda: {"error": "secret"})
    with pytest.raises(InterpretationError) as error:
        AIInterpreter("secret", tmp_path / "cache", post).interpret("graphs")
    assert "secret" not in str(error.value)
    assert len(calls) == 1


def test_network_errors_sanitized(tmp_path):
    def post(*a, **k):
        raise requests.Timeout("secret raw request")
    with pytest.raises(InterpretationError, match="could not be reached") as error:
        AIInterpreter("secret", tmp_path / "cache", post).interpret("graphs")
    assert "secret" not in str(error.value)


def test_insufficient_quota_is_actionable(tmp_path):
    def post(*a, **k):
        return SimpleNamespace(status_code=429, json=lambda: {"error": {"code": "insufficient_quota", "message": "secret"}})
    with pytest.raises(InterpretationError, match="quota is exhausted") as error:
        AIInterpreter("secret", tmp_path / "cache", post).interpret("graphs")
    assert "secret" not in str(error.value)


def test_refusal_not_retried(tmp_path):
    calls = []
    def post(*a, **k):
        calls.append(k)
        return response(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "raw"}]}])
    with pytest.raises(InterpretationError, match="declined"):
        AIInterpreter("secret", tmp_path / "cache", post).interpret("graphs")
    assert len(calls) == 1


def test_http_reviewed_ai_search_and_provenance(engine, tmp_path):
    from test_semantic_search import FakeEmbeddings
    engine.enable_semantics(tmp_path / "vectors", FakeEmbeddings())
    calls = []
    def post(*a, **k):
        calls.append(k)
        return response()
    interpreter = AIInterpreter("secret", tmp_path / "cache", post)
    server = create_server(engine, 0, interpreter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        meta = json.loads(request(base + "/api/meta")[1])
        assert meta["ai"]["enabled"] and meta["ai"]["model"] == MODEL
        assert meta["ai"]["weekly_quota"]["limit"] == 1000
        algorithm = json.loads(request(base + "/api/algorithm")[1])
        assert algorithm["ai"]["enabled"]
        assert algorithm["example"]["kind"] == "illustrative"
        assert not calls
        assert interpreter.quota.status()["used"] == 0
        auth = {"X-Local-Token": meta["csrf_token"]}
        # Local mode is still free and never reaches the interpreter.
        request(base + "/api/search", {"prompt": "graph neural networks"}, auth)
        keyword = json.loads(request(base + "/api/search", {
            "prompt": "graph neural networks, drug discovery", "search_mode": "keyword"}, auth)[1])
        assert keyword["total"] == 1 and "retrieval" not in keyword["config"]
        assert not calls
        assert interpreter.quota.status()["used"] == 0
        prompt = "I want graph methods, not a survey of every neural network"
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base + "/api/interpret", {"prompt": prompt})
        assert error.value.code == 403
        parsed = json.loads(request(base + "/api/interpret", {"prompt": prompt}, auth)[1])
        result = json.loads(request(base + "/api/search", {"prompt": prompt,
            "interpretation_id": parsed["id"], "years": [2024],
            "groups": [{"label": "Edited", "terms": ["drug discovery"]}], "exclusions": []}, auth)[1])
        assert result["total"] == 1
        assert result["config"]["interpretation"]["groups"] == answer()["groups"]
        assert result["config"]["groups"][0]["label"] == "Edited"
        assert "secret" not in json.dumps(result)
        assert engine.load(result["id"])["config"]["interpretation"]["model"] == MODEL
        assert len(calls) == 1
        with pytest.raises(urllib.error.HTTPError):
            request(base + "/api/search", {"prompt": "graph", "groups": answer()["groups"]}, auth)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_disabled_ai_never_calls_provider(engine):
    server = create_server(engine, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        meta = json.loads(request(base + "/api/meta")[1])
        assert not meta["ai"]["enabled"]
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base + "/api/interpret", {"prompt": "graphs"}, {"X-Local-Token": meta["csrf_token"]})
        assert b"AI is disabled" in error.value.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
