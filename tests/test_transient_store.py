from paper_atlas.transient_store import TransientStore
from paper_atlas.conference_search import SearchEngine
from paper_atlas.ai_interpreter import AIInterpreter, InterpretationError
from test_conference_web import engine
from test_ai_interpreter import response
import pytest


def test_store_is_bounded_and_returns_independent_values():
    store = TransientStore(capacity=2)
    original = {"terms": ["graphs"]}
    store.put("a", original)
    original["terms"].append("changed")
    assert store.get("a") == {"terms": ["graphs"]}
    store.get("a")["terms"].append("changed again")
    store.put("b", {}); store.put("c", {})
    with pytest.raises(KeyError):
        store.get("a")
    assert store.get("b") == {}


def test_old_query_databases_are_not_opened_or_modified(engine, tmp_path):
    engine.state_root.mkdir(parents=True, exist_ok=True)
    old_searches = engine.state_root / "searches.sqlite3"
    old_interpretations = engine.state_root / "interpretations.sqlite3"
    for path in (old_searches, old_interpretations):
        path.write_bytes(b"not a database: retained legacy file")
    restarted = SearchEngine(engine.root, engine.state_root)
    found = restarted.search({"prompt": "graph neural networks", "search_mode": "keyword"})
    assert found["total"] == 2
    interpreter = AIInterpreter("secret", old_interpretations, lambda *a, **k: response(),
                                quota_path=tmp_path / "quota.sqlite3")
    interpreted = interpreter.interpret("graph models")
    new_interpreter = AIInterpreter("secret", old_interpretations, lambda *a, **k: response(),
                                    quota_path=tmp_path / "quota.sqlite3")
    with pytest.raises(InterpretationError):
        new_interpreter.load(interpreted["id"], "graph models")
    for path in (old_searches, old_interpretations):
        assert path.read_bytes() == b"not a database: retained legacy file"
    assert b"graph models" not in (tmp_path / "quota.sqlite3").read_bytes()
