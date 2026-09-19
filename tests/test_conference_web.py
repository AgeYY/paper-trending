"""Local-only tests for prompt retrieval, persistence, HTTP and exports."""
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from paper_atlas.conference_corpus import dump_json, sha256
from paper_atlas.conference_search import SearchEngine, interpret_prompt, validate_groups
from paper_atlas.conference_web import create_server, csv_bytes


@pytest.fixture
def engine(tmp_path):
    specs = [
        ("Masked diffusion language models with reinforcement learning", "We optimize discrete tokens with GRPO.", 2025, "iclr"),
        ("Flow matching image generation with reinforcement learning", "A rectified flow text-to-image model with PPO.", 2025, "icml"),
        ("Graph neural networks for drug discovery", "Molecular generation with graph transformers.", 2024, "iclr"),
        ("Reinforcement learning for autoregressive language models", "Policy optimization of LLMs.", 2025, "iclr"),
        ("Continuous diffusion language models", "Policy gradient updates for continuous embedding space text generation.", 2024, "icml"),
        ("Private graph learning", "Differential privacy for graph neural networks, with federated learning.", 2024, "iclr"),
    ]
    papers = [{"id": f"paper:{i}", "title": title, "abstract": abstract, "year": year, "venue": venue,
               "track": "research", "authors": ["A Researcher"], "url": "https://example.org/paper"}
              for i, (title, abstract, year, venue) in enumerate(specs)]
    raw = "".join(json.dumps(p) + "\n" for p in papers)
    (tmp_path / "papers.jsonl").write_text(raw)
    dump_json(tmp_path / "manifest.json", {"corpus_sha256": sha256(raw.encode()), "paper_count": len(papers),
              "created_at": "2026-09-17T00:00:00+00:00", "years": [2024, 2025, 2026],
              "coverage": [{"venue": v, "year": y, "status": "complete" if y < 2026 else "unavailable"}
                           for v in ["iclr", "icml", "neurips"] for y in [2024, 2025, 2026]]})
    return SearchEngine(tmp_path, tmp_path / "state")


@pytest.mark.parametrize("prompt,labels", [
    ("Reinforcement learning for autoregressive language models", ["Reinforcement learning", "Autoregressive / general LMs", "Language generation"]),
    ("RL for discrete diffusion language models", ["Reinforcement learning", "Discrete state / masking", "Diffusion", "Language generation"]),
    ("RL for continuous diffusion language models", ["Reinforcement learning", "Continuous / latent state", "Diffusion", "Language generation"]),
    ("RL for flow matching image generation", ["Reinforcement learning", "Flow matching", "Image generation"]),
    ("Please find papers about graph neural networks for drug discovery", ["Graph neural networks", "Drug discovery"]),
    ("Efficient transformers", ["efficient", "transformer"]),
])
def test_prompt_interpretation(prompt, labels):
    assert [g["label"] for g in interpret_prompt(prompt)["groups"]] == labels


def test_prompt_exclusions_and_unsupported_negation():
    result = interpret_prompt("Differential privacy excluding federated learning")
    assert result["exclusions"] == ["federated learning"]
    with pytest.raises(ValueError, match="exclusions"):
        interpret_prompt("diffusion not images")


@pytest.mark.parametrize("bad", [[], [{"terms": []}], [{"terms": [1]}], [{"terms": ["x"], "label": 123}]])
def test_invalid_groups(bad):
    with pytest.raises(ValueError):
        validate_groups(bad)


def test_search_groups_are_and_phrases_are_or(engine):
    result = engine.search({"prompt": "RL for discrete diffusion language models", "years": [2024, 2025, 2026]})
    assert [p["id"] for p in result["papers"]] == ["paper:0"]
    assert result["total"] == 1
    assert result["incomplete"]
    assert all(r["candidates"] is None for r in result["counts"] if r["year"] == 2026)
    assert result["papers"][0]["evidence"]


def test_unknown_topic_literal_matching_and_empty(engine):
    result = engine.search({"prompt": "molecular generation", "years": [2024]})
    assert result["total"] == 1
    assert engine.search({"prompt": "zzzxxyy unheardtopic"})["total"] == 0


def test_contiguous_phrases_not_bag_of_words(engine):
    assert engine.phrase_matches("graph neural network") == {2, 5}
    assert engine.phrase_matches("graph discovery") == set()
    assert engine.phrase_matches("PPO") == {1}
    assert engine.phrase_matches("appropriate") == set()


def test_venue_year_track_filters_and_exclusions(engine):
    assert engine.search({"prompt": "RL for flow matching image generation", "venues": ["iclr"]})["total"] == 0
    assert engine.search({"prompt": "Differential privacy excluding federated learning"})["total"] == 0
    assert engine.search({"prompt": "graph neural networks", "years": [2025]})["total"] == 0


def test_transient_review_and_deterministic_search_within_process(engine):
    result = engine.search({"prompt": "Reinforcement learning language models"})
    assert not (engine.state_root / "searches.sqlite3").exists()
    engine.review(result["id"], "paper:0", "include")
    again = engine.search({"prompt": "Reinforcement learning language models"})
    assert result["id"] == again["id"]
    assert again["review_counts"]["include"] == 1
    restarted = SearchEngine(engine.root, engine.state_root)
    with pytest.raises(KeyError, match="expired"):
        restarted.load(result["id"])
    assert len(engine.page(result["id"], review="include")["papers"]) == 1
    with pytest.raises(ValueError, match="Invalid decision"):
        engine.review(result["id"], "missing", "include")
    engine.review(result["id"], "paper:0", "unreviewed")
    assert "include" not in engine.page(result["id"])["review_counts"]


def test_refinement_broadening_and_validation(engine):
    query = {"prompt": "discrete diffusion language models", "mode": "any"}
    broad = engine.search(query)
    query["mode"] = "all"
    narrow = engine.search(query)
    assert broad["total"] > narrow["total"]
    assert broad["id"] != narrow["id"]
    with pytest.raises(ValueError):
        engine.search({"prompt": "graph learning", "venues": []})
    with pytest.raises(ValueError):
        engine.page(narrow["id"], page=0)


def test_comma_keywords_are_literal_required_phrases(engine):
    result = engine.search({"prompt": "reinforcement learning, language model", "search_mode": "keyword"})
    assert result["total"] == 2
    assert result["config"]["groups"] == [
        {"label": "reinforcement learning", "terms": ["reinforcement learning"]},
        {"label": "language model", "terms": ["language model"]}]
    assert result["config"]["exclusions"] == []
    assert "retrieval" not in result["config"] and "interpretation" not in result["config"]
    assert engine.search({"prompt": "GNN", "search_mode": "keyword"})["total"] == 0
    assert engine.search({"prompt": "graph neural networks", "search_mode": "keyword"})["total"] == 2
    assert engine.search({"prompt": "RL", "search_mode": "keyword"})["total"] == 0
    assert engine.search({"prompt": " graph neural networks, ,graph neural networks, ", "search_mode": "keyword"})["total"] == 2


@pytest.mark.parametrize("prompt", [", ,", "!!!", "a" * 121, ",".join(f"term{i}" for i in range(17))])
def test_invalid_keyword_phrases_rejected(engine, prompt):
    with pytest.raises(ValueError):
        engine.search({"prompt": prompt, "search_mode": "keyword"})


def test_keyword_mode_works_without_ai_server(server):
    meta = json.loads(request(server + "/api/meta")[1])
    assert not meta["ai"]["enabled"]
    auth = {"X-Local-Token": meta["csrf_token"]}
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(server + "/api/history")
    assert caught.value.code == 404
    query = {"prompt": "graph neural networks, drug discovery", "search_mode": "keyword"}
    result = json.loads(request(server + "/api/search", query, auth)[1])
    assert result["total"] == 1 and result["config"]["search_mode"] == "keyword"
    for extra in [{"interpretation_id": "fake"}, {"groups": []}, {"search_mode": "unknown"}, {"search_mode": "ai"}]:
        with pytest.raises(urllib.error.HTTPError) as caught:
            request(server + "/api/search", {**query, **extra}, auth)
        assert caught.value.code == 400


def test_export_prevents_spreadsheet_formulas():
    data = csv_bytes([{"title": "=CMD()"}], ["title"]).decode("utf-8-sig")
    assert "'=CMD()" in data


@pytest.fixture
def server(engine):
    app = create_server(engine, 0)
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{app.server_port}"
    app.shutdown(); app.server_close(); thread.join(timeout=2)


def request(url, body=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, response.read(), response.headers


def test_http_search_automatic_counts_and_exports(server, engine):
    status, body, headers = request(server + "/")
    assert status == 200 and b"Paper Trending" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    _, body, _ = request(server + "/api/meta")
    token = json.loads(body)["csrf_token"]
    auth = {"X-Local-Token": token}
    _, body, _ = request(server + "/api/search", {"prompt": "RL for flow matching image generation"}, auth)
    result = json.loads(body)
    assert result["total"] == 1
    assert "review_counts" not in result
    assert "decision" not in result["papers"][0]
    assert all("confirmed" not in r for r in result["counts"])
    assert len(result["acceptance"]["rows"]) == 5
    assert {r["venue"] for r in result["acceptance"]["rows"]} == {"iclr"}
    assert all(r["topic_rate"] is None for r in result["acceptance"]["rows"])
    # Existing local decisions are preserved but cannot affect the web count.
    engine.review(result["id"], "paper:1", "exclude")
    _, body, _ = request(server + "/api/search/" + result["id"] + "?review=include")
    assert json.loads(body)["total"] == 1
    assert len(json.loads(body)["papers"]) == 1
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server + "/api/review", {"search_id": result["id"], "paper_id": "paper:1", "decision": "include"}, auth)
    assert error.value.code == 404
    assert engine.decisions(result["id"])["paper:1"] == "exclude"
    for ext, expected in [("csv", "text/csv"), ("counts", "text/csv"), ("json", "application/json"), ("svg", "image/svg+xml"), ("png", "image/png"), ("pdf", "application/pdf")]:
        _, body, headers = request(server + f"/api/export/{result['id']}.{ext}")
        assert expected in headers["Content-Type"]
        assert body
        assert "attachment" in headers["Content-Disposition"]
        if ext in ("csv", "counts", "json"):
            assert b'"confirmed"' not in body and b'"decisions"' not in body
        if ext == "json":
            assert json.loads(body)["acceptance"]["metadata"]["sha256"]
    for ext, magic in [("svg", b"<?xml"), ("png", b"\x89PNG"), ("pdf", b"%PDF")]:
        _, body, headers = request(server + f"/api/export/{result['id']}.{ext}?chart=acceptance")
        assert body.startswith(magic)
        assert "acceptance" in headers["Content-Disposition"]
    _, body, _ = request(server + f"/api/export/{result['id']}.acceptance")
    assert b"topic_submitted" in body and b"baseline_source" in body
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server + f"/api/export/{result['id']}.svg?chart=invalid")
    assert error.value.code == 400


def test_algorithm_page_uses_current_rules(server):
    _, body, _ = request(server + "/algorithm")
    assert b"How search and counting work" in body
    assert b"60 + keyword rank" in body
    assert b"strict JSON schema" in body
    assert b"not a live AI response" in body
    assert b"algorithm-vocabulary" not in body
    assert b"small synonym dictionary" not in body
    _, body, _ = request(server + "/api/algorithm")
    data = json.loads(body)
    assert [g["label"] for g in data["example"]["groups"]] == ["Reinforcement learning", "Flow matching", "Image generation"]
    assert data["catalog_publications"] == 6
    assert data["example"]["kind"] == "illustrative"
    assert "vocabulary" not in data
    from paper_atlas.ai_interpreter import MODEL, VERSION, SCHEMA
    from paper_atlas.weekly_limit import WEEKLY_LIMIT, TIMEZONE
    assert data["ai"]["model"] == MODEL
    assert data["ai"]["prompt_version"] == VERSION
    assert data["ai"]["fields"] == SCHEMA["required"]
    assert data["ai"]["max_groups"] == 8 and data["ai"]["max_terms"] == 12
    assert data["ai"]["weekly_limit"] == WEEKLY_LIMIT
    assert data["ai"]["timezone"] == TIMEZONE
    for path in ("/compact.css", "/algorithm.js", "/algorithm/"):
        assert request(server + path)[0] == 200


def test_http_rejects_cross_site_bad_host_and_traversal(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server + "/api/search", {"prompt": "diffusion"})
    assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server + "/api/meta", headers={"Host": "evil.example"})
    assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server + "/../../AGENTS.md")
    assert error.value.code == 404
    _, body, _ = request(server + "/api/meta")
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server + "/api/search", {"prompt": "diffusion"}, {"X-Local-Token": json.loads(body)["csrf_token"], "Origin": "https://evil.example"})
    assert error.value.code == 403
