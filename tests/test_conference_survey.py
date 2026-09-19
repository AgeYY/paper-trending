"""Offline tests: metadata integrity, topic screening, and honest coverage."""
import csv
import json
from pathlib import Path
import sys
import types
import urllib.error

import numpy as np
import pytest

from paper_atlas.conference_corpus import JsonCache, dump_json, fetch_source, normalize_catalog, safe_link, sha256, track_of
from paper_atlas.conference_survey import DEFAULT_TOPICS, analyze, load_topics, main, match_topic, read_reviews, semantic_ranking


def event(event_id=1, **kwargs):
    return {"id": event_id, "name": "RL for language models", "abstract": "Reinforcement learning for language models.",
            "authors": [{"fullname": "A. Researcher"}], "eventtype": "Poster", "decision": "Accept (Poster)",
            "sourceurl": "https://openreview.net/group?id=ICLR.cc/2025/Conference",
            "paper_url": "https://openreview.net/forum?id=abc", **kwargs}


class FakeCache:
    def __init__(self, pages):
        self.pages = iter(pages)

    def get(self, url):
        return next(self.pages)


def test_duplicate_oral_changed_title_and_missing_decision():
    rows = [event(), event(2, name="New title", eventtype="Oral", decision=None)]
    papers, audit = normalize_catalog(rows, {}, "iclr", 2025)
    assert len(papers) == 1
    assert papers[0]["event_ids"] == ["1", "2"]
    assert audit["duplicate_presentations"] == 1
    assert audit["issues"] == []


def test_older_submission_ids_and_separate_abstracts():
    rows = [event(paper_url="", sourceid=10, abstract=""),
            event(2, name="Changed title", paper_url="", sourceid=-10, eventtype="Oral")]
    papers, _ = normalize_catalog(rows, {"1": "Recovered abstract"}, "iclr", 2025)
    assert len(papers) == 1
    assert papers[0]["abstract"] == "Recovered abstract"


@pytest.mark.parametrize("venue,year,source,title,expected", [
    ("icml", 2022, "https://cmt3.research.microsoft.com/api/odata/icml2022", "A paper", "research"),
    ("icml", 2024, "https://openreview.net/group?id=ICML.cc/2024/Conference", "Position: A paper", "position"),
    ("neurips", 2022, "https://openreview.net/group?id=NeurIPS.cc/2022/Track/Datasets_and_Benchmarks", "A paper", "datasets-and-benchmarks"),
    ("iclr", 2025, "https://openreview.net/group?id=TMLR", "A paper", "journal-to-conference"),
    ("neurips", 2023, "https://openreview.net/group?id=ML_Reproducibility_Challenge/2022", "A paper", "reproducibility-challenge"),
    ("iclr", 2025, "workshop", "A paper", "unknown"),
])
def test_tracks(venue, year, source, title, expected):
    assert track_of({"sourceurl": source, "name": title}, venue, year) == expected


def test_unknown_track_and_nonaccepted_research_are_audited():
    _, audit = normalize_catalog([event(sourceurl="unexpected")], {}, "iclr", 2025)
    assert audit["issues"][0]["reason"] == "unknown track"
    papers, audit = normalize_catalog([event(decision="Reject")], {}, "iclr", 2025)
    assert not papers
    assert audit["issues"]


def test_session_is_not_paper_even_with_authors():
    papers, audit = normalize_catalog([event(sourceurl=None, children_ids=[2, 3])], {}, "iclr", 2025)
    assert not papers
    assert audit["auxiliary_event_ids"] == ["1"]


def test_empty_source_is_unavailable_not_zero():
    papers, audit = fetch_source("neurips", 2026, FakeCache([{"count": 0, "results": []}]))
    assert papers == []
    assert audit["status"] == "unavailable"


def test_publication_links_reject_executable_schemes():
    assert safe_link("https://iclr.cc", "/paper") == "https://iclr.cc/paper"
    with pytest.raises(ValueError, match="Invalid publication URL"):
        safe_link("https://iclr.cc", "javascript:alert(1)")


@pytest.mark.parametrize("pages,match", [
    ([{"count": 2, "results": [event()]}], "counts disagree"),
    ([{"count": 2, "results": [event(), event()]}], "counts disagree"),
    ([{"count": 2, "results": [event()], "next": "https://example.org/page"}], "pagination host"),
    ([{"count": 2, "results": [event()], "next": "/page2"}, {"count": 3, "results": []}], "changed"),
])
def test_bad_catalog_rejected(pages, match):
    with pytest.raises(ValueError, match=match):
        fetch_source("iclr", 2025, FakeCache(pages))


def test_pagination_and_unique_papers():
    pages = [{"count": 2, "results": [event()], "next": "/page2"},
             {"count": 2, "results": [event(2, name="Another", paper_url="https://openreview.net/forum?id=def")]}]
    papers, audit = fetch_source("iclr", 2025, FakeCache(pages))
    assert len(papers) == 2
    assert audit["status"] == "complete"


def test_cache_integrity_and_offline(tmp_path):
    url, raw = "https://iclr.cc/test", b'{"count": 0}'
    body = tmp_path / "bodies" / (sha256(raw) + ".json")
    body.parent.mkdir()
    body.write_bytes(raw)
    dump_json(tmp_path / "urls" / (sha256(url.encode()) + ".json"), {"sha256": sha256(raw), "url": url})
    cache = JsonCache(tmp_path, offline=True)
    assert cache.get(url) == {"count": 0}
    body.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache.get(url)
    with pytest.raises(ValueError, match="offline cache"):
        cache.get(url + "missing")


def test_404_cache_supports_offline_replay(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise urllib.error.HTTPError("https://iclr.cc/missing", 404, "Not found", None, None)
    monkeypatch.setattr("urllib.request.urlopen", missing)
    for offline in (False, True):
        cache = JsonCache(tmp_path, offline=offline)
        with pytest.raises(urllib.error.HTTPError) as error:
            cache.get("https://iclr.cc/missing")
        assert error.value.code == 404
        assert cache.used["https://iclr.cc/missing"]["http_status"] == 404


def test_keyword_boundaries_case_and_model_groups():
    _, topics = load_topics(DEFAULT_TOPICS)
    ar, discrete, continuous, flow_image = topics
    assert match_topic({"title": "Supportive LLM", "abstract": "Appropriate methods"}, ar) is None  # PPO substring
    assert match_topic({"title": "GRPO for LLMs", "abstract": "Reward learning"}, ar)
    paper = {"title": "Reinforcement learning for masked diffusion language models", "abstract": "Discrete token generation."}
    assert match_topic(paper, discrete)
    assert match_topic(paper, continuous) is None
    assert match_topic(paper, flow_image) is None
    assert match_topic({"title": "Policy optimization for flow-matching image generation", "abstract": ""}, flow_image)


def test_custom_topics_exclusions_and_validation(tmp_path):
    path = tmp_path / "topics.json"
    config = {"version": 1, "topics": [{"id": "custom", "label": "Custom", "all_of": [{"any": ["a+b"]}], "none_of": ["robot"]}]}
    dump_json(path, config)
    _, topics = load_topics(path)
    assert match_topic({"title": "a+b", "abstract": ""}, topics[0])
    assert match_topic({"title": "a+b robot", "abstract": ""}, topics[0]) is None
    config["topics"][0]["all_of"] = ["undefined"]
    dump_json(path, config)
    with pytest.raises(ValueError, match="Unknown keyword"):
        load_topics(path)


def corpus_fixture(root):
    papers, _ = normalize_catalog([event(), event(2, name="Unrelated optimization", abstract="Mathematics.",
                                                    paper_url="https://openreview.net/forum?id=other")], {}, "iclr", 2025)
    raw = "".join(json.dumps(p) + "\n" for p in papers)
    (root / "papers.jsonl").write_text(raw)
    manifest = {"years": [2025, 2026], "venues": ["iclr"], "paper_count": 2, "corpus_sha256": sha256(raw.encode()),
                "coverage": [{"venue": "iclr", "year": 2025, "status": "complete", "paper_count": 2},
                             {"venue": "iclr", "year": 2026, "status": "unavailable", "paper_count": 0}]}
    dump_json(root / "manifest.json", manifest)
    return papers


def test_review_validation_and_manual_false_negative(tmp_path):
    papers = corpus_fixture(tmp_path)
    review = tmp_path / "review.csv"
    review.write_text(f"paper_id,topic_id,decision,reason\n{papers[1]['id']},ar_lm_rl,include,Checked full paper\n")
    summary = analyze(tmp_path, tmp_path / "out", DEFAULT_TOPICS, reviews_path=review)
    assert summary["screened_papers"] == 2
    assert summary["complete_sources"] == 1
    counts = list(csv.DictReader((tmp_path / "out/counts.csv").open()))
    row = next(r for r in counts if r["topic_id"] == "ar_lm_rl" and r["year"] == "2025")
    assert row["keyword_candidates"] == "1"
    assert row["confirmed_included"] == "1"
    assert next(r for r in counts if r["year"] == "2026")["keyword_candidates"] == ""
    assert (tmp_path / "out/topic_trends.svg").exists()
    assert (tmp_path / "out/topic_trends.pdf").exists()
    assert (tmp_path / "out/reviewed_trends.svg").exists()
    review.write_text("paper_id,topic_id,decision\nmissing,ar_lm_rl,include\n")
    with pytest.raises(ValueError, match="Unknown or out-of-scope"):
        read_reviews(review, {p["id"]: p for p in papers}, {"ar_lm_rl"})


def test_tampered_corpus_fails(tmp_path):
    corpus_fixture(tmp_path)
    with (tmp_path / "papers.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        analyze(tmp_path, tmp_path / "out", DEFAULT_TOPICS)


def test_cli_incomplete_exit_and_explicit_acknowledgement(tmp_path):
    corpus_fixture(tmp_path)
    args = ["analyze", "--root", str(tmp_path)]
    assert main(args) == 2
    assert main(args + ["--allow-incomplete"]) == 0


def test_semantic_cpu_ranking_and_cache(tmp_path, monkeypatch):
    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False): return 2
        def encode(self, text, **kwargs): return [ord(c) for c in text]
        def decode(self, ids, **kwargs): return "".join(chr(i) for i in ids)

    class Encoder:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"
            self.tokenizer, self.max_seq_length = Tokenizer(), 512

        def encode(self, texts, **kwargs):
            return np.asarray([[1., 0.] if "reward" in t else [0., 1.] for t in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    papers = [{"id": "a", "title": "reward", "abstract": "reward " * 200, "year": 2025, "venue": "iclr", "url": "https://example.org"},
              {"id": "b", "title": "other", "abstract": "other", "year": 2025, "venue": "iclr", "url": "https://example.org"}]
    topics = [{"id": "t", "description": "reward", "query": "reward"}]
    result = semantic_ranking(papers, topics, tmp_path, model_name="test-model", top_k=1)
    assert ("a", "t") in result
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert semantic_ranking(papers, topics, tmp_path, model_name="test-model", top_k=1) == result
