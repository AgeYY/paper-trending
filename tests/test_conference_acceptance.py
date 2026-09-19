"""Synthetic cohorts only: never promote fixtures to the user's live corpus."""
import json

import pytest

from paper_atlas.conference_acceptance import AcceptanceStore, validate_snapshot, main
from paper_atlas.conference_corpus import dump_json
from paper_atlas.conference_web import acceptance_chart_bytes


@pytest.fixture
def cohort(tmp_path):
    baseline = {"venue": "iclr", "year": 2025, "accepted": 2, "submitted": 5,
                "population": "All five fixture full submissions, including withdrawals and desk rejections",
                "source": "https://example.org/official-totals"}
    data = {"venue": "iclr", "year": 2025, "track": "research", "complete": True,
            "population": baseline["population"], "baseline_source": baseline["source"],
            "source": "https://example.org/submissions", "retrieved_at": "2026-09-17T00:00:00Z",
            "coverage_audit": "Synthetic complete cohort for tests only.",
            "records": [
                {"id": "1", "title": "Graph neural networks", "abstract": "Reinforcement learning with PPO.", "status": "accepted"},
                {"id": "2", "title": "Image synthesis", "abstract": "Reinforcement learning with GRPO.", "status": "accepted"},
                {"id": "3", "title": "Graph neural networks", "abstract": "Supervised graph prediction.", "status": "rejected"},
                {"id": "4", "title": "Graph neural network", "abstract": "Federated graph prediction.", "status": "withdrawn"},
                {"id": "5", "title": "Graph neural networks", "abstract": "Another graph prediction.", "status": "desk_rejected"},
            ]}
    baselines = tmp_path/"baselines.json"
    dump_json(baselines, {"checked_at": "2026-09-17", "rows": [baseline]})
    dump_json(tmp_path/"acceptance"/"iclr-2025.json", data)
    return tmp_path, baselines, baseline, data


def config(terms=None):
    return {"years": [2025], "venues": ["iclr"], "tracks": ["research"], "mode": "all",
            "groups": [{"label": "GNN", "terms": terms or ["graph neural networks"]}], "exclusions": []}


def test_topic_rate_and_matched_cohort_baseline(cohort):
    root, path, baseline, data = cohort
    store = AcceptanceStore(root, path)
    row = store.compute(config())["rows"][0]
    assert (row["topic_accepted"], row["topic_submitted"], row["topic_rate"]) == (1, 4, 25.)
    assert row["conference_rate"] == 40
    assert row["delta_pp"] == -15
    assert row["topic_status"] == "complete"
    assert row["topic_source"]["coverage_audit"]
    above = store.compute(config(["reinforcement learning"]))["rows"][0]
    assert above["topic_rate"] == 100 and above["delta_pp"] == 60
    # Cached results cannot be corrupted by caller mutation.
    row["topic_rate"] = 99
    assert store.compute(config())["rows"][0]["topic_rate"] == 25


def test_shared_normalization_and_exclusions(cohort):
    store = AcceptanceStore(cohort[0], cohort[1])
    query = config(["GRAPH-NEURAL NETWORKS"])
    query["exclusions"] = ["federated"]
    row = store.compute(query)["rows"][0]
    assert row["topic_submitted"] == 3
    assert row["topic_rate"] == pytest.approx(100/3)
    query["groups"].append({"label": "RL", "terms": ["PPO", "GRPO"]})
    assert store.compute(query)["rows"][0]["topic_submitted"] == 1
    query["mode"] = "any"
    assert store.compute(query)["rows"][0]["topic_submitted"] == 4


def test_zero_is_distinct_from_no_denominator(cohort):
    store = AcceptanceStore(cohort[0], cohort[1])
    zero = store.compute(config(["supervised"]))["rows"][0]
    assert zero["topic_rate"] == 0 and zero["topic_submitted"] == 1
    empty = store.compute(config(["zzzz"]))["rows"][0]
    assert empty["topic_rate"] is None and empty["topic_submitted"] == 0
    assert empty["topic_status"] == "no_matches"
    assert empty["delta_pp"] is None


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x.update(complete=False), "complete"),
    (lambda x: x.update(population="accepted and rejected only"), "populations"),
    (lambda x: x.update(baseline_source="https://example.org/other"), "populations"),
    (lambda x: x.update(coverage_audit=""), "audit"),
    (lambda x: x["records"].pop(), "count"),
    (lambda x: x["records"][1].update(id="1"), "duplicate"),
    (lambda x: x["records"][1].update(status="unknown"), "Unresolved"),
    (lambda x: x["records"][1].update(status="rejected"), "Accepted count"),
    (lambda x: x["records"][1].update(abstract=""), "incomplete"),
    (lambda x: x.update(track="position"), "track"),
    (lambda x: x.update(source="javascript:alert(1)"), "source"),
])
def test_reject_biased_or_unreconciled_pool(cohort, mutation, match):
    root, path, baseline, data = cohort
    mutation(data)
    with pytest.raises(ValueError, match=match):
        validate_snapshot(data, baseline)
    dump_json(root/"acceptance"/"iclr-2025.json", data)
    row = AcceptanceStore(root, path).compute(config())["rows"][0]
    assert row["topic_rate"] is None
    assert row["conference_rate"] == 40


def test_all_venues_years_missing_and_track_scope(tmp_path):
    store = AcceptanceStore(tmp_path)
    query = config()
    query.update(years=list(range(2022, 2027)), venues=["iclr", "icml", "neurips"])
    data = store.compute(query)
    assert len(data["rows"]) == 5
    assert all(r["topic_rate"] is None and r["delta_pp"] is None for r in data["rows"])
    assert all(r["venue"] == "iclr" for r in data["rows"])
    query["tracks"] = ["research", "position"]
    assert all(r["conference_rate"] is None for r in store.compute(query)["rows"])


def test_no_invented_denominator_from_rounded_rate(cohort):
    root, path, baseline, data = cohort
    baseline.pop("submitted")
    baseline["reported_rate"] = 40
    dump_json(path, {"checked_at": "2026-09-17", "rows": [baseline]})
    row = AcceptanceStore(root, path).compute(config())["rows"][0]
    assert row["conference_rate"] == 40
    assert row["conference_submitted"] is None and row["topic_rate"] is None


def test_snapshot_hash_changes_and_import_validates(cohort, monkeypatch):
    root, path, baseline, data = cohort
    first = AcceptanceStore(root, path).snapshot_hash
    data["coverage_audit"] += " Updated audit."
    dump_json(root/"acceptance"/"iclr-2025.json", data)
    assert AcceptanceStore(root, path).snapshot_hash != first
    import paper_atlas.conference_acceptance as module
    monkeypatch.setattr(module, "AcceptanceStore", lambda target: AcceptanceStore(target, path))
    incoming = root/"incoming.json"
    dump_json(incoming, data)
    with pytest.raises(SystemExit):
        main(["--corpus", str(root), "--import-snapshot", str(incoming)])
    main(["--corpus", str(root), "--import-snapshot", str(incoming), "--replace"])
    assert json.loads((root/"acceptance"/"iclr-2025.json").read_text()) == data


@pytest.mark.parametrize("extension,magic", [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")])
def test_acceptance_exports_with_real_and_missing_values(cohort, extension, magic):
    query = config(); query["venues"].append("neurips"); query["years"].append(2024)
    result = {"config": query, "acceptance": AcceptanceStore(cohort[0], cohort[1]).compute(query)}
    raw = acceptance_chart_bytes(result, extension)
    assert raw.startswith(magic)
    if extension == "svg":
        assert b"Acceptance (%)" in raw and b"NA" in raw
