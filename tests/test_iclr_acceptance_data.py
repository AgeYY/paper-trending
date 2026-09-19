"""Synthetic official-reply fixtures; never modify the live downloaded corpus."""
import json

import pytest

from paper_atlas.conference_acceptance import AcceptanceStore
from paper_atlas.conference_corpus import dump_json, sha256
from paper_atlas.iclr_acceptance_data import audit_download, main, normalize_record, validate_public_snapshot


@pytest.fixture
def download(tmp_path):
    records = []
    for i, decision in enumerate(("Accept (Poster)", "Accept: notable-top-5%", "Reject", None, None)):
        row = {"id": str(i), "title": "Graph networks" if i != 1 else "Image synthesis",
               "abstract": "Complete abstract with a literal\u2028paragraph separator.",
               "source": f"https://openreview.net/forum?id={i}",
               "decision": decision, "decision_status": "resolved" if decision else "not_public_or_absent",
               "decision_evidence": [], "venue": "ICLR 2025 Conference Withdrawn Submission" if i in (0, 3) else "",
               "venueid": "ICLR.cc/2025/Conference", "invitation": "ICLR.cc/2025/Conference/-/Submission"}
        if decision:
            row["decision_evidence"] = [{"id": f"d{i}", "forum": str(i), "decision": decision,
                                         "invitations": [f"ICLR.cc/2025/Conference/Submission{i}/-/Decision"]}]
        elif i == 4:
            row["venue"] = "ICLR 2025 Conference Desk Rejected Submission"
        records.append(row)
    raw = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records).encode()
    (tmp_path/"iclr-2025.jsonl").write_bytes(raw)
    summary = {"venue": "iclr", "year": 2025, "sha256": sha256(raw), "records_saved": 5,
               "api_visible_count_at_start": 5, "api_count_matches_unique_returned": True,
               "decisions_requested": True, "excluded_nonpublic_or_deleted": 0, "conflicting_decisions": 0,
               "public_query_exhausted": True, "retrieved_at": "2026-09-17T00:00:00Z",
               "records_with_official_decision": 3, "records_without_official_decision": 2}
    dump_json(tmp_path/"iclr-2025.summary.json", summary)
    return tmp_path, summary, records


def query(terms):
    return {"years": [2025], "venues": ["iclr", "icml", "neurips"], "tracks": ["research"],
            "groups": [{"label": "topic", "terms": terms}], "exclusions": [], "mode": "all"}


def install(root, data):
    dump_json(root/"acceptance"/"iclr-public-2025.json", data)
    baseline = {"venue": "iclr", "year": 2025, "accepted": 1, "submitted": 4,
                "source": "https://example.org/organizer", "population": "Different published cohort"}
    path = root/"baselines.json"
    dump_json(path, {"checked_at": "2026-09-17", "rows": [baseline]})
    return AcceptanceStore(root, path)


def test_official_public_pool_same_denominator_and_separate_reference(download):
    root, _, _ = download
    data = audit_download(root, 2025)
    assert data["accepted"] == 2 and data["submitted"] == 5
    assert data["complete"] is False and data["usable"] is True
    assert data["status_counts"] == {"accepted": 2, "rejected": 1, "withdrawn": 1, "desk_rejected": 1}
    store = install(root, data)
    row = store.compute(query(["graph"]))["rows"][0]
    assert row["topic_rate"] == 25 and row["conference_rate"] == 40 and row["delta_pp"] == -15
    assert row["official_conference_rate"] == 25  # NOT the comparison baseline
    assert row["baseline_kind"] == "official_public_snapshot"
    assert row["topic_status"] == "public_snapshot"
    assert row["topic_source"]["input_sha256"] == data["input_sha256"]
    assert store.metadata()["public_snapshot_cohorts"] == 1
    assert store.metadata()["verified_topic_cohorts"] == 0
    assert store.compute(query(["unmatched"]))["rows"][0]["topic_rate"] is None
    assert store.compute(query(["image"]))["rows"][0]["topic_rate"] == 100


def test_decision_precedes_later_withdrawal_and_unknown_is_not_rejected(download):
    _, _, records = download
    assert normalize_record(records[0], 2025)["status"] == "accepted"
    row = records[3]
    row["venue"] = "Submitted to ICLR 2025"
    with pytest.raises(ValueError, match="Missing decision"):
        normalize_record(row, 2025)


@pytest.mark.parametrize("change,match", [
    ({"decision_status": "conflict"}, "Conflicting"),
    ({"decision": "Reject"}, "Conflicting"),
    ({"decision_evidence": []}, "no official evidence"),
])
def test_inconsistent_evidence_fails(download, change, match):
    row = download[2][0]
    row.update(change)
    with pytest.raises(ValueError, match=match):
        normalize_record(row, 2025)


def test_nonofficial_reply_rejected(download):
    row = download[2][0]
    row["decision_evidence"][0]["invitations"] = ["ICLR.cc/2025/Conference/Submission0/-/Official_Review"]
    with pytest.raises(ValueError, match="invitation"):
        normalize_record(row, 2025)


def test_partial_download_has_no_rates_but_retains_published_reference(download):
    root, summary, _ = download
    summary.update(public_query_exhausted=False, error="RateLimitError", api_visible_count_at_start=10)
    dump_json(root/"iclr-2025.summary.json", summary)
    data = audit_download(root, 2025)
    assert data["usable"] is False and "5 / 10" in data["reason"]
    row = install(root, data).compute(query(["graph"]))["rows"][0]
    assert row["topic_rate"] is None and row["conference_rate"] is None and row["delta_pp"] is None
    assert row["official_conference_rate"] == 25
    assert "RateLimitError" in row["topic_reason"]


def test_hash_and_summary_checked(download):
    root, summary, _ = download
    summary["sha256"] = "0" * 64
    dump_json(root/"iclr-2025.summary.json", summary)
    with pytest.raises(ValueError, match="hash"):
        audit_download(root, 2025)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(complete=True), "completeness"),
    (lambda d: d.update(api_visible_count=10), "count"),
    (lambda d: d.update(accepted=3), "counts"),
    (lambda d: d["records"][0].update(status="withdrawn"), "status"),
    (lambda d: d["records"][0].update(abstract=""), "abstract"),
    (lambda d: d["records"][1].update(id="0"), "duplicate"),
])
def test_snapshot_validation(download, mutation, match):
    data = audit_download(download[0], 2025)
    mutation(data)
    with pytest.raises(ValueError, match=match):
        validate_public_snapshot(data, {"year": 2025})


def test_import_idempotent_and_refuses_changed_overwrite(download):
    root = download[0]
    args = ["--download", str(root), "--corpus", str(root/"corpus"), "--years", "2025"]
    main(args)
    main(args)
    target = root/"corpus"/"acceptance"/"iclr-public-2025.json"
    data = json.loads(target.read_text())
    data["coverage_audit"] += " Modified"
    dump_json(target, data)
    with pytest.raises(SystemExit):
        main(args)
