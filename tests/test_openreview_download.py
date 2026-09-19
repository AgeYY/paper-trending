import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from paper_atlas.openreview_download import collect_cohort, download_public_metadata, public_record, public_decisions


def note(id, readers=None, **fields):
    return SimpleNamespace(id=id, readers=["everyone"] if readers is None else readers,
                           content=fields, ddate=None)


def client_with_pages(*pages):
    client = Mock()
    client.get_group.return_value = SimpleNamespace(content={"submission_name": {"value": "Submission"}})
    client.get_notes.side_effect = pages
    return client


def test_public_allowlist():
    row = public_record(note("abc", title={"value": "Public title"},
        abstract={"value": "private-text", "readers": ["~Someone1"]},
        authorids={"value": ["secret@example.com"]}))
    assert row["title"] == "Public title" and row["abstract"] is None
    assert "secret@example.com" not in json.dumps(row)
    assert public_record(note("private", readers=["~Someone1"])) is None
    assert public_record(note("unknown", readers=[])) is None


def test_pagination_and_counts(tmp_path):
    a = note("a", title={"value": "A"}, abstract={"value": "Text"}, venue={"value": "Reject"})
    b = note("b", readers=["~Someone1"], title={"value": "PRIVATE"})
    c = note("c", title="C", abstract="Text", venue="Poster")
    client = client_with_pages(([a, b], 3), [c], [])
    summary = collect_cohort(client, "iclr", 2025, tmp_path, page_size=2, pause=0)
    assert summary["records_saved"] == 2
    assert summary["excluded_nonpublic_or_deleted"] == 1
    assert summary["api_count_matches_unique_returned"] is True
    assert summary["public_query_exhausted"] is True and summary["complete"] is False
    assert "PRIVATE" not in (tmp_path / "iclr-2025.jsonl").read_text()
    assert client.get_notes.call_args_list[1].kwargs["after"] == "b"
    assert client.get_notes.call_args_list[0].kwargs["invitation"] == "ICLR.cc/2025/Conference/-/Submission"


def test_partial_failure_preserved_and_redacted(tmp_path):
    client = client_with_pages(([note("a", title="A")], 2),
        Exception({"name": "ChallengeRequiredError", "message": "private-token"}))
    with pytest.raises(Exception):
        collect_cohort(client, "iclr", 2025, tmp_path, pause=0)
    summary = json.loads((tmp_path / "iclr-2025.summary.json").read_text())
    assert summary["records_saved"] == 1 and not summary["public_query_exhausted"]
    assert summary["error"] == "ChallengeRequiredError"
    assert "private-token" not in json.dumps(summary)


def test_challenge_stops_other_venues(tmp_path, capsys):
    client = client_with_pages(Exception({"name": "ChallengeRequiredError"}))
    assert download_public_metadata(client, tmp_path / "out", ["iclr", "icml"], [2025]) == 1
    assert client.get_group.call_count == 1
    assert (tmp_path / "out" / "manifest.json").exists()


def test_duplicate_page_cannot_loop_forever(tmp_path):
    a = note("a", title="A")
    client = client_with_pages(([a], 1), [a])
    with pytest.raises(ValueError, match="advancing"):
        collect_cohort(client, "iclr", 2025, tmp_path, pause=0)
    assert client.get_notes.call_count == 2


def test_existing_directory_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        download_public_metadata(Mock(), tmp_path)


def test_zero_records_never_complete(tmp_path):
    summary = collect_cohort(client_with_pages(([], 0)), "icml", 2025, tmp_path)
    assert summary["records_saved"] == 0 and summary["complete"] is False


def test_legacy_collections_and_deduplication(tmp_path, monkeypatch):
    from paper_atlas import openreview_download as mod
    active = client_with_pages()
    active.get_group.return_value = SimpleNamespace(content={})
    a, b = note("a", title="A"), note("b", title="B")
    legacy = client_with_pages(([a], 1), [], ([a, b], 2), [], ([], 0), ([], 0))
    factory = Mock(return_value=legacy)
    monkeypatch.setattr(mod, "legacy_client_from", factory)
    summary = collect_cohort(active, "iclr", 2023, tmp_path, pause=0)
    factory.assert_called_once_with(active)
    assert summary["api"] == "https://api.openreview.net"
    assert len(summary["queries"]) == 4
    assert summary["records_saved"] == 2 and summary["duplicates"] == 1
    assert not summary["complete"]


def test_missing_year_continues(tmp_path):
    client = client_with_pages(([], 0))
    client.get_group.side_effect = [Exception({"name": "NotFoundError"}),
        SimpleNamespace(content={"submission_name": {"value": "Submission"}})]
    assert download_public_metadata(client, tmp_path / "out", ["icml"], [2022, 2023]) == 1
    assert client.get_group.call_count == 2
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert len(manifest["cohorts"]) == 2


def test_in_memory_legacy_session(monkeypatch):
    from paper_atlas import openreview_auth_check as auth
    from paper_atlas.openreview_download import legacy_client_from
    active = Mock(headers={"Authorization": "Bearer test-only"}, token="test-only")
    legacy = Mock(headers={})
    factory = Mock(return_value=legacy)
    monkeypatch.setattr(auth, "make_client", factory)
    result = legacy_client_from(active)
    assert result.session is active.session
    assert result.headers["Authorization"] == "Bearer test-only"
    factory.assert_called_once_with(api_version=1)


def reply(id="decision1", decision="Accept (Poster)", **overrides):
    return {"id": id, "forum": "a", "readers": ["everyone"],
            "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Decision"],
            "content": {"decision": {"value": decision}, "comment": {"value": "DO NOT SAVE"}},
            **overrides}


def test_decisions_only_public_official_and_same_forum():
    a = note("a", title="A")
    a.details = {"directReplies": [reply(), reply(id="private", readers=["~Person1"]),
        reply(id="other", forum="b"), reply(id="unofficial", invitations=["OtherVenue/-/Decision"]),
        reply(id="review", invitations=["ICLR.cc/2025/Conference/Submission1/-/Official_Review"]),
        reply(id="hiddenfield", content={"decision": {"value": "Reject", "readers": ["~Person1"]}})]}
    result = public_decisions(a, "ICLR.cc/2025/Conference")
    assert len(result) == 1 and result[0]["decision"] == "Accept (Poster)"
    assert "DO NOT SAVE" not in json.dumps(result)


def test_legacy_decision_format():
    a = note("a", title="A")
    a.details = {"directReplies": [reply(invitations=None,
        invitation="ICLR.cc/2022/Conference/Paper1/-/Decision", content={"decision": "Reject"})]}
    assert public_decisions(a, "ICLR.cc/2022/Conference")[0]["decision"] == "Reject"


@pytest.mark.parametrize("values,status", [(["Reject"], "resolved"),
    (["Reject", "Accept (Poster)"], "conflict"), ([], "not_public_or_absent")])
def test_collect_decision_coverage(tmp_path, values, status):
    a = note("a", title="A", abstract="Text")
    a.details = {"directReplies": [reply(id=str(i), decision=v) for i,v in enumerate(values)]}
    client = client_with_pages(([a], 1), [])
    summary = collect_cohort(client, "iclr", 2025, tmp_path, pause=0, with_decisions=True)
    assert client.get_notes.call_args_list[0].kwargs["details"] == "directReplies"
    row = json.loads((tmp_path / "iclr-2025.jsonl").read_text())
    assert row["decision_status"] == status
    assert (row["decision"] is not None) == (status == "resolved")
    assert summary["records_with_official_decision"] == (status == "resolved")
    assert "DO NOT SAVE" not in json.dumps(row)
