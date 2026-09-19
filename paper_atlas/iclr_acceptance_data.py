"""Audit official OpenReview downloads for explicitly labelled public-pool rates.

This does not assert that the public API pool equals the organizer's reported
submission population. No Paper Copilot or publication-catalog decisions are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re

from paper_atlas.conference_corpus import dump_json, sha256

KIND = "official_public_snapshot"
POPULATION = (
    "All records in the exhausted official ICLR public submission query, including "
    "withdrawals and desk rejections. An official decision takes precedence over "
    "a later workflow withdrawal. This public pool is not the organizer's reported cohort."
)


def normalize_record(row, year):
    """Use official decisions first, explicit workflow outcomes only if absent."""
    prefix = f"ICLR.cc/{year}/Conference/"
    evidence = row.get("decision_evidence", [])
    if row.get("decision_status") == "conflict":
        raise ValueError("Conflicting official decisions.")
    if evidence:
        for decision in evidence:
            if not decision.get("id") or not any(
                invitation.startswith(prefix) and invitation.endswith("/-/Decision")
                for invitation in decision.get("invitations", [])
            ):
                raise ValueError("Decision lacks official ICLR invitation evidence.")
        labels = {d["decision"] for d in evidence}
        if len(labels) != 1 or row.get("decision") not in labels or row.get("decision_status") != "resolved":
            raise ValueError("Conflicting or unresolved official decision evidence.")
        label = row["decision"].strip()
        if re.fullmatch(r"Accept(?:\s*\([^\n]+\)|:\s*[^\n]+)?", label, re.I):
            status = "accepted"
        elif label.lower() == "reject":
            status = "rejected"
        else:
            raise ValueError(f"Unknown official decision: {label}")
        origin = "official_decision"
    else:
        if row.get("decision") or row.get("decision_status") != "not_public_or_absent":
            raise ValueError("Decision has no official evidence.")
        labels = " ".join(str(row.get(k) or "") for k in ("venue", "venueid", "invitation"))
        labels = labels.lower().replace("_", " ")
        withdrawal = "withdrawn submission" in labels
        desk = "desk rejected submission" in labels
        if withdrawal == desk:
            raise ValueError("Missing decision without an unambiguous explicit withdrawal/desk-rejection status.")
        status = "withdrawn" if withdrawal else "desk_rejected"
        origin = "official_workflow_status"
    result = {k: row.get(k) for k in ("id", "title", "abstract", "source", "decision", "decision_evidence", "decision_status", "venue", "venueid", "invitation")}
    result.update(status=status, status_origin=origin)
    return result


def validate_public_snapshot(data, baseline):
    """Validate the public query, never substitute organizer totals as denominator."""
    if data.get("cohort_kind") != KIND or data.get("usable") is not True:
        raise ValueError(data.get("reason") or "Public submission query is incomplete.")
    if (data.get("venue"), data.get("year"), data.get("track")) != ("iclr", baseline["year"], "research"):
        raise ValueError("Public snapshot has a different venue, year or track.")
    if data.get("complete") is not False or data.get("public_query_exhausted") is not True:
        raise ValueError("Public-query coverage must not be confused with official-cohort completeness.")
    if data.get("population") != POPULATION or not data.get("retrieved_at") or not data.get("coverage_audit"):
        raise ValueError("Missing public-pool definition or audit.")
    if data.get("source") != f'https://openreview.net/group?id=ICLR.cc/{data["year"]}/Conference':
        raise ValueError("Expected official OpenReview source.")
    if not re.fullmatch(r"[0-9a-f]{64}", data.get("input_sha256", "")):
        raise ValueError("Missing download hash.")
    records = data.get("records", [])
    if not records or len(records) != data.get("submitted") or len(records) != data.get("api_visible_count"):
        raise ValueError("Public snapshot count does not match the exhausted query.")
    seen = set()
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("Missing or duplicate submission ID.")
        seen.add(row["id"])
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("title", "abstract")):
            raise ValueError("Incomplete title/abstract; topic classification would be biased.")
        normalized = normalize_record(row, data["year"])
        if any(row.get(k) != normalized[k] for k in ("status", "status_origin")):
            raise ValueError("Stored status does not match official decision evidence.")
    counts = dict(Counter(r["status"] for r in records))
    if counts != data.get("status_counts") or counts.get("accepted", 0) != data.get("accepted"):
        raise ValueError("Public snapshot outcome counts do not reconcile.")
    return records


def audit_download(folder, year):
    folder = Path(folder)
    summary = json.loads((folder / f"iclr-{year}.summary.json").read_text())
    raw = (folder / f"iclr-{year}.jsonl").read_bytes()
    if sha256(raw) != summary.get("sha256"):
        raise ValueError(f"ICLR {year}: download hash mismatch.")
    # splitlines() treats literal Unicode paragraph separators inside abstracts
    # as record boundaries. JSONL uses actual newline characters only.
    records = [json.loads(line) for line in io.StringIO(raw.decode("utf-8")) if line.strip()]
    if len(records) != summary.get("records_saved") or len({r["id"] for r in records}) != len(records):
        raise ValueError(f"ICLR {year}: count mismatch or duplicate IDs.")
    data = {"venue": "iclr", "year": year, "track": "research", "cohort_kind": KIND,
            "complete": False, "usable": False, "public_query_exhausted": summary.get("public_query_exhausted", False),
            "source": f"https://openreview.net/group?id=ICLR.cc/{year}/Conference",
            "retrieved_at": summary["retrieved_at"], "input_sha256": sha256(raw),
            "input_file": str((folder / f"iclr-{year}.jsonl").resolve()),
            "api_visible_count": summary.get("api_visible_count_at_start"), "downloaded_records": len(records),
            "population": POPULATION, "download_audit": summary}
    if summary.get("error") or summary.get("public_query_exhausted") is not True:
        data["reason"] = (f'Official decision download incomplete: {len(records):,} / '
                          f'{summary.get("api_visible_count_at_start", 0):,} records '
                          f'({summary.get("error", "query not exhausted")}). No rates computed from a partial pool.')
        return data
    if (summary.get("venue"), summary.get("year")) != ("iclr", year):
        raise ValueError("Download summary venue/year mismatch.")
    if (summary.get("decisions_requested") is not True or summary.get("api_count_matches_unique_returned") is not True
            or summary.get("excluded_nonpublic_or_deleted") != 0 or summary.get("conflicting_decisions") != 0):
        raise ValueError("Official decisions or public-query coverage have not been verified.")
    normalized = [normalize_record(row, year) for row in records]
    official = sum(r["status_origin"] == "official_decision" for r in normalized)
    if official != summary.get("records_with_official_decision") or len(records)-official != summary.get("records_without_official_decision"):
        raise ValueError("Decision counts disagree with download audit.")
    counts = dict(Counter(r["status"] for r in normalized))
    data.update(usable=True, records=normalized, submitted=len(records), accepted=counts.get("accepted", 0), status_counts=counts,
                coverage_audit="Exhausted official public query; API count, SHA256, unique IDs, complete text and every decision/workflow status checked. Organizer totals remain unreconciled.")
    validate_public_snapshot(data, {"year": year})
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2022, 2027)))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    # Audit every input before changing any installed snapshot.
    snapshots = [audit_download(args.download, year) for year in args.years]
    targets = [args.corpus / "acceptance" / f'iclr-public-{d["year"]}.json' for d in snapshots]
    for target, data in zip(targets, snapshots):
        if target.exists() and json.loads(target.read_text()) != data and not args.replace:
            parser.error(f"Refusing to replace {target}; use --replace intentionally.")
    for target, data in zip(targets, snapshots):
        dump_json(target, data)
        print(json.dumps({"path": str(target), **{k: data.get(k) for k in ("year", "usable", "accepted", "submitted", "reason")}}))


if __name__ == "__main__":
    main()
