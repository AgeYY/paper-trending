"""Auditable topic acceptance rates; incomplete queries never become rates.

Separate from the accepted-publication corpus. An acceptance snapshot must cover
the SAME submission cohort as its conference baseline, with explicit provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from urllib.parse import urlencode, urlparse
import urllib.error

from paper_atlas.conference_corpus import VENUES, JsonCache, dump_json, sha256

BASELINES = Path(__file__).with_name("configs") / "conference_acceptance.json"
STATUSES = {"accepted", "rejected", "withdrawn", "desk_rejected"}


def url_ok(value):
    return isinstance(value, str) and urlparse(value).scheme in ("http", "https") and bool(urlparse(value).netloc)


def validate_snapshot(data, baseline):
    """Fail closed: completeness is an audited assertion AND checked totals.

Matching totals alone cannot prove coverage. Require an explicit source and
cohort audit, then independently check IDs, text, statuses and both totals.
"""
    if not isinstance(data, dict):
        raise ValueError("Submission snapshot must be an object.")
    if data.get("complete") is not True:
        raise ValueError(data.get("reason") or "Submission coverage has not been verified complete.")
    if not baseline.get("submitted"):
        raise ValueError("No exact conference denominator for cohort reconciliation.")
    if (data.get("venue"), data.get("year"), data.get("track")) != (baseline["venue"], baseline["year"], "research"):
        raise ValueError("Submission snapshot has a different venue, year or track.")
    if data.get("population") != baseline["population"] or data.get("baseline_source") != baseline["source"]:
        raise ValueError("Topic and conference denominators refer to different populations or source versions.")
    if not url_ok(data.get("source")) or not data.get("retrieved_at") or not data.get("coverage_audit"):
        raise ValueError("Submission snapshot lacks source, retrieval date or coverage audit.")
    records = data.get("records", [])
    if not isinstance(records, list) or len(records) != baseline["submitted"]:
        raise ValueError("Submission count does not reconcile with the conference denominator.")
    seen = set()
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("Missing or duplicate submission ID.")
        seen.add(row["id"])
        if row.get("status") not in STATUSES:
            raise ValueError("Unresolved submission decision.")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("title", "abstract")):
            raise ValueError("Submission titles/abstracts are incomplete; topic classification would be biased.")
    if sum(r["status"] == "accepted" for r in records) != baseline["accepted"]:
        raise ValueError("Accepted count does not reconcile with the conference baseline.")
    return records


class AcceptanceStore:
    def __init__(self, root, baselines=BASELINES):
        self.root = Path(root) / "acceptance"
        raw = Path(baselines).read_bytes()
        self.catalog = json.loads(raw)
        self.baselines = {}
        self.pools, self.reasons, self.provenance = {}, {}, {}
        self.documents = {}
        self.embeddings = None
        self.public_keys = set()
        hashes = [sha256(raw)]
        for row in self.catalog["rows"]:
            if row["venue"] != "iclr":
                continue
            key = row["year"], row["venue"]
            if key in self.baselines:
                raise ValueError("Duplicate acceptance baseline.")
            if row["venue"] not in VENUES:
                raise ValueError("Unknown acceptance venue.")
            if "submitted" in row and not (type(row["submitted"]) is int and type(row.get("accepted")) is int and 0 <= row["accepted"] <= row["submitted"] and row["submitted"] > 0):
                raise ValueError("Invalid acceptance baseline counts.")
            if "reported_rate" in row and not (0 <= row["reported_rate"] <= 100):
                raise ValueError("Invalid reported rate.")
            if ("submitted" in row or "reported_rate" in row) and (not url_ok(row.get("source")) or not row.get("population")):
                raise ValueError("Acceptance baseline lacks source/population.")
            self.baselines[key] = row
            path = self.root / f'{row["venue"]}-{row["year"]}.json'
            public_path = self.root / f'iclr-public-{row["year"]}.json'
            if not path.exists() and public_path.exists():
                path = public_path
                self.public_keys.add(key)
            if not path.exists():
                self.reasons[key] = ("Complete submission cohort not loaded. Accepted papers or opt-in public rejections cannot establish a topic denominator."
                                     if row["venue"] in ("icml", "neurips") else "Complete submission snapshot not loaded.")
                continue
            content = path.read_bytes()
            hashes.append(sha256(content))
            try:
                data = json.loads(content)
                if key in self.public_keys:
                    from paper_atlas.iclr_acceptance_data import validate_public_snapshot
                    records = validate_public_snapshot(data, row)
                else:
                    records = validate_snapshot(data, row)
                # Import lazily to share EXACTLY the website's matching rules.
                from paper_atlas.conference_search import phrase_text
                self.pools[key] = [(" " + phrase_text(r["title"] + ". " + r["abstract"]) + " ", r["status"]) for r in records]
                self.documents[key] = [r["title"] + ". " + r["abstract"] for r in records]
                self.provenance[key] = {k: data[k] for k in ("source", "retrieved_at", "coverage_audit")}
                self.provenance[key]["sha256"] = sha256(content)
                self.provenance[key]["population"] = data["population"]
                if key in self.public_keys:
                    self.provenance[key].update(input_sha256=data["input_sha256"],
                                                cohort_kind=data["cohort_kind"],
                                                status_counts=data["status_counts"])
            except (ValueError, KeyError, TypeError) as error:
                self.reasons[key] = str(error)
        self.snapshot_hash = sha256("\n".join(hashes).encode())

    def metadata(self):
        return {"sha256": self.snapshot_hash, "checked_at": self.catalog["checked_at"],
                "verified_topic_cohorts": len(self.pools.keys() - self.public_keys),
                "public_snapshot_cohorts": len(self.pools.keys() & self.public_keys),
                "baseline_cohorts": sum("submitted" in r or "reported_rate" in r for r in self.baselines.values())}

    def compute(self, config):
        # Return fresh values so callers cannot mutate the shared cache.
        return json.loads(self._compute(json.dumps(config, sort_keys=True)))

    @lru_cache(maxsize=128)
    def _compute(self, config_json):
        from paper_atlas.conference_search import phrase_text
        config = json.loads(config_json)
        groups = [[" " + phrase_text(term) + " " for term in g["terms"]] for g in config["groups"]]
        exclusions = [" " + phrase_text(t) + " " for t in config["exclusions"]]
        required = len(groups) if config["mode"] == "all" else math.ceil(len(groups)/2) if config["mode"] == "half" else 1
        rows = []
        for year in config["years"]:
            for venue in (v for v in config["venues"] if v == "iclr"):
                key = year, venue
                base = self.baselines.get(key, {})
                row = {"year": year, "venue": venue, "topic_rate": None, "conference_rate": None,
                       "topic_accepted": None, "topic_submitted": None, "delta_pp": None,
                       "conference_accepted": base.get("accepted"), "conference_submitted": base.get("submitted"),
                       "baseline_source": base.get("source"), "population": base.get("population", ""),
                       "additional_sources": base.get("additional_sources", []),
                       "baseline_kind": "unavailable", "topic_status": "unavailable",
                       "baseline_reason": base.get("reason", "No verified conference baseline."),
                       "topic_reason": self.reasons.get(key, "No complete submission snapshot."), "topic_source": None}
                row.update(official_accepted=base.get("accepted"), official_submitted=base.get("submitted"),
                           official_source=base.get("source"), official_population=base.get("population"),
                           official_conference_rate=(100*base["accepted"]/base["submitted"] if "submitted" in base else base.get("reported_rate")))
                if config["tracks"] != ["research"]:
                    row.update(topic_reason="Acceptance comparison currently supports the main research track only.",
                               baseline_reason="Selected tracks differ from the main-track baseline.",
                               conference_accepted=None, conference_submitted=None)
                    rows.append(row)
                    continue
                if "submitted" in base:
                    row.update(conference_rate=100*base["accepted"]/base["submitted"], baseline_kind="computed", baseline_reason="")
                elif "reported_rate" in base:
                    row.update(conference_rate=base["reported_rate"], baseline_kind="reported_rounded", baseline_reason="Published rounded percentage; exact denominator unavailable.")
                if key in self.public_keys:
                    # Never pair a partial public pool with a published baseline.
                    row.update(conference_rate=None, conference_accepted=None, conference_submitted=None,
                               baseline_kind="unavailable", baseline_reason=self.reasons.get(key, ""),
                               population="Official OpenReview public submission pool; organizer totals shown separately.",
                               baseline_source=None, additional_sources=[])
                if key in self.pools:
                    if key in self.public_keys:
                        statuses = [status for _, status in self.pools[key]]
                        total, accepted = len(statuses), statuses.count("accepted")
                        row.update(conference_rate=100*accepted/total, conference_accepted=accepted,
                                   conference_submitted=total, baseline_kind="official_public_snapshot",
                                   baseline_source=self.provenance[key]["source"],
                                   population=self.provenance[key]["population"])
                    if config.get("retrieval"):
                        from paper_atlas.semantic_search import evaluate_hybrid, SemanticUnavailable
                        if self.embeddings is None:
                            row.update(topic_reason="Hybrid submission index unavailable; no lexical substitution was made.")
                            rows.append(row)
                            continue
                        try:
                            membership, _, _ = evaluate_hybrid([text for text, _ in self.pools[key]],
                                self.documents[key], config, self.embeddings)
                        except SemanticUnavailable as error:
                            row.update(topic_reason=str(error))
                            rows.append(row)
                            continue
                        matches = [status for keep, (_, status) in zip(membership, self.pools[key]) if keep]
                    else:
                        matches = [status for text, status in self.pools[key]
                                   if sum(any(term in text for term in terms) for terms in groups) >= required
                                   and not any(term in text for term in exclusions)]
                    a, n = matches.count("accepted"), len(matches)
                    row.update(topic_accepted=a, topic_submitted=n, topic_source=self.provenance[key],
                               topic_status="complete" if n else "no_matches", topic_reason="" if n else "No matching submissions; 0/0 is undefined.")
                    if n and key in self.public_keys:
                        row["topic_status"] = "public_snapshot"
                    if n:
                        row.update(topic_rate=100*a/n, delta_pp=100*a/n-row["conference_rate"])
                rows.append(row)
        return json.dumps({"rows": rows, "metadata": self.metadata(), "scope": "iclr_research", "venues": ["iclr"] if "iclr" in config["venues"] else [],
                           "matching": config.get("retrieval", {"version": "lexical"}),
                           "definition": "ICLR only. Topic and overall bars use the same submission pool. Official-public snapshots include withdrawals and desk rejections; official decisions take precedence over later workflow withdrawals. Published organizer rates are separate references, not substituted denominators. Partial queries are unavailable. Topic membership uses the search's recorded rules, not verified relevance; semantic similarity is not a probability."})


def collect_probe(root, venue, year):
    """Bounded public API probe. Never retry an access challenge or claim completeness.

    Raw responses remain in the shared content-addressed cache. This command
    records availability; complete normalized exports are a separate audited input.
    """
    target = Path(root)/"acceptance"/f"{venue}-{year}.json"
    if target.exists():
        raise ValueError(f"Refusing to overwrite an existing submission snapshot: {target}")
    host = "https://api.openreview.net" if year < 2024 else "https://api2.openreview.net"
    invitation = f"{VENUES[venue]}.cc/{year}/Conference/-/Submission"
    url = host + "/notes?" + urlencode({"invitation": invitation, "limit": 1})
    record = {"venue": venue, "year": year, "track": "research", "complete": False, "source": url,
              "retrieved_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = JsonCache(Path(root)/"acceptance"/"cache").get(url)
        record.update(public_note_count=response.get("count"), reason="Public API is reachable, but cohort completeness and final decisions have not been reconciled with organizer totals.")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as error:
        record["reason"] = f"Public submission API unavailable ({error}). Topic denominator not verified."
    dump_json(target, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--probe", action="store_true", help="Record public submission endpoint availability; never assert complete coverage")
    parser.add_argument("--import-snapshot", type=Path, help="Validate and install an audited normalized submission JSON")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace an existing snapshot after validation")
    parser.add_argument("--venue", choices=VENUES, default="iclr")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args(argv)
    if args.probe and args.import_snapshot:
        parser.error("Choose either --probe or --import-snapshot.")
    if args.import_snapshot:
        data = json.loads(args.import_snapshot.read_text())
        store = AcceptanceStore(args.corpus)
        key = data.get("year"), data.get("venue")
        if key not in store.baselines:
            parser.error("No baseline configured for this snapshot.")
        validate_snapshot(data, store.baselines[key])
        target = args.corpus/"acceptance"/f"{key[1]}-{key[0]}.json"
        if target.exists() and not args.replace:
            parser.error(f"Snapshot exists: {target}; use --replace only intentionally.")
        dump_json(target, data)
        print(f"Validated and installed {target}; restart the web server to load it.")
    elif args.probe:
        print(json.dumps(collect_probe(args.corpus, args.venue, args.year), indent=2))
    else:
        store = AcceptanceStore(args.corpus)
        print(json.dumps({**store.metadata(), "unavailable": {f"{v}-{y}": r for (y,v),r in store.reasons.items()}}, indent=2))


if __name__ == "__main__":
    main()
