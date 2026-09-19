"""Download public metadata with an already authenticated official client.

No credential storage, private submissions, author profiles, or review texts.
These are audit inputs, NOT verified complete acceptance-rate cohorts.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from paper_atlas.settings import DATA_DIR


VENUES = {"iclr": "ICLR", "neurips": "NeurIPS", "icml": "ICML"}
FIELDS = ("title", "abstract", "keywords", "venue", "venueid", "decision", "primary_area")
LEGACY_COHORTS = {("iclr", 2022), ("iclr", 2023), ("neurips", 2022)}


def legacy_client_from(client):
    """Reuse the in-memory official session on the official legacy API only."""
    from paper_atlas.openreview_auth_check import make_client

    legacy = make_client(api_version=1)
    legacy.headers.update(client.headers)
    legacy.token = client.token
    legacy.session.close()
    legacy.session = client.session
    return legacy


def public_value(content, key):
    value = content.get(key)
    if isinstance(value, dict):
        if "readers" in value and "everyone" not in (value["readers"] or []):
            return None
        return value.get("value")
    return value


def public_record(note):
    # An authenticated reviewer/author may see more than the public. Fail closed.
    if "everyone" not in (getattr(note, "readers", None) or []):
        return None
    if getattr(note, "ddate", None):
        return None
    row = {key: public_value(note.content or {}, key) for key in FIELDS}
    row.update(id=note.id, source=f"https://openreview.net/forum?id={note.id}")
    invitation = getattr(note, "invitation", None)
    if isinstance(invitation, str):
        row["invitation"] = invitation
    return row


def public_decisions(note, venue_id):
    """Keep only public, venue-authored Decision reply metadata, never reviews.

    The official API's directReplies response includes other kinds of replies;
    all non-decision content is discarded rather than written to disk.
    """
    details = getattr(note, "details", None) or {}
    evidence = {}
    for reply in details.get("directReplies", []):
        if "everyone" not in (reply.get("readers") or []) or reply.get("ddate"):
            continue
        forum = reply.get("forum")
        if forum not in {note.id, getattr(note, "forum", None)} or not forum:
            continue
        invitations = reply.get("invitations") or [reply.get("invitation", "")]
        official = [i for i in invitations if isinstance(i, str)
                    and i.startswith(venue_id + "/") and i.endswith("/-/Decision")]
        if not official:
            continue
        decision = public_value(reply.get("content") or {}, "decision")
        if not isinstance(decision, str) or not decision.strip() or not reply.get("id"):
            continue
        evidence[reply["id"]] = {"id": reply["id"], "forum": forum, "decision": decision,
                                  "invitations": official, "cdate": reply.get("cdate"),
                                  "mdate": reply.get("mdate"),
                                  "source": f"https://openreview.net/forum?id={forum}&noteId={reply['id']}"}
    return sorted(evidence.values(), key=lambda r: r["id"])


def iter_pages(client, invitations, summary, page_size, pause, with_decisions=False):
    """Cursor pagination over explicitly named submission collections."""
    summary["api_visible_count_at_start"] = 0
    summary["queries"] = []
    for invitation in invitations:
        query = {"invitation": invitation, "exhausted": False}
        summary["queries"].append(query)
        params = dict(invitation=invitation, limit=page_size, sort="id", with_count=True)
        if with_decisions:
            params["details"] = "directReplies"
        cursors = set()
        for page_index in range(2000):
            result = client.get_notes(**params)
            if page_index == 0:
                notes, count = result
                query["api_visible_count_at_start"] = count
                summary["api_visible_count_at_start"] += count
            else:
                notes = result
            if not notes:
                query["exhausted"] = True
                break
            cursor = notes[-1].id
            if not cursor or cursor in cursors:
                raise ValueError("Pagination stopped advancing")
            cursors.add(cursor)
            yield notes
            params.update(after=cursor, with_count=False)
            time.sleep(pause)
        else:
            raise ValueError("Page safety limit reached")
    summary["public_query_exhausted"] = True


def dump_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def collect_cohort(client, venue, year, root, page_size=500, pause=0.3, with_decisions=False):
    venue_id = f"{VENUES[venue]}.cc/{year}/Conference"
    path = root / f"{venue}-{year}.jsonl"
    summary = {
        "venue": venue, "year": year, "api": "https://api2.openreview.net",
        "venue_id": venue_id, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "file": path.name, "complete": False, "public_query_exhausted": False,
        "reason": "Official submission population and final decisions have not been reconciled.",
        "records_saved": 0, "records_returned": 0, "excluded_nonpublic_or_deleted": 0,
        "duplicates": 0, "missing_title": 0, "missing_abstract": 0,
        "status_labels": {},
        "decisions_requested": with_decisions, "records_with_official_decision": 0,
        "conflicting_decisions": 0, "records_without_official_decision": 0,
    }
    seen, statuses = set(), Counter()
    try:
        group = client.get_group(venue_id)
        submission_name = public_value(group.content or {}, "submission_name")
        if isinstance(submission_name, str) and submission_name:
            invitations = [f"{venue_id}/-/{submission_name}"]
        elif (venue, year) in LEGACY_COHORTS:
            client = legacy_client_from(client)
            summary["api"] = "https://api.openreview.net"
            # Older venues split public anonymized/withdrawn/desk-rejected notes
            # across invitations. Original Submission notes may be nonpublic.
            invitations = [f"{venue_id}/-/{name}" for name in (
                "Submission", "Blind_Submission", "Withdrawn_Submission", "Desk_Rejected_Submission"
            )]
            summary["reason"] += " Legacy original/blind-note identities also require reconciliation."
        else:
            raise ValueError("Venue submission invitation could not be established")
        summary["invitations"] = invitations
        with path.open("x", encoding="utf-8") as stream:
            for notes in iter_pages(client, invitations, summary, page_size, pause, with_decisions):
                for note in notes:
                    summary["records_returned"] += 1
                    if not isinstance(note.id, str) or not note.id:
                        raise ValueError("Missing paper identifier")
                    if note.id in seen:
                        summary["duplicates"] += 1
                        continue
                    seen.add(note.id)
                    record = public_record(note)
                    if record is None:
                        summary["excluded_nonpublic_or_deleted"] += 1
                        continue
                    if with_decisions:
                        evidence = public_decisions(note, venue_id)
                        values = {r["decision"] for r in evidence}
                        record["decision_evidence"] = evidence
                        record["decision"] = next(iter(values)) if len(values) == 1 else None
                        record["decision_status"] = "resolved" if len(values) == 1 else "conflict" if values else "not_public_or_absent"
                        summary["records_with_official_decision"] += len(values) == 1
                        summary["conflicting_decisions"] += len(values) > 1
                        summary["records_without_official_decision"] += not values
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    summary["records_saved"] += 1
                    for key in ("title", "abstract"):
                        if not isinstance(record[key], str) or not record[key].strip():
                            summary[f"missing_{key}"] += 1
                    label = record["decision"] or record["venue"] or record["venueid"] or record.get("invitation") or "unresolved"
                    statuses[str(label)] += 1
                stream.flush()
                print(f"{venue.upper()} {year}: {summary['records_saved']:,} public records saved", flush=True)
        summary["unique_records_returned"] = len(seen)
        summary["api_count_matches_unique_returned"] = len(seen) == summary["api_visible_count_at_start"]
    except (Exception, KeyboardInterrupt) as exc:
        # Whitelist labels only: exceptions can contain credentials or private data.
        payload = exc.args[0] if exc.args and isinstance(exc.args[0], dict) else {}
        name = payload.get("name")
        summary["error"] = name if name in {
            "ChallengeRequiredError", "NotFoundError", "RateLimitError", "AuthenticationError"
        } else "Interrupted" if isinstance(exc, KeyboardInterrupt) else "FetchFailed"
        raise
    finally:
        summary["status_labels"] = dict(statuses)
        if path.exists():
            summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        dump_json(root / f"{venue}-{year}.summary.json", summary)
    return summary


def download_public_metadata(client, output=None, venues=None, years=None, with_decisions=False):
    from paper_atlas.openreview_auth_check import report_error

    root = Path(output) if output else Path(DATA_DIR) / "literature" / (
        "openreview_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    root.mkdir(parents=True, exist_ok=False)
    print(f"Download directory: {root.resolve()}", flush=True)
    print("Only public metadata is saved. Private rejections remain unavailable.", flush=True)
    manifest = {"source": "official OpenReview APIs v2/v1", "complete": False,
                "decisions_requested": with_decisions, "cohorts": []}
    status = 0
    try:
        for year in dict.fromkeys(years or range(2022, 2027)):
            for venue in dict.fromkeys(venues or VENUES):
                try:
                    summary = collect_cohort(client, venue, year, root,
                                             page_size=100 if with_decisions else 500,
                                             with_decisions=with_decisions)
                    manifest["cohorts"].append(summary)
                except Exception as exc:
                    summary_path = root / f"{venue}-{year}.summary.json"
                    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                    manifest["cohorts"].append(summary)
                    status = report_error(exc, f"{venue.upper()} {year}")
                    # A missing/legacy venue can be skipped. Any other failure,
                    # especially authentication/challenge/rate limiting, stops ALL work.
                    if summary.get("error") != "NotFoundError":
                        break
            else:
                continue
            break
    finally:
        dump_json(root / "manifest.json", manifest)
        print(f"Saved audit: {root.resolve() / 'manifest.json'}", flush=True)
        print("Snapshots are NOT automatically imported into the website or labelled complete.")
    return status
