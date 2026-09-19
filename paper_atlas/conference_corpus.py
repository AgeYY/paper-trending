"""Auditable title/abstract collection from official conference event catalogs.

No OpenReview credentials, PDFs, GPU, or paper-reader database are required.
The raw event feed is retained, including records outside the analysis scope.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

VENUES = {"iclr": "ICLR", "icml": "ICML", "neurips": "NeurIPS"}
SCHEMA_VERSION = 1


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text.translate(str.maketrans("‐‑–—", "----"))).strip().lower()


def safe_link(origin: str, value: str) -> str:
    url = urllib.parse.urljoin(origin, value)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid publication URL: {value!r}")
    return url


class JsonCache:
    """Content-addressed bodies plus URL manifests; cache reuse is the default."""

    def __init__(self, root: Path, *, offline=False, refresh=False):
        if offline and refresh:
            raise ValueError("--offline and --refresh cannot be combined")
        self.root, self.offline, self.refresh = Path(root), offline, refresh
        self.used = {}

    def get(self, url: str):
        pointer = self.root / "urls" / (sha256(url.encode()) + ".json")
        if pointer.exists() and not self.refresh:
            meta = json.loads(pointer.read_text())
            if meta.get("http_status") == 404:
                self.used[url] = meta
                raise urllib.error.HTTPError(url, 404, "Cached HTTP 404", None, None)
            raw = (self.root / "bodies" / (meta["sha256"] + ".json")).read_bytes()
            if sha256(raw) != meta["sha256"]:
                raise ValueError(f"Cached body checksum mismatch: {url}")
        else:
            if self.offline:
                raise ValueError(f"Not in offline cache: {url}")
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "conference-survey/1.0", "Accept": "application/json"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        raw = response.read(64 * 1024 * 1024 + 1)
                        headers = dict(response.headers)
                    if len(raw) > 64 * 1024 * 1024:
                        raise ValueError("Metadata response exceeds 64 MiB")
                    json.loads(raw)  # Never cache a challenge/HTML/error as metadata.
                    break
                except urllib.error.HTTPError as error:
                    if error.code == 404:
                        meta = {"url": url, "http_status": 404, "fetched_at": datetime.now(timezone.utc).isoformat()}
                        dump_json(pointer, meta)
                        self.used[url] = meta
                    if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                time.sleep(2 ** attempt)
            meta = {"url": url, "sha256": sha256(raw), "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "etag": headers.get("ETag"), "last_modified": headers.get("Last-Modified")}
            body = self.root / "bodies" / (meta["sha256"] + ".json")
            body.parent.mkdir(parents=True, exist_ok=True)
            if not body.exists():
                body.write_bytes(raw)
            dump_json(pointer, meta)
        self.used[url] = meta
        return json.loads(raw)


def track_of(row: dict, venue: str, year: int) -> str:
    source = (row.get("sourceurl") or "").removeprefix("cdmx-").removesuffix("-cdmx")
    prefix = f"https://openreview.net/group?id={VENUES[venue]}.cc/{year}/"
    if source == prefix + "Conference":
        # ICML 2024's position papers share the research submission group.
        if venue == "icml" and (row.get("name") or "").lower().startswith("position:"):
            return "position"
        return "research"
    if venue == "icml" and year == 2022 and source == "https://cmt3.research.microsoft.com/api/odata/icml2022":
        return "research"
    if source in (prefix + "Datasets_and_Benchmarks_Track", prefix + "Track/Datasets_and_Benchmarks"):
        return "datasets-and-benchmarks"
    if source == prefix + "Position_Paper_Track":
        return "position"
    if source == prefix + "BlogPosts":
        return "blog"
    if source.startswith("https://openreview.net/group?id=ML_Reproducibility_Challenge/") or source == f"{year}_rescience_journal_track":
        return "reproducibility-challenge"
    if (source.startswith(("https://api.github.com/repos/JmlrOrg/", "https://projecteuclid.org/journals/annals-of-statistics/",
                           "https://imstat.org/journals-and-publications/annals-of-statistics/"))
            or source in (f"TMLR-{year}", f"ANN-STATS-{year}", f"{year}-TMLR-J2C-Spreadsheet",
                          f"{year}_jmlr_journal_track", "https://openreview.net/group?id=TMLR")):
        return "journal-to-conference"
    return "unknown"


def normalize_catalog(rows: list, abstracts: dict, venue: str, year: int):
    """Keep accepted publications, deduplicating alternate presentations by title.

    Unknown tracks and non-accepted authored events are surfaced, not silently
    treated as research. A source with such records is marked partial.
    """
    papers, issues, auxiliary, aliases = {}, [], [], {}
    for row in sorted(rows, key=lambda r: r.get("eventtype") != "Poster"):
        if row.get("eventtype") not in ("Poster", "Oral", "Spotlight"):
            issues.append({"event_id": row.get("id"), "reason": "unrecognized event type"})
            continue
        title = (row.get("name") or "").strip()
        title_key = normalized_text(title)
        if not row.get("sourceurl") and (not row.get("authors") or row.get("children_ids")):
            auxiliary.append(str(row.get("id")))
            continue
        link = row.get("paper_url") or row.get("paper_pdf_url") or ""
        parsed = urllib.parse.urlparse(link)
        review_id = urllib.parse.parse_qs(parsed.query).get("id", [None])[0] if parsed.hostname == "openreview.net" else None
        identity = f"openreview:{review_id}" if review_id else None
        # Submission IDs in older catalogs and negated IDs for oral records.
        if (not identity and track_of(row, venue, year) == "research"
                and isinstance(row.get("sourceid"), int) and row["sourceid"]):
            identity = f"submission:{track_of(row, venue, year)}:{abs(row['sourceid'])}"
        identity = identity or "title:" + sha256(title_key.encode())[:20]
        existing = identity if identity in papers else aliases.get(title_key)
        # Alternate oral records sometimes omit the accepted-decision metadata.
        if existing is not None:
            previous = papers[existing]
            new_track = track_of(row, venue, year)
            if new_track not in (previous["track"], "unknown"):
                raise ValueError(f"Conflicting tracks for duplicate title: {title}")
            previous["event_ids"].append(str(row["id"]))
            aliases[title_key] = existing
            continue
        track = track_of(row, venue, year)
        if not title or (not str(row.get("decision", "")).lower().startswith("accept")
                         and track not in ("journal-to-conference", "reproducibility-challenge")):
            issues.append({"event_id": row.get("id"), "title": title, "track": track, "reason": "missing title or accepted decision"})
            continue
        if track == "unknown":
            issues.append({"event_id": row.get("id"), "title": title, "reason": "unknown track", "source": row.get("sourceurl")})
        abstract = abstracts.get(str(row["id"])) or row.get("abstract") or ""
        if not isinstance(abstract, str):
            raise ValueError(f"Invalid abstract for event {row['id']}")
        origin = f"https://{venue}.cc"
        paper_url = row.get("paper_url") or row.get("paper_pdf_url") or ""
        if not paper_url:
            for medium in row.get("eventmedia", []):
                candidate = medium.get("uri") or ""
                if ("openreview.net/" in candidate or medium.get("name", "").lower() in ("paper", "paper pdf", "full paper")):
                    paper_url = candidate
                    break
        page_url = row.get("virtualsite_url") or f"/virtual/{year}/poster/{row['id']}"
        papers[identity] = {
            "id": f"{venue}:{year}:" + identity,
            "venue": venue, "year": year, "title": title, "abstract": abstract.strip(),
            "authors": [a.get("fullname", "") for a in row.get("authors", [])],
            "track": track, "event_ids": [str(row["id"])], "source_group": row.get("sourceurl"),
            "url": safe_link(origin, paper_url or page_url),
            "conference_url": safe_link(origin, page_url),
        }
        aliases[title_key] = identity
    result = sorted(papers.values(), key=lambda p: p["id"])
    return result, {"issues": issues, "auxiliary_event_ids": auxiliary,
                    "tracks": dict(Counter(p["track"] for p in result)),
                    "duplicate_presentations": sum(len(p["event_ids"]) - 1 for p in result),
                    "missing_abstracts": sum(not p["abstract"] for p in result)}


def fetch_source(venue: str, year: int, cache: JsonCache):
    url = f"https://{venue}.cc/static/virtual/data/{venue}-{year}-orals-posters.json"
    rows, seen, declared = [], set(), None
    while url:
        if url in seen or len(seen) >= 100:
            raise ValueError("Repeated/excessive pagination")
        seen.add(url)
        page = cache.get(url)
        if not isinstance(page, dict) or type(page.get("count")) is not int or not isinstance(page.get("results"), list):
            raise ValueError("Catalog does not contain a declared count and results list")
        declared = page["count"] if declared is None else declared
        if page["count"] != declared:
            raise ValueError("Catalog changed during pagination")
        rows.extend(page["results"])
        next_url = page.get("next")
        url = urllib.parse.urljoin(url, next_url) if next_url else None
        if url and urllib.parse.urlparse(url).netloc != f"{venue}.cc":
            raise ValueError("Unexpected pagination host")
    if len(rows) != declared or len({str(r["id"]) for r in rows}) != len(rows):
        raise ValueError("Declared/observed/unique event counts disagree")
    if not rows:
        return [], {"status": "unavailable", "reason": "Official catalog is empty; not a zero-paper year", "raw_events": 0}
    abstracts = {}
    if any(not r.get("abstract") for r in rows if r.get("eventtype") == "Poster"):
        abstracts_url = f"https://{venue}.cc/static/virtual/data/{venue}-{year}-abstracts.json"
        try:
            abstracts = cache.get(abstracts_url)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        if not isinstance(abstracts, dict):
            raise ValueError("Invalid abstract map")
    papers, audit = normalize_catalog(rows, abstracts, venue, year)
    status = "complete" if papers and not audit["issues"] and not audit["missing_abstracts"] else "partial"
    return papers, {"status": status, "raw_events": len(rows), "declared_events": declared, "pages": len(seen), **audit}


def collect(root: Path, years: list[int], venues: list[str], *, offline=False, refresh=False, workers=3):
    """Fetch each conference independently; failures remain explicit in coverage."""
    root = Path(root)

    def one(pair):
        venue, year = pair
        cache = JsonCache(root / "cache", offline=offline, refresh=refresh)
        try:
            papers, audit = fetch_source(venue, year, cache)
        except (ValueError, KeyError, TypeError, OSError) as error:
            papers, audit = [], {"status": "failed", "reason": str(error)}
        record = {"venue": venue, "year": year, "paper_count": len(papers), **audit, "sources": cache.used}
        print(f"{VENUES[venue]} {year}: {record['status']}, {len(papers)} publications", flush=True)
        return papers, record

    pairs = [(v, y) for y in years for v in venues]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, pairs))
    papers = sorted([p for group, _ in results for p in group], key=lambda p: p["id"])
    root.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in papers)
    (root / "papers.jsonl").write_text(raw)
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "years": years, "venues": venues, "paper_count": len(papers), "corpus_sha256": sha256(raw.encode()),
                "scope": "Accepted conference event catalog publications, not workshops or full PDFs. Tracks retained separately.",
                "coverage": [audit for _, audit in results]}
    dump_json(root / "manifest.json", manifest)
    return manifest
