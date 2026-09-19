"""Build a secret-free, browser-only snapshot; never copy a working directory."""
from __future__ import annotations

import argparse
import gzip
import html
import itertools
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from paper_atlas.conference_corpus import sha256, VENUES
from paper_atlas.conference_search import SearchEngine

ASSETS = Path(__file__).with_name("static_assets")
TRACKS = ["research", "position", "datasets-and-benchmarks"]


def public_url(value):
    """Export external links, never file URLs or credentials embedded in URLs."""
    try:
        parsed = urlsplit(value or "")
        if parsed.scheme in ("https", "http") and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return ""


def build(corpus, output, repository):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory; do not overwrite a release.")
    if not repository.startswith("https://github.com/") or not public_url(repository):
        raise ValueError("Use a public GitHub repository URL.")
    engine = SearchEngine(corpus, output / "unused-state")
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir()
    for file in ASSETS.iterdir():
        if file.is_file():
            shutil.copyfile(file, output / file.name)
    (output / ".nojekyll").touch()
    parts = urlsplit(repository).path.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError("Repository URL must identify an owner and repository.")
    site_url = f"https://{parts[0].lower()}.github.io/{parts[1]}/"
    index = output / "index.html"
    index.write_text(index.read_text().replace("<!-- canonical -->", f'<link rel="canonical" href="{html.escape(site_url, quote=True)}">'))
    manifest = {"format": 1, "repository": repository, "site_url": site_url, "snapshot": engine.manifest["created_at"],
                "corpus_sha256": engine.manifest["corpus_sha256"], "years": engine.manifest["years"],
                "venues": VENUES, "tracks": TRACKS, "shards": [], "coverage": {}, "acceptance": {}}

    def shard(name, value, **info):
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        zipped = gzip.compress(raw, mtime=0)
        filename = f"data/{name}-{sha256(zipped)[:12]}.json.gz"
        (output / filename).write_bytes(zipped)
        return {"file": filename, "bytes": len(zipped), "sha256": sha256(zipped), **info}

    for year in manifest["years"]:
        for venue in VENUES:
            rows = []
            for i, paper in enumerate(engine.papers):
                if paper["year"] != year or paper["venue"] != venue:
                    continue
                # Explicit allowlist: no raw source blobs, user data or local provenance paths.
                rows.append({"id": paper["id"], "title": paper["title"], "abstract": paper["abstract"],
                             "authors": paper.get("authors", []), "track": paper["track"],
                             "url": public_url(paper.get("url"))})
            if rows:
                manifest["shards"].append(shard(f"papers-{venue}-{year}", rows, kind="papers", year=year, venue=venue, count=len(rows)))
    # Coverage depends on selected tracks; preserve the Python engine's exact rules.
    for n in range(1, len(TRACKS) + 1):
        for tracks in itertools.combinations(TRACKS, n):
            manifest["coverage"][",".join(sorted(tracks))] = [
                {k: r[k] for k in ("year", "venue", "status", "papers", "missing_abstracts")}
                for r in engine.coverage(manifest["years"], list(VENUES), list(tracks))]
    for year in manifest["years"]:
        key = year, "iclr"
        base = engine.acceptance.baselines.get(key, {})
        info = {"available": False, "reason": "No verified public submission pool in this snapshot.",
                "official_accepted": base.get("accepted"), "official_submitted": base.get("submitted"),
                "official_rate": 100 * base["accepted"] / base["submitted"] if base.get("submitted") else base.get("reported_rate"),
                "official_source": public_url(base.get("source"))}
        if key in engine.acceptance.pools:
            pool = engine.acceptance.pools[key]
            provenance = engine.acceptance.provenance[key]
            descriptor = shard(f"submissions-iclr-{year}", pool, kind="submissions", year=year, venue="iclr", count=len(pool))
            manifest["shards"].append(descriptor)
            info.update(available=True, reason="", submitted=len(pool), accepted=sum(s == "accepted" for _, s in pool),
                        population=provenance["population"], source=public_url(provenance["source"]),
                        retrieved_at=provenance["retrieved_at"], sha256=provenance["sha256"],
                        public_snapshot=key in engine.acceptance.public_keys)
        manifest["acceptance"][str(year)] = info
    manifest["download_bytes"] = sum(s["bytes"] for s in manifest["shards"])
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args(argv)
    result = build(args.corpus, args.output, args.repository)
    print(json.dumps({"shards": len(result["shards"]), "download_bytes": result["download_bytes"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
