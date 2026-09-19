"""Configurable conference-paper screening, review exports and trend figures.

Run ``python -m paper_atlas.conference_survey --help``. Automatic matches are
candidates, never claims about the number of new algorithms in a field.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import sys

from paper_atlas.settings import DATA_DIR
from paper_atlas.conference_corpus import VENUES, collect, dump_json, normalized_text, sha256

DEFAULT_TOPICS = Path(__file__).with_name("configs") / "conference_rl_topics.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def load_topics(path):
    config = json.loads(Path(path).read_text())
    if config.get("version") != 1 or not isinstance(config.get("topics"), list) or not config["topics"]:
        raise ValueError("Expected version 1 and a nonempty topics list")
    groups = config.get("groups", {})
    ids = set()

    def compile_terms(terms):
        if not isinstance(terms, list) or not terms or not all(isinstance(t, str) and t.strip() for t in terms):
            raise ValueError("Each group must contain nonempty literal phrases")
        # Literal phrases, Unicode-normalized and case-insensitive; no unsafe regex input.
        return [(t, re.compile(r"(?<!\w)" + re.escape(normalized_text(t)) + r"(?!\w)")) for t in terms]

    result = []
    for item in config["topics"]:
        topic = dict(item)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", topic["id"]) or topic["id"] in ids:
            raise ValueError("Topic IDs must be unique lowercase identifiers")
        ids.add(topic["id"])
        if not topic.get("label") or not topic.get("all_of"):
            raise ValueError("Each topic needs a label and nonempty all_of")
        topic.setdefault("description", topic["label"])
        topic["required"] = []
        for group in topic["all_of"]:
            if isinstance(group, str):
                if group not in groups:
                    raise ValueError(f"Unknown keyword group: {group}")
                terms, name = groups[group], group
            elif isinstance(group, dict) and set(group) == {"any"}:
                terms, name = group["any"], "inline"
            else:
                raise ValueError("all_of entries must be group names or {any: [...]} objects")
            topic["required"].append((name, compile_terms(terms)))
        exclusions = topic.get("none_of", [])
        topic["excluded"] = compile_terms(exclusions) if exclusions else []
        result.append(topic)
    return config, result


def match_topic(paper, topic):
    text = normalized_text(paper["title"] + "\n" + paper["abstract"])
    evidence = []
    for name, phrases in topic["required"]:
        matched = [(term, regex.search(text)) for term, regex in phrases]
        matched = [(term, match) for term, match in matched if match]
        if not matched:
            return None
        term, match = matched[0]
        evidence.append({"group": name, "terms": [t for t, _ in matched],
                         "excerpt": text[max(0, match.start() - 65):match.end() + 100]})
    if any(pattern.search(text) for _, pattern in topic["excluded"]):
        return None
    return evidence


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_reviews(path, papers, topics):
    if path is None:
        return {}
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"paper_id", "topic_id", "decision"}.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV requires paper_id, topic_id, decision columns")
        for row in reader:
            key = (row["paper_id"], row["topic_id"])
            decision = row["decision"].strip().lower()
            if key[0] not in papers or key[1] not in topics:
                raise ValueError(f"Unknown or out-of-scope paper/topic in review: {key}")
            if decision not in ("", "include", "exclude"):
                raise ValueError(f"Decision must be blank, include or exclude: {key}")
            if key in result:
                raise ValueError(f"Duplicate review row: {key}")
            result[key] = {"decision": decision, "reason": row.get("reason", "")}
    return result


def semantic_ranking(papers, topics, output, *, model_name=EMBEDDING_MODEL, revision=EMBEDDING_REVISION, top_k=30):
    """Optional CPU retrieval: normalized, token-weighted chunk pooling.

    No cosine cutoff is treated as a relevance verdict. The top-k pairs join
    the review queue, but not the keyword-candidate counts.
    """
    import numpy as np
    if top_k < 1:
        return {}
    signature = {"model": model_name, "revision": revision, "pooling": "token-weighted-chunks-v1",
                 "papers": [(p["id"], p["title"], p["abstract"]) for p in papers],
                 "queries": [(t["id"], t.get("query", t["description"])) for t in topics]}
    key = sha256(json.dumps(signature, sort_keys=True).encode())
    cache = Path(output) / f"semantic_scores_{key}.npz"
    if cache.exists():
        scores = np.load(cache, allow_pickle=False)["scores"]
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ValueError("Semantic retrieval needs the optional sentence-transformers package; keyword analysis does not.") from error
        model = SentenceTransformer(model_name, revision=revision, device="cpu", trust_remote_code=False,
                                    model_kwargs={"use_safetensors": True})
        capacity = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
        chunks, owners, weights = [], [], []
        for i, paper in enumerate(papers):
            ids = model.tokenizer.encode(paper["title"] + ". " + paper["abstract"], add_special_tokens=False, truncation=False)
            for start in range(0, len(ids), capacity):
                part = ids[start:start + capacity]
                chunks.append(model.tokenizer.decode(part, skip_special_tokens=True))
                owners.append(i)
                weights.append(len(part))
        # Re-tokenizing decoded chunks can lengthen them: fail rather than silently truncate.
        if any(len(model.tokenizer.encode(chunk, add_special_tokens=False, truncation=False)) > capacity for chunk in chunks):
            raise ValueError("Tokenizer round-trip changed chunk lengths; use a larger-context embedding model")
        embeddings = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        pooled = np.zeros((len(papers), embeddings.shape[1]), dtype=np.float32)
        np.add.at(pooled, owners, embeddings * np.asarray(weights, dtype=np.float32)[:, None])
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        queries = [t.get("query", t["description"]) for t in topics]
        if model_name.startswith("BAAI/bge-"):
            queries = ["Represent this sentence for searching relevant passages: " + q for q in queries]
        if any(len(model.tokenizer.encode(q, truncation=False)) > model.max_seq_length for q in queries):
            raise ValueError("An embedding query exceeds the model context")
        query_vectors = model.encode(queries, normalize_embeddings=True)
        scores = pooled @ query_vectors.T
        np.savez_compressed(cache, scores=scores)
    result = {}
    rows = []
    for j, topic in enumerate(topics):
        for rank, i in enumerate(np.argsort(-scores[:, j], kind="stable")[:top_k], 1):
            paper, score = papers[i], float(scores[i, j])
            result[(paper["id"], topic["id"])] = score
            rows.append({"topic_id": topic["id"], "rank": rank, "cosine_similarity": score,
                         "paper_id": paper["id"], "year": paper["year"], "venue": paper["venue"],
                         "title": paper["title"], "url": paper["url"]})
    write_csv(Path(output) / "semantic_ranking.csv", rows, ["topic_id", "rank", "cosine_similarity", "paper_id", "year", "venue", "title", "url"])
    return result


def draw_counts(output, counts, topics, years, venues, coverage, *, reviewed=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    ncols = min(2, len(topics))
    nrows = math.ceil(len(topics) / ncols)
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols + 0.5, 3.5 * nrows), squeeze=False)
    colors = {"iclr": "#4b7c69", "icml": "#93ae9b", "neurips": "#bcc8c0"}
    incomplete = {y for y in years if any(coverage[(v, y)]["status"] != "complete" for v in venues)}
    values = {(r["topic_id"], r["year"], r["venue"]): r for r in counts}
    for ax, topic in zip(axes.flat, topics):
        bottom = np.zeros(len(years))
        for venue in venues:
            metric = "confirmed_included" if reviewed else "keyword_candidates"
            heights = np.array([values[(topic["id"], y, venue)][metric] or 0 for y in years])
            bars = ax.bar(range(len(years)), heights, bottom=bottom, width=.68,
                          color=colors[venue], edgecolor="white", linewidth=.6)
            for bar, y in zip(bars, years):
                if y in incomplete:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#536258")
            bottom += heights
        for i, value in enumerate(bottom):
            available = any(coverage[(v, years[i])]["status"] in ("complete", "partial") for v in venues)
            ax.text(i, value + max(0.06 * max(bottom), .12), str(int(value)) if available else "NA", ha="center", fontsize=12)
        ax.set_ylim(0, max(1, max(bottom) * 1.3))
        ax.set_xticks(range(len(years)), [str(y)[2:] + ("*" if y in incomplete else "") for y in years])
        ax.set_xlabel("Conference year (20xx)")
        title = topic["label"].replace(" language", "\nlanguage").replace(" images", "\nimages")
        ax.set_title(title, fontsize=16)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=".88", linewidth=.8)
    for ax in axes.flat[len(topics):]:
        ax.set_visible(False)
    fig.supylabel("Confirmed paper matches" if reviewed else "Keyword candidate papers", fontsize=16)
    fig.legend(handles=[Patch(facecolor=colors[v], label=VENUES[v]) for v in venues],
               loc="upper center", ncol=len(venues), frameon=False, fontsize=13, bbox_to_anchor=(.54, 1))
    footer = ("Human-confirmed includes; unreviewed papers are not negatives." if reviewed else
              "Title + abstract screening; unverified, overlapping categories.")
    if incomplete:
        footer += "\n* Incomplete coverage: totals include available sources only."
    fig.text(.52, .01, footer, ha="center", fontsize=10)
    fig.tight_layout(rect=(.03, .065, 1, .935), h_pad=1.1)
    for extension in ("png", "svg", "pdf"):
        name = "reviewed_trends" if reviewed else "topic_trends"
        fig.savefig(Path(output) / f"{name}.{extension}", dpi=220)
    plt.close(fig)


def analyze(root, output, config_path, *, tracks=("research",), reviews_path=None, semantic_top_k=0,
            embedding_model=EMBEDDING_MODEL, embedding_revision=EMBEDDING_REVISION):
    root, output = Path(root), Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    raw = (root / "papers.jsonl").read_bytes()
    if sha256(raw) != manifest["corpus_sha256"]:
        raise ValueError("Corpus checksum does not match manifest")
    corpus = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(corpus) != manifest["paper_count"] or len({p["id"] for p in corpus}) != len(corpus):
        raise ValueError("Duplicate paper IDs or corpus size mismatch")
    papers = [p for p in corpus if p["track"] in tracks]
    if not papers:
        raise ValueError("No publications in selected tracks")
    config, topics = load_topics(config_path)
    by_id = {p["id"]: p for p in papers}
    reviews = read_reviews(reviews_path, by_id, {t["id"] for t in topics})
    if reviews_path is not None and Path(reviews_path).resolve() in {
            (output / "matches.csv").resolve(), (output / "review_template.csv").resolve()}:
        raise ValueError("Keep the working review CSV separate from generated matches/review_template, or choose another --output")
    output.mkdir(parents=True, exist_ok=True)
    semantic = semantic_ranking(papers, topics, output, model_name=embedding_model,
                                revision=embedding_revision, top_k=semantic_top_k) if semantic_top_k else {}
    pairs = {}
    for paper in papers:
        for topic in topics:
            key = (paper["id"], topic["id"])
            evidence = match_topic(paper, topic)
            if evidence is not None or key in semantic or key in reviews:
                review = reviews.get(key, {})
                pairs[key] = {"paper_id": key[0], "topic_id": key[1], "year": paper["year"], "venue": paper["venue"],
                              "title": paper["title"], "url": paper["url"], "keyword_candidate": evidence is not None,
                              "semantic_score": semantic.get(key, ""), "decision": review.get("decision", ""),
                              "reason": review.get("reason", ""), "evidence": json.dumps(evidence or [], ensure_ascii=False)}
    rows = sorted(pairs.values(), key=lambda r: (r["topic_id"], r["year"], r["venue"], r["title"]))
    fields = ["paper_id", "topic_id", "year", "venue", "title", "url", "keyword_candidate", "semantic_score", "decision", "reason", "evidence"]
    write_csv(output / "matches.csv", rows, fields)
    # A separate template never overwrites a user's working reviews.csv.
    write_csv(output / "review_template.csv", rows, fields)
    coverage = {(r["venue"], r["year"]): dict(r) for r in manifest["coverage"]}
    for (venue, year), record in coverage.items():
        if record["status"] not in ("failed", "unavailable"):
            scoped_issues = [i for i in record.get("issues", []) if i.get("track", "unknown") in (*tracks, "unknown")]
            missing = any(not p["abstract"] for p in papers if p["venue"] == venue and p["year"] == year)
            record["status"] = "partial" if scoped_issues or missing else "complete"
    counts = []
    for topic in topics:
        for year in manifest["years"]:
            for venue in manifest["venues"]:
                selected = [r for r in rows if r["topic_id"] == topic["id"] and r["year"] == year and r["venue"] == venue]
                available = coverage[(venue, year)]["status"] in ("complete", "partial")
                screened = sum(p["year"] == year and p["venue"] == venue for p in papers)
                candidates = sum(r["keyword_candidate"] for r in selected)
                counts.append({"topic_id": topic["id"], "year": year, "venue": venue,
                               "source_status": coverage[(venue, year)]["status"],
                               "papers_screened": screened if available else None,
                               "keyword_candidates": candidates if available else None,
                               "keyword_candidates_per_1000": 1000 * candidates / screened if screened else None,
                               "confirmed_included": sum(r["decision"] == "include" for r in selected) if available else None,
                               "reviewed_excluded": sum(r["decision"] == "exclude" for r in selected) if available else None,
                               "pending_review": sum(not r["decision"] for r in selected) if available else None})
    write_csv(output / "counts.csv", counts, list(counts[0]))
    coverage_rows = [{"venue": v, "year": y, "status": r["status"], "all_track_papers": r["paper_count"],
                      "selected_track_papers": sum(p["venue"] == v and p["year"] == y for p in papers),
                      "missing_abstracts": sum(not p["abstract"] for p in papers if p["venue"] == v and p["year"] == y),
                      "reason": r.get("reason", ""), "issues": json.dumps(r.get("issues", []))}
                     for (v, y), r in coverage.items()]
    write_csv(output / "coverage.csv", coverage_rows, list(coverage_rows[0]))
    draw_counts(output, counts, topics, manifest["years"], manifest["venues"], coverage)
    if reviews_path is not None:
        draw_counts(output, counts, topics, manifest["years"], manifest["venues"], coverage, reviewed=True)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "corpus_sha256": manifest["corpus_sha256"],
               "topics_sha256": sha256(Path(config_path).read_bytes()), "tracks": list(tracks),
               "screened_papers": len(papers), "missing_abstracts": sum(not p["abstract"] for p in papers),
               "complete_sources": sum(r["status"] == "complete" for r in coverage.values()), "total_sources": len(coverage),
               "candidate_totals": dict(Counter(r["topic_id"] for r in rows if r["keyword_candidate"])),
               "review_file": str(reviews_path) if reviews_path else None,
               "review_sha256": sha256(Path(reviews_path).read_bytes()) if reviews_path else None,
               "semantic": {"top_k": semantic_top_k, "model": embedding_model, "revision": embedding_revision} if semantic_top_k else None,
               "interpretation": "Keyword candidate publications, not verified relevance or number of new algorithms. Conference year, not arXiv year. Multi-label; counts are not additive. Unavailable sources are not zero research activity."}
    dump_json(output / "summary.json", summary)
    dump_json(output / "topics.used.json", config)
    dump_json(output / "manifest.used.json", manifest)
    sections = []
    for topic in topics:
        entries = []
        for row in rows:
            if row["topic_id"] != topic["id"]:
                continue
            p = by_id[row["paper_id"]]
            entries.append(f'<details><summary>{row["year"]} {VENUES[row["venue"]]} — '
                           f'{html.escape(p["title"])} [{row["decision"] or "unreviewed"}]</summary>'
                           f'<p><a href="{html.escape(p["url"], quote=True)}">Paper</a> · {html.escape(p["id"])}</p>'
                           f'<p>{html.escape(p["abstract"])}</p><pre>{html.escape(row["evidence"])}</pre></details>')
        sections.append(f'<h2>{html.escape(topic["label"])}</h2><p>{html.escape(topic["description"])}</p>' + "\n".join(entries))
    body = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Conference topic survey</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#243a30}'
            'img{max-width:100%}summary{cursor:pointer;padding:8px}details{border-bottom:1px solid #ddd}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:6px;text-align:left}</style>'
            '<h1>Conference topic survey</h1><p>' + html.escape(summary["interpretation"]) + '</p>'
            '<p>Search field: titles and abstracts, not full papers. Main research track by default; scope is configurable.</p>'
            '<img src="topic_trends.svg" alt="Topic candidate counts by conference year">'
            + ('<h2>Reviewed includes (not an exhaustive count)</h2><img src="reviewed_trends.svg" alt="Human-confirmed includes">' if reviews_path is not None else '')
            + '<h2>Source coverage</h2><table>'
            '<tr><th>Venue</th><th>Year</th><th>Status</th><th>Screened</th></tr>'
            + ''.join(f'<tr><td>{VENUES[r["venue"]]}</td><td>{r["year"]}</td><td>{r["status"]}</td><td>{r["selected_track_papers"]}</td></tr>' for r in coverage_rows)
            + '</table>' + '\n'.join(sections) + '</html>')
    (output / "index.html").write_text(body)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {(output / 'index.html').resolve()}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "analyze", "run"))
    parser.add_argument("--root", type=Path, default=Path(DATA_DIR) / "literature/conference_survey")
    parser.add_argument("--output", type=Path, help="Analysis directory (default: ROOT/results)")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(date.today().year - 4, date.today().year + 1)))
    parser.add_argument("--venues", choices=tuple(VENUES), nargs="+", default=list(VENUES))
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--tracks", nargs="+", default=["research"],
                        choices=("research", "position", "datasets-and-benchmarks", "journal-to-conference", "reproducibility-challenge", "blog", "unknown"))
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--offline", action="store_true", help="Use raw cache only; no HTTP requests")
    parser.add_argument("--refresh", action="store_true", help="Refresh URL pointers; preserve content-addressed previous bodies")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--allow-incomplete", action="store_true", help="Exit successfully even with missing/partial sources")
    parser.add_argument("--semantic-top-k", type=int, default=0, help="Optional CPU embedding retrieval added to review queue, NOT counted automatically")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=EMBEDDING_REVISION, help="Pin a model commit, not a moving main branch")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.semantic_top_k < 0:
        parser.error("workers must be 1..8 and semantic-top-k nonnegative")
    if args.offline and args.refresh:
        parser.error("--offline and --refresh cannot be combined")
    if args.offline and args.semantic_top_k:
        parser.error("--offline guarantees no downloads; run cached-corpus semantic analysis without --offline")
    if args.semantic_top_k and not re.fullmatch(r"[0-9a-f]{40}", args.embedding_revision):
        parser.error("Use a 40-character embedding model commit for reproducibility")
    try:
        if args.command in ("collect", "run"):
            manifest = collect(args.root, sorted(set(args.years)), list(dict.fromkeys(args.venues)),
                               offline=args.offline, refresh=args.refresh, workers=args.workers)
        else:
            manifest = json.loads((args.root / "manifest.json").read_text())
        if args.command in ("analyze", "run"):
            summary = analyze(args.root, args.output or args.root / "results", args.topics, tracks=args.tracks,
                    reviews_path=args.reviews, semantic_top_k=args.semantic_top_k,
                    embedding_model=args.embedding_model, embedding_revision=args.embedding_revision)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    incomplete = (summary["complete_sources"] != summary["total_sources"] if args.command in ("analyze", "run")
                  else any(r["status"] != "complete" for r in manifest["coverage"]))
    if incomplete and not args.allow_incomplete:
        print("Incomplete source coverage: artifacts retained. Inspect manifest; use --allow-incomplete to acknowledge.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
