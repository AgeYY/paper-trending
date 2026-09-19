"""Loopback-only conference topic explorer. Not a public production server."""
from __future__ import annotations

import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlparse

from paper_atlas.settings import DATA_DIR
from paper_atlas.conference_corpus import VENUES
from paper_atlas.conference_search import SearchEngine
from paper_atlas.ai_interpreter import AIInterpreter, MODEL, VERSION, SCHEMA, load_api_key
from paper_atlas.weekly_limit import WeeklyLimitReached, WEEKLY_LIMIT, TIMEZONE

ASSETS = Path(__file__).with_name("conference_web_assets")
PLOT_LOCK = threading.Lock()


def chart_bytes(result, extension):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    with PLOT_LOCK, plt.rc_context({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                                   "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
                                   "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(4., 3.5))
        years, venues = result["config"]["years"], result["config"]["venues"]
        bottom = np.zeros(len(years))
        palette = {"iclr": "#214b40", "icml": "#72947d", "neurips": "#b9cdb3"}
        metric = "candidates"
        partial = {r["year"] for r in result["counts"] if r["status"] != "complete"}
        for venue in venues:
            values = [next(r[metric] or 0 for r in result["counts"] if r["year"] == y and r["venue"] == venue) for y in years]
            bars = ax.bar(range(len(years)), values, bottom=bottom, color=palette[venue], label=VENUES[venue])
            for bar, year in zip(bars, years):
                if year in partial:
                    bar.set_hatch("///")
            bottom += values
        for i, year in enumerate(years):
            if all(r["status"] in ("failed", "unavailable") for r in result["counts"] if r["year"] == year):
                ax.text(i, 0, "NA", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(years)), [str(y)[2:] + ("*" if y in partial else "") for y in years])
        ax.set_xlabel("Conference year (20xx)")
        ax.set_ylabel("Hybrid candidates" if result["config"].get("retrieval") else "Automatic paper matches")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=".88", linewidth=.8)
        ax.set_ylim(0, max(1, max(bottom) * 1.15))
        fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=10)
        footer = ("Hybrid screening; not verified relevance." if result["config"].get("retrieval")
                  else "Keyword screening; not verified relevance.")
        if partial:
            footer += "\n* Incomplete conference coverage."
        fig.text(.52, .015, footer, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .08, 1, .9), pad=.6)
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=220, metadata={"Creator": "Local conference explorer"} if extension == "pdf" else None)
        plt.close(fig)
        return buffer.getvalue()


def csv_bytes(rows, fields):
    def safe(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: safe(row.get(k, "")) for k in fields} for row in rows)
    return buffer.getvalue().encode("utf-8-sig")


def acceptance_chart_bytes(result, extension):
    """ICLR-only paired bars; publication counts still include all conferences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    years = result["config"]["years"]
    venues = [v for v in result["config"]["venues"] if v == "iclr"]
    palette = {"iclr": "#214b40", "icml": "#72947d", "neurips": "#8b7049"}
    rows = {(r["year"], r["venue"]): r for r in result["acceptance"]["rows"]}
    with PLOT_LOCK, plt.rc_context({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                                   "axes.linewidth": 1.8, "xtick.major.width": 1.8, "ytick.major.width": 1.8,
                                   "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(4., 3.5))
        width = .8 / (2*max(1, len(venues)))
        for i, year in enumerate(years):
            for j, venue in enumerate(venues):
                row = rows[year, venue]
                for k, metric in enumerate(("topic_rate", "conference_rate")):
                    x = i - .4 + width*(2*j+k+.5)
                    val = row[metric]
                    if val is None:
                        ax.text(x, 2, "NA", ha="center", va="bottom", rotation=90, fontsize=7, color=palette[venue])
                    else:
                        ax.bar(x, val, width*.84, facecolor=palette[venue] if k == 0 else "white",
                               edgecolor=palette[venue], linewidth=1.4)
                        if val == 0:
                            ax.text(x, 2, "0", ha="center", fontsize=8)
        ax.set(ylim=(0, 100), ylabel="Acceptance (%)", xlabel="ICLR year")
        if not venues:
            ax.text(.5, .5, "Select ICLR to view acceptance", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11)
        ax.set_xticks(range(len(years)), years)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=".88", linewidth=.8)
        legend = [Patch(facecolor=palette["iclr"], label="Topic"), Patch(facecolor="white", edgecolor=palette["iclr"], label="Overall")]
        fig.legend(handles=legend, loc="upper center", ncol=2, fontsize=16, frameon=False)
        public = any(r.get("baseline_kind") == "official_public_snapshot" for r in rows.values())
        caption = ("Official public pool; includes withdrawals/desks.\nNot organizer rates. NA is unavailable, not zero." if public else
                   "ICLR main track. NA is unavailable, not zero.\nSee exported data for denominators and sources.")
        if result["config"].get("retrieval"):
            caption += "\nHybrid topic screening; not verified relevance."
        fig.text(.5, .012, caption, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .13 if result["config"].get("retrieval") else .1, 1, .89), pad=.6)
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=220)
        plt.close(fig)
        return buffer.getvalue()


def automatic_result(result):
    """Expose automatic counts only; leave historical review data untouched."""
    result.pop("review_counts", None)
    result.pop("review_filter", None)
    for row in result["counts"]:
        row.pop("confirmed", None)
    for paper in result["papers"]:
        paper.pop("decision", None)
    return result


def create_server(engine, port=8765, interpreter=None):
    token = secrets.token_urlsafe(32)
    search_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Do not log requested URLs, query strings, result IDs or user input.
            pass

        def respond(self, status, body, content_type="application/json; charset=utf-8", filename=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            try:
                return urlparse("http://" + self.headers.get("Host", "")).hostname in ("localhost", "127.0.0.1", "::1")
            except ValueError:
                return False

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, {"error": "Localhost access only."})
            try:
                route = urlparse(self.path)
                params = parse_qs(route.query)
                if route.path == "/favicon.ico":
                    return self.respond(204, b"", "image/x-icon")
                if route.path in ("/", "/algorithm", "/algorithm/", "/app.js", "/algorithm.js", "/style.css", "/compact.css"):
                    name = "index.html" if route.path == "/" else "algorithm.html" if route.path.rstrip("/") == "/algorithm" else route.path[1:]
                    mime = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}[Path(name).suffix]
                    return self.respond(200, (ASSETS / name).read_bytes(), mime)
                if route.path.startswith("/report/"):
                    name = route.path.removeprefix("/report/") or "index.html"
                    mime = {"index.html": "text/html", "topic_trends.svg": "image/svg+xml", "topic_trends.png": "image/png", "topic_trends.pdf": "application/pdf"}
                    if name not in mime:
                        raise KeyError("Report file not found.")
                    return self.respond(200, (engine.root / "results" / name).read_bytes(), mime[name])
                if route.path == "/api/meta":
                    return self.respond(200, {**engine.metadata(), "csrf_token": token,
                        "ai": {"enabled": interpreter is not None, "model": MODEL,
                               "weekly_quota": interpreter.quota.status() if interpreter else None}})
                if route.path == "/api/algorithm":
                    return self.respond(200, {"metadata": engine.metadata(), "catalog_publications": len(engine.papers),
                        "ai": {"enabled": interpreter is not None, "model": MODEL, "prompt_version": VERSION,
                            "fields": SCHEMA["required"], "max_groups": SCHEMA["properties"]["groups"]["maxItems"],
                            "max_terms": SCHEMA["properties"]["groups"]["items"]["properties"]["terms"]["maxItems"],
                            "max_exclusions": SCHEMA["properties"]["exclusions"]["maxItems"],
                            "weekly_limit": WEEKLY_LIMIT, "timezone": TIMEZONE},
                        "example": {"kind": "illustrative", "groups": [
                            {"label": "Reinforcement learning", "terms": ["reinforcement learning", "policy optimization", "PPO", "GRPO"]},
                            {"label": "Flow matching", "terms": ["flow matching", "rectified flow"]},
                            {"label": "Image generation", "terms": ["image generation", "text-to-image", "image synthesis"]}]},
                        "acceptance": engine.acceptance.metadata()})
                if route.path.startswith("/api/search/"):
                    result = engine.page(route.path.removeprefix("/api/search/"), int(params.get("page", ["1"])[0]),
                                         params.get("sort", ["relevance"])[0])
                    return self.respond(200, automatic_result(result))
                if route.path.startswith("/api/export/"):
                    search_id, extension = route.path.removeprefix("/api/export/").rsplit(".", 1)
                    result = automatic_result(engine.page(search_id))
                    if extension in ("svg", "png", "pdf"):
                        chart = params.get("chart", ["count"])[0]
                        if chart not in ("count", "acceptance"):
                            raise ValueError("Unknown chart type.")
                        data = acceptance_chart_bytes(result, extension) if chart == "acceptance" else chart_bytes(result, extension)
                        return self.respond(200, data, {"svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf"}[extension], f"conference-{chart}-{search_id}.{extension}")
                    saved = engine.load(search_id)
                    if extension == "csv":
                        rows = [{**engine.papers[engine.by_id[m["paper_id"]]],
                                 "score": m["score"], "evidence": json.dumps(m["evidence"], ensure_ascii=False)} for m in saved["matches"]]
                        return self.respond(200, csv_bytes(rows, ["id", "year", "venue", "track", "title", "url", "abstract", "score", "evidence"]), "text/csv; charset=utf-8", f"papers-{search_id}.csv")
                    if extension == "counts":
                        return self.respond(200, csv_bytes(result["counts"], ["year", "venue", "status", "papers", "candidates", "per_1000"]), "text/csv; charset=utf-8", f"counts-{search_id}.csv")
                    if extension == "acceptance":
                        return self.respond(200, csv_bytes(result["acceptance"]["rows"], ["year", "venue", "topic_status", "topic_accepted", "topic_submitted", "topic_rate", "conference_accepted", "conference_submitted", "conference_rate", "baseline_kind", "delta_pp", "population", "topic_reason", "baseline_reason", "baseline_source", "additional_sources", "topic_source", "official_accepted", "official_submitted", "official_conference_rate", "official_source", "official_population"]), "text/csv; charset=utf-8", f"acceptance-{search_id}.csv")
                    if extension == "json":
                        return self.respond(200, {**saved, "counts": result["counts"], "acceptance": result["acceptance"]}, filename=f"search-{search_id}.json")
                raise KeyError("Not found.")
            except KeyError as error:
                self.respond(404, {"error": str(error)})
            except FileNotFoundError:
                self.respond(404, {"error": "This file is not available. Generate the original report with the survey CLI if needed."})
            except (ValueError, TypeError) as error:
                self.respond(400, {"error": str(error)})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Local-Token") != token:
                return self.respond(403, {"error": "Invalid local session token. Refresh this page."})
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond(403, {"error": "Cross-origin requests are not accepted."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768 or self.headers.get_content_type() != "application/json":
                    return self.respond(400, {"error": "Expected a JSON request smaller than 32 KiB."})
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                if self.path in ("/api/search", "/api/interpret"):
                    allowed = ({"prompt"} if self.path == "/api/interpret" else
                               {"prompt", "years", "venues", "tracks", "search_mode", "interpretation_id", "groups", "supporting_groups", "exclusions"})
                    if set(body) - allowed:
                        raise ValueError("Unsupported search fields.")
                    if not search_lock.acquire(blocking=False):
                        return self.respond(429, {"error": "Another search is running. Please try again shortly."})
                    try:
                        if self.path == "/api/interpret":
                            if interpreter is None:
                                raise ValueError("AI is disabled. Use local keyword search or start the server with --enable-ai.")
                            return self.respond(200, interpreter.interpret(body.get("prompt", "")))
                        interpretation = None
                        if body.get("search_mode") == "keyword" and any(k in body for k in ("interpretation_id", "groups", "supporting_groups", "exclusions")):
                            raise ValueError("Keyword mode accepts comma-separated phrases, not AI or custom criteria.")
                        if "interpretation_id" in body:
                            if interpreter is None or not isinstance(body["interpretation_id"], str):
                                raise ValueError("AI interpretation is unavailable.")
                            interpretation = interpreter.load(body.pop("interpretation_id"), body.get("prompt", ""))
                            if interpretation["needs_clarification"]:
                                raise ValueError("Clarify your description and interpret it again before searching.")
                            body.setdefault("groups", interpretation["groups"])
                            body.setdefault("supporting_groups", interpretation.get("supporting_groups", []))
                            body.setdefault("exclusions", interpretation["exclusions"])
                            body["hybrid"] = True
                        elif any(k in body for k in ("groups", "supporting_groups", "exclusions")):
                            raise ValueError("Reviewed criteria need an interpretation_id.")
                        return self.respond(200, automatic_result(engine.search(body, interpretation=interpretation)))
                    finally:
                        search_lock.release()
                raise KeyError("Not found.")
            except WeeklyLimitReached as error:
                self.respond(429, {"error": str(error), "weekly_quota": error.quota})
            except KeyError as error:
                self.respond(404, {"error": str(error)})
            except (ValueError, TypeError) as error:
                self.respond(400, {"error": str(error),
                    "weekly_quota": interpreter.quota.status() if interpreter else None})
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--state", type=Path, help="Legacy compatibility path; search history is not persisted")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--enable-ai", action="store_true", help="Enable OpenAI paragraph interpretation (sends descriptions to OpenAI)")
    parser.add_argument("--enable-semantic", action="store_true", help="Prepare local CPU embeddings (also enabled by --enable-ai)")
    parser.add_argument("--api-key-file", type=Path, help="Private key file; defaults to ~/.config/paper-atlas/openai.env")
    args = parser.parse_args(argv)
    key = None
    if args.enable_ai:
        try:
            key = load_api_key(args.api_key_file)
        except (ValueError, OSError):
            parser.error("Cannot load the private API key. Check the key file format, ownership and permissions (600).")
        if not key:
            parser.error("AI requested but OPENAI_API_KEY is not configured.")
    if args.corpus is None:
        candidates = list((Path(DATA_DIR) / "literature").glob("conference_survey*/manifest.json"))
        if not candidates:
            parser.error("No corpus found. Run python -m paper_atlas.conference_survey collect first.")
        args.corpus = max(candidates, key=lambda p: p.stat().st_mtime).parent
    print(f"Loading and indexing {args.corpus.resolve()} (CPU only)…", flush=True)
    engine = SearchEngine(args.corpus, args.state or args.corpus / "web_app" / "local")
    if args.enable_semantic or args.enable_ai:
        print("Preparing local semantic indexes (CPU only; cached across restarts)…", flush=True)
        try:
            engine.enable_semantics(Path(DATA_DIR) / "semantic")
        except ValueError as error:
            parser.error(str(error))
    interpreter = AIInterpreter(key, engine.state_root / "interpretations.sqlite3",
                               quota_path=Path(DATA_DIR) / "ai_usage.sqlite3") if key else None
    server = create_server(engine, args.port, interpreter=interpreter)
    print("AI interpretation: " + (MODEL + " (descriptions sent to OpenAI)" if interpreter else "disabled; local search only"), flush=True)
    print(f"Local app ready: http://127.0.0.1:{server.server_port} — {len(engine.papers):,} catalog publications", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
