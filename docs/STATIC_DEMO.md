# Browser-only demo and publication

The GitHub Pages demo serves static files. Searches run in a Web Worker on the
visitor's device, never on the maintainer's computer. GitHub handles file delivery,
not search execution. Visitors download public data shards for selected years and
conferences. No API keys, AI calls, accounts, query logs or search history are used.

The demo retains keyword search, conference/year/track filters, publication
counts, same-pool ICLR acceptance comparisons, paper links/abstracts, pagination,
sorting, CSV/JSON exports and SVG/PNG charts. PDF export, AI interpretation,
semantic search and data collection remain in the full local Python application.
Keyword membership matches Python; ranking is simpler (title hits, then year).

## Build locally

After collecting a corpus as described in the README:

```bash
python -m paper_atlas.static_export \
  --corpus data/literature/conference_survey \
  --output data/site-preview \
  --repository https://github.com/AgeYY/paper-trending
python -m http.server 8768 --bind 127.0.0.1 --directory data/site-preview
```

Open http://localhost:8768. The output directory must be new or empty to avoid
mixing snapshots. Direct `file://` opening is unsupported because module workers
and dataset fetching require HTTP. Modern Chrome, Firefox, Safari or Edge with
DecompressionStream and Web Workers support is required.

Only explicitly selected public fields leave the corpus. Raw API responses,
credentials, login tokens, local paths, query caches, usage counters and model
weights are never copied. Submission cohorts pass the existing Python audits
before export. Compressed shards have content hashes; the worker checks their
SHA256 and record count before searching. A download failure aborts the search,
not a misleading partial count. The snapshot date and coverage are always shown.

## Publish from the public repository

The public repository contains source and an audited export in `site/`.
`.github/workflows/pages.yml` deploys **only `site/`**, not the Python source or
repository root. Choose **GitHub Actions** in Settings → Pages. This workflow
requires no OpenAI or OpenReview credentials. Pages hosting logs are controlled
by GitHub; the application itself records no queries.

To update the dataset, run the collector locally, export to a fresh directory,
audit its contents and test the browser results against Python, then replace the
public `site/` snapshot in a reviewed commit. No automatic collection or data
refresh is scheduled. Do not copy the entire local `data/` directory.

The code-only change workflow is separate: update `paper_atlas/static_assets/`,
then copy those reviewed assets into `site/` alongside the unchanged manifest and
data shards. Do not silently regenerate a different corpus.

## Resource limits

Downloads and searching use the visitor's bandwidth and memory. Data is loaded
on demand and processed sequentially to bound decompression concurrency; it can
still be substantial for all years. Public shards are cached in worker memory
for the current page and may be cached by HTTP. No service worker, localStorage,
IndexedDB, analytics or browser query history is installed by this application.
Generated CSV/JSON files are downloaded only when requested and can contain the
user's query. Treat those user-owned exports accordingly.

Published counts describe the snapshot, not a live feed or verified semantic
census. Repository code is MIT; third-party data is not relicensed. See [DATA.md](DATA.md).
