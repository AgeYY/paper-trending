# Paper Trending — local conference topic explorer

An interactive website on top of the existing conference corpus. **Keyword** mode
is selected by default: enter comma-separated phrases and search directly, without
AI calls, synonym expansion or embeddings. All phrases must match the normalized
title/abstract. Up to 16 phrases of 120 characters each are supported; empty comma
segments are ignored. Select **AI** for GPT-5.4 nano description interpretation,
reviewable concepts and hybrid retrieval. The **Search algorithm** page explains
both modes. See the README's key setup and `--enable-ai` instructions.
No GPU or Node runtime is required. If AI is disabled, only AI searches are blocked;
keyword searches remain available. The weekly AI cap stays
enforced without a persistent usage banner. Keyword searches do not consume it.
Install the Python dependencies using the repository's `pyproject.toml`.

## Start and open

From the repository root:

```bash
python -m paper_atlas.conference_web \
  --corpus data/literature/conference_survey_20260916 --port 8765 --enable-ai
```

Equivalent thin entrypoint: `python bin/conference_web.py`.
If `--corpus` is omitted, the app finds the most recently updated
`DATA_DIR/literature/conference_survey*/manifest.json`.

Open `http://localhost:8765/`. In VS Code Remote SSH, forward port **8765** in
the **Ports** panel, then choose **Open in Browser**. Keep the process running;
Ctrl+C stops it. If the old static report server occupies the port, stop that
specific process or choose another port. The old report is no longer linked in
the menu; its previously generated files remain intact.

The listener is **127.0.0.1 only**. There is deliberately no public bind option,
deployment, tunnel publication, or authentication service. External AI requests
are enabled only by the explicit `--enable-ai` option.
This is a personal local preview, **not a production internet-facing server**.
Use private SSH forwarding, not a public VS Code tunnel. Anyone with access to
the same machine's loopback service can access the workspace; the session token
protects against cross-site requests, not other local users.

## Workflow

1. Enter keywords such as `reinforcement learning, language model`, or select
   **AI** and describe the topic. Example buttons use phrases in Keyword mode
   and descriptions in AI mode.
2. Defaults are the current year plus four preceding years, all three
   conferences, and the research track. Expand **Search scope** to change years,
   venues or add position/datasets-and-benchmarks tracks.
3. The centered Paper Atlas brand sits above the compact search area, which leads
   directly to the results charts, without a summary
   statistics row. Keyword mode searches immediately. AI mode first displays
   editable concepts and waits for **Search with these concepts**. On short screens the chart is automatically brought
   fully into view after search. Example-topic buttons collapse after a search.
   Publication and acceptance charts share a row on desktop (at least 1,200 px)
   and stack on narrower screens, with aligned plotting areas on desktop.
4. Expand abstracts and follow paper links. There are no confirmation counters,
   approval buttons, review filters or reviewer-dependent chart modes.
5. Download SVG/PNG/PDF charts, all matching papers as CSV, yearly counts as CSV,
   or the search criteria/provenance as JSON. Exports include
   all matches, not just the visible page. Unavailable source counts are blank.
6. There is no saved history. Download results yourself if needed; current-page
   results remain available for pagination and export while cached in memory.
7. Open **Search algorithm** in the left menu (also available on mobile) for
   collection/deduplication, normalization, prompt parsing, phrase matching,
   counting equations, ranking, a worked example and limitations. The model,
   schema limits, prompt version, weekly-cap configuration and corpus metadata
   come from the live backend. Example groups are explicitly hand-written and
   illustrative, not an actual AI response. Opening the page makes no AI call. Direct URL:
   `http://localhost:8765/algorithm`.
8. The second chart compares topic acceptance with overall **ICLR** acceptance.
   Filled bars are topic rates; outlines are ICLR baselines, grouped by year.
   The publication-count chart still includes ICLR, ICML, and NeurIPS.
   Expand **Counts, denominators & sources** for the exact populations,
   numerator/denominator counts, percentage-point differences and source links.
   The chart has separate SVG/PNG/PDF and data-CSV downloads.

All UI-generated dates explicitly use English (`en-US`, UTC), regardless of the
browser language. Paper titles/abstracts retain their original text.

The chart uses conference publication year, not arXiv submission year. A cached
source absent from the requested years is explicitly unavailable. At this
snapshot NeurIPS 2026 is unavailable; partial years have hatching and a warning.

## What “understands the prompt” means here

AI mode uses **AI interpretation followed by local hybrid retrieval**,
not per-paper LLM classification. Keyword mode bypasses this interpretation and
uses the same all-core-groups lexical membership for publications and submissions:

- GPT-5.4 nano converts the description into essential concepts, alternative
  phrases, optional supporting groups and AI-proposed exclusions using strict structured output. The old
  `interpret_prompt` dictionary parser remains only for legacy APIs and tests.
- The UI shows these concepts for review/editing before searching. Clarification
  requests block search until the description is revised. Strict JSON guarantees
  a structured answer, not correct interpretation of the user's intent.
- Input is case/Unicode normalized; hyphens become token boundaries and simple
  plurals are normalized. Phrase matching is contiguous, not just a bag of words.
- The model is instructed to ignore conversational scaffolding and background
  motivation, and to ask for clarification if the intended logic cannot be
  represented with required groups of alternatives. This is not a guarantee that
  the model always chooses good concepts.
- Every core concept must match by keyword or local embedding similarity.
  Supporting concepts only boost rank. RL for autoregressive LMs uses broad RL
  and language-model cores (including RLHF/RLVR/PPO/GRPO/LLM/model-name alternatives)
  with optional autoregressive evidence. These automatic recall adjustments are
  shown before user review and recorded alongside the raw AI output.
- Semantic expansion uses pinned all-MiniLM-L6-v2 embeddings: each missing core
  needs cosine >= 0.30, and the complete core query needs cosine >= 0.45. Exact
  all-core keyword matches remain eligible regardless of cosine. Thresholds are
  experimental, not calibrated. All qualifying records are counted, never top-k.
  Publications and accepted/rejected/withdrawn/desk-rejected submissions use the
  exact same membership function. Optional evidence never changes a denominator.
- Ranking fuses complete keyword and semantic rankings with reciprocal rank
  fusion (k=60), plus a small optional-evidence boost. Scores and similarities
  are **not calibrated probabilities**. Per-paper evidence identifies semantic
  expansion and supporting-only matches.
- Embeddings use title plus abstract, truncated to 256 wordpieces, on CPU with
  BF16 computation and FP32 cached vectors. Keywords still inspect the full text.
  Cache identity includes text, pinned model revision, precision and preprocessing.
  Missing semantic indexes fail visibly; no silent lexical fallback is allowed.
  The first build may take several minutes; later starts reuse cached vectors.
- Exclusions also inspect the entire title/abstract, so a background mention can
  exclude an otherwise relevant paper. The AI proposes exclusions from the
  description, including explicit exclusions and inferred out-of-scope concepts,
  and explains its choices. There is no hard-coded exclusion list. Normalization
  preserves the AI's exclusions and explanation; an empty list stays empty.
  Review or edit them before searching. The reviewed exclusions apply to both
  keyword and semantic matches in publication counts and topic acceptance rates.
  User edits are respected without rewriting them. Rerun a search to use the new
  interpretation instructions. Download JSON yourself to retain a result.
- An AI-generated alternative can be too broad (e.g. generic LLM terms to retrieve
  AR papers that omit their architecture). Continuous *time* alone does not mean
  continuous *state*. This is a limitation of the automatic count, explained on
  the algorithm page; the app does not ask the user to approve matches.
- Successful interpretations (including clarification requests) are cached by
  description, configured model, prompt version, instructions and schema. The
  original interpretation and effective edited criteria are retained in exports.
  AI/provider errors never silently switch to the legacy dictionary parser.

Therefore the main chart is **candidate counts**, not verified field sizes or
numbers of distinct algorithms. A long/complex prompt can produce restrictive
concepts; try a shorter/rephrased topic rather than taking a zero literally.
Titles and abstracts are screened, not entire PDFs. Broader synonyms improve
recall but can reduce precision. The original command-line survey still offers
optional embedding retrieval; it is not connected to this web app and no
embedding model has been downloaded for the website.

## Corpus and persistence

### Official ICLR decision download

The first public metadata download did not retrieve decision replies. For an
authenticated metadata-and-decision download in one session, run:

```bash
python bin/check_openreview_auth.py \
  --download --with-decisions --venues iclr --years 2022 2023 2024 2025 2026
```

Enter credentials and any MFA verification privately in the terminal. They are
not saved. A new timestamped `DATA_DIR/literature/openreview_*` directory preserves
the earlier metadata download. The official `directReplies` API response is
filtered to public Decision notes; review text and private replies are discarded.
Saved decision evidence includes note IDs and URLs. Missing or conflicting
decisions remain unresolved, not rejected. These downloads do not automatically
enable topic rates: final decisions and the denominator still need reconciliation.
Paper Copilot decisions are not imported.

The app verifies the corpus SHA256 and unique paper IDs before building an
in-memory inverted index. Search requests never fetch conference pages; they use
the snapshot documented on the Search algorithm page. The discovery page has no
introductory banner, corpus-total counter or snapshot date above the search box.
This makes searches fast and reproducible.
To update the catalog, use the existing collector, then restart the web server:

```bash
python -m paper_atlas.conference_survey collect \
  --root data/literature/conference_survey_20260916 \
  --years 2022 2023 2024 2025 2026 --refresh --allow-incomplete
```

Adjust the years when needed; refreshing is an explicit operation, not performed
on every prompt. Inspect the collection manifest for failures before restarting.
For retaining independent historical snapshots, use a different corpus root.

Searches and AI interpretations are never persisted to disk or browser storage.
Each has a bounded, 64-entry in-memory cache for active-page operations; restarting
clears it. No history is displayed or exposed by an API. `--state` is retained
only for command-line compatibility. HTTP request logging is disabled. Search IDs
include a per-process nonce and are not permanent links. Corpus/document embedding
caches and the aggregate weekly counter remain on disk; the counter stores only
week/count pairs, never descriptions. User-requested downloads are the user's own
saved copies. Generated development artifacts belong under ignored `data/`.

Historical review records, if present, are left untouched in the database but
are not used by the web interface, counts or exports. The old review API is
removed. The independent command-line survey's review functionality is unchanged.

API routes: `GET /api/meta`, `GET /api/algorithm`,
`POST /api/search`, `GET /api/search/ID`, and `GET /api/export/ID.EXT`.
Local web searches accept `prompt`, `years`, `venues`, `tracks` and `search_mode`.
`search_mode: "keyword"` parses literal comma-separated phrases and rejects AI or
custom criteria. `search_mode: "ai"` requires an `interpretation_id`. Omitting
the mode preserves the legacy local dictionary API. `POST /api/interpret` accepts a prompt only
and, when explicitly enabled, returns strict-schema AI concepts for review.
AI-assisted searches also accept a cached `interpretation_id` and edited `groups`
and `exclusions`. The server validates these, verifies the description matches,
and retains original and effective criteria only in temporary memory for export.
Legacy query database files are not read or updated, and have not been deleted.
Review fields are omitted from web responses and
exports, and historical review query parameters cannot filter the count or list.
Writes require the per-process token returned by the same-origin metadata route.
The server checks Host/Origin, rejects oversized JSON, escapes rendered text,
serves a fixed asset allowlist, and guards spreadsheet exports against formulas.
Requests cannot execute shell commands, submit a model name, access arbitrary
files, refresh the corpus or publish anything.

## Validation

### Acceptance data and current limitations

As of 2026-09-17, the user's authenticated official OpenReview download supplies
**ICLR 2022–2025 public-pool rates**. These are explicitly labelled descriptive
rates among the downloaded public submissions, **not reconciled organizer
conference rates**. No Paper Copilot decision labels are used. The ICLR 2026
decision query stopped at 15,400 / 19,814 records with `RateLimitError`; both
public-pool bars stay NA. The separate published reference remains available.

| Year | Official accept decisions | Public submissions | Public-pool rate |
| --- | ---: | ---: | ---: |
| 2022 | 1,095 | 3,422 | 32.00% |
| 2023 | 1,575 | 4,955 | 31.79% |
| 2024 | 2,261 | 7,404 | 30.54% |
| 2025 | 3,708 | 11,672 | 31.77% |

The importer checks file hashes, unique IDs, complete titles/abstracts, API
count agreement, exhausted pagination, and outcome evidence for every record.
Official Decision replies take precedence over current venue labels (some papers
were withdrawn after receiving a decision). If a decision is absent, only an
explicit withdrawal or desk-rejection status is accepted; “Submitted to ICLR”
is never inferred to mean rejected. All withdrawals and desk rejections remain
in the public-pool denominator. Unknown/conflicting outcomes fail validation.
Exhausting an API query does not prove it covers every submission ever made.

Both plotted rates and their percentage-point difference use this **same public
pool**. Published organizer totals have slightly different accepted/submitted
counts, so those rates are shown in a separate reference column, not silently
used as the public-pool baseline. A rate may differ due to snapshot timing,
workflow status or population definitions; these discrepancies are not resolved.
Keyword classification itself can miss relevant submissions or include irrelevant
ones. ICML/NeurIPS acceptance is not estimated; their publication counts remain.

Reproduce the offline audit and installation (no login required):

```bash
python -m paper_atlas.iclr_acceptance_data \
  --download data/literature/openreview_20260917T063038752671Z \
  --corpus data/literature/conference_survey_20260916
```

Files are stored at `CORPUS/acceptance/iclr-public-YEAR.json`. Repeating the
command is idempotent; changed existing snapshots require explicit `--replace`.
Partial downloads produce an unavailable marker, never a rate. Original raw
downloads are preserved. Restart the server after importing new data.

`paper_atlas/configs/conference_acceptance.json` contains sourced organizer baselines for
12 of the 15 default conference-years. ICML 2022/2023 have no verified baseline
in this configuration, and NeurIPS 2026 is not finalized. ICML 2025 is the
organizer's rounded 27% rate, not an inferred exact denominator. ICML 2024 uses
the organizer's decision email, subtracting position-paper counts from both
accepted and submitted totals: `(2609 - 75) / (9473 - 286)`. Do not use the
combined-track 27.5% figure as a main-track denominator.

These baselines mean **all topics within the organizer-defined main-track
submission cohort**, not every initiated abstract registration. Cohort policies
vary by conference/year. Topic and overall rates must use the identical cohort.
Mixed/non-research track selections explicitly show unavailable rates.

The accepted-publication corpus, saved searches and review records are untouched.
The original, stricter **organizer-reconciled cohort** import also remains
available and takes precedence over a public snapshot for that year. Its
submission snapshots are separate files at
`CORPUS/acceptance/{venue}-{year}.json`, loaded at startup. A complete export has:

```json
{
  "venue": "iclr",
  "year": 2025,
  "track": "research",
  "complete": true,
  "population": "EXACT population string from the baseline configuration",
  "baseline_source": "EXACT source URL from the baseline configuration",
  "source": "https://example.org/complete-submission-export",
  "retrieved_at": "2026-09-17T00:00:00Z",
  "coverage_audit": "Explain how this export covers the entire reported cohort, including how withdrawals and desk rejections were handled.",
  "records": [
    {"id": "unique-submission-id", "title": "Paper title", "abstract": "Full abstract", "status": "accepted"}
  ]
}
```

This is a schema illustration, **not an installable dataset**. Statuses are
`accepted`, `rejected`, `withdrawn`, `desk_rejected`. Unknown decisions, missing
text, duplicate IDs, different populations, or discrepancies with **either**
official submitted/accepted total disable the topic rate. The coverage audit is
an explicit curator assertion; matching counts alone is insufficient evidence.
Both topic numerator and denominator come from the same submission pool, using
the same core membership rules, exclusions and semantic thresholds as search.
There is no sampling, no extrapolation from accepted papers, and no substitution
of accepted/(accepted+public rejections). Zero matches yields NA (0/0); matched
submissions with no acceptances yield a valid 0%.

Validate and install an audited export:

```bash
python -m paper_atlas.conference_acceptance \
  --corpus data/literature/conference_survey_20260916 \
  --import-snapshot /absolute/path/to/verified-submissions.json
```

An existing snapshot is preserved unless `--replace` is supplied. Restart the
web server after installing. `--probe --venue iclr --year 2025` records public
endpoint availability only; **it is not a full submission collector**. No flag
automatically asserts that a public API response is complete. Running the module
without these flags reports coverage without network access.

The acceptance-data hash is independent of the publication-corpus hash. Loading
a temporary result computes acceptance against the currently loaded snapshot.
Archive the JSON export to preserve an exact acceptance result; it includes the
acceptance hash, sources and counts. Small cohorts can be noisy, and topic rates
are descriptive candidate-based summaries, not causal or significance tests.

### Test commands

```bash
python -m pytest -q \
  tests/test_conference_acceptance.py tests/test_conference_web.py tests/test_conference_survey.py \
  tests/test_rl_literature_figure.py
```

An optional browser test uses Playwright. Start a **separate test server**
on port 8766 to avoid replacing the user's active in-memory results:

```bash
python -m paper_atlas.conference_web \
  --corpus data/literature/conference_survey_20260916 \
  --state data/literature/conference_survey_20260916/web_app/browser_test \
  --port 8766
```

In another terminal, with Playwright available to Node:

```bash
CONFERENCE_APP_URL=http://127.0.0.1:8766 node tests/conference_web_browser.cjs
```

`PLAYWRIGHT_MODULE` can point to an existing Playwright installation; Node and
Playwright are testing tools only. `CONFERENCE_APP_PREVIEW` changes the screenshot
output directory. The tests exercise typing/clicking, custom queries, pagination,
downloads, empty states, absence of history and creator credit. They assert that the full chart
is visible immediately after search at 1366×768, 1440×900 and 390×844, all removed
controls are absent, and dates stay English in a `zh-CN` browser. They also open
the algorithm page and verify its live AI configuration, illustrative example,
weekly-limit description and formulas.
Scientific chart exports follow the same compact, uncluttered
style as the command-line survey.
