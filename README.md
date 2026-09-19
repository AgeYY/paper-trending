# Paper Trending

A local website for exploring research topics across **ICLR, ICML and NeurIPS**.
Describe a topic, inspect matching papers, and compare yearly publication counts
with ICLR topic-versus-overall acceptance rates.

**[Try the lightweight live demo →](https://ageyy.github.io/paper-trending/)**

> The demo runs in your browser: keyword search, paper-count charts, ICLR
> acceptance comparisons, and downloads. It downloads a public metadata snapshot
> to your device; large selections may take longer to load. AI-assisted and
> semantic search require running the full Python app locally with your own
> OpenAI API key. The demo has no accounts or saved search history.

Source code: [AgeYY/paper-trending](https://github.com/AgeYY/paper-trending).
The software is MIT-licensed; third-party paper data retains its original rights.
See [data provenance](docs/DATA.md) and [the browser-demo guide](docs/STATIC_DEMO.md).

## Current features

- Core keyword concepts with synonym expansion, optional supporting evidence,
  and local CPU text embeddings for hybrid search over titles and abstracts.
- Optional GPT-5.4 nano paragraph interpretation with strict JSON output,
  editable concepts, temporary memory-only caching and exportable query provenance.
- Accepted-publication counts across three conferences, with coverage warnings.
- ICLR acceptance comparisons using official OpenReview decisions and the same
  public submission pool for topic and overall rates. Published organizer rates
  remain separate references when denominators differ.
- Side-by-side charts on desktop, stacked charts on mobile; SVG/PNG/PDF and CSV
  exports, paper links, no saved search history, and a documented search algorithm.
- Reproducible corpus collection, interactive OpenReview download tools, input
  hashes, and validation that prevents partial decision queries producing rates.

The full Python app is **loopback-only**, not a public backend. The separate
GitHub Pages demo is static and does not contact a Python server.
New website searches default to **Keyword** mode: comma-separated phrases, all
required, with no AI or synonym expansion. Optional **AI** mode sends the topic
description to OpenAI, never the paper corpus or search history. Hybrid matches are candidates, not a
verified semantic census of a research field.

## Install

Python 3.10 or newer is required. No GPU is needed for the website.

```bash
git clone https://github.com/AgeYY/paper-trending.git
cd paper-trending
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For AI/hybrid search also install `python -m pip install -e '.[embeddings]'`.
For tests add `.[dev]`; authenticated data collection additionally needs
`.[openreview]`. On Windows activate with `.venv\Scripts\Activate.ps1` instead.

The `openreview` extra is needed only for authenticated downloads. The `embeddings`
extra is needed for hybrid website search and the survey CLI's semantic ranking.
The web encoder runs only on CPU; no paper text leaves the machine. On first
startup it downloads a pinned MiniLM checkpoint and builds cached indexes for
publications and verified submission pools under `data/semantic/`. This one-time
build can take several minutes. Downloaded models/data are not included in Git.

## Run

First collect public conference metadata (no login required):

```bash
python -m paper_atlas.conference_survey collect \
  --years 2022 2023 2024 2025 2026 \
  --root data/literature/conference_survey --allow-incomplete
python -m paper_atlas --corpus data/literature/conference_survey --port 8765
```

This starts keyword search without an API key. Collection may take several
minutes and depends on public source availability. Acceptance comparisons also
need separately audited submission/decision downloads; until then, topic rates
are **unavailable**, never inferred from accepted papers. Follow the interactive
instructions in [the website guide](docs/conference_web.md).

Alternatively run `paper-atlas --port 8765 --enable-ai`; without `--corpus`, it discovers the
latest corpus under `data/literature/conference_survey*`. Set
`PAPER_ATLAS_DATA_DIR` to use another data directory.

Open **http://localhost:8765**. With VS Code Remote SSH, forward port 8765
privately in the Ports panel. The server never binds to a public interface.

To prepare embeddings separately, without OpenAI calls:

```bash
python -m paper_atlas.semantic_search --corpus data/literature/conference_survey
```

`--enable-ai` enables hybrid indexes automatically. `--enable-semantic` can
prepare them on an AI-disabled diagnostic server. Legacy keyword API requests
remain local and do not call the embedding encoder or OpenAI.

## Getting data on another machine

```bash
python -m paper_atlas.conference_survey --help
python -m paper_atlas.conference_survey collect --help
python -m paper_atlas.openreview_auth_check --help
```

See [corpus collection and analysis](docs/conference_survey.md),
[website and official decision downloads](docs/conference_web.md), and
[data provenance and reuse](docs/DATA.md).

ICLR 2022–2025 public decision snapshots were audited at extraction time. The
2026 decision download was rate-limited and remains unavailable for topic rates.
NeurIPS 2026 publication coverage is also unavailable in the current corpus.

## Tests

```bash
python -m pytest -q
```

`tests/conference_web_browser.cjs` is an optional Playwright browser check against
the real local corpus. It checks desktop/mobile layout, search, exports, and
acceptance-rate provenance. See the website guide for setup and environment
variables. Node is not required to run the website.

## AI query interpretation

**Paragraph → AI-generated structured search concepts → existing search engine → charts.**

Save `OPENAI_API_KEY=your_key` privately in `~/.config/paper-atlas/openai.env`,
outside this repository, with permissions `600`. Alternatively set the server's
`OPENAI_API_KEY` environment variable (which takes precedence). Never put a key
in browser JavaScript, Git, screenshots or chat. The file is parsed, not executed.

```bash
python -m paper_atlas --port 8765 --enable-ai
```

Without `--enable-ai`, no key is loaded; Keyword mode and current results remain
available, while AI mode is blocked. A custom private file
can be selected with `--api-key-file PATH`. The backend uses `requests` to call
the Responses API; local hybrid matching additionally requires the embeddings extra.

Select **AI**, enter a paragraph, click **Explore topic**, review/edit its concepts,
then **Search with these concepts**. **Keyword** is the default and works without
an API key or AI-enabled server. Each comma-separated phrase is required in the
normalized title/abstract; keywords are not expanded or embedded. AI failures never
silently switch modes. Matching and same-pool acceptance computation remain local.
The model does not read or classify individual papers and never supplies counts.
Strict JSON guarantees shape, not semantic correctness; review the concepts.

Search results and successful interpretations are kept only in bounded process
memory (up to 64 each), never in browser storage or query databases. There is no
history menu/API. Restarting clears them; pagination and exports may require
rerunning an evicted search. Existing legacy query databases are left untouched,
but never loaded. HTTP request logging is disabled. The aggregate weekly counter
and public document caches still persist. A temporary cache hit makes no new API
call. Input is limited to 1,500 characters;
the output budget is 1,800 tokens, with one bounded retry at 3,000 for malformed or
truncated output. Refusals, transport and HTTP errors stop without silent fallback
or raw provider-error logging. Each HTTP request has a 10-second connection and
45-second read timeout. The API uses `store=false`; this is not a guarantee of zero
provider retention. Descriptions are sent to OpenAI and normal API billing applies.

### Weekly AI limit

The app allows **1,000 outbound AI requests per calendar week**, resetting Monday
at 00:00 in `America/Chicago` (including daylight-saving changes). Every network
attempt consumes one slot, including failures and retries. Invalid prompts, cache
hits, applying reviewed concepts, downloading current results, and local keyword
searches consume none. At the cap, new API calls are blocked on the backend; the
UI only reports the limit when it is reached; no persistent allowance banner is
shown. This is an app-wide limit, not
a per-browser limit or an OpenAI-account spending cap.

The atomic counter persists in `DATA_DIR/ai_usage.sqlite3`, shared across server
processes and `--state` directories using that data root. Restarting the app does
not reset it. Changing/deleting the data root or making API calls outside this
app is outside this protection. Tracking starts when this feature is enabled;
earlier diagnostic calls are not reconstructed. This is a request cap, not a
guarantee that usage is free or a fixed dollar-budget limit.
