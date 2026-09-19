# Reusable conference topic survey

This tool collects official NeurIPS, ICML and ICLR conference catalogs, screens
every collected title and abstract, and draws yearly **candidate-paper** counts.
It is independent of the personal paper-reader database and does not change it.
It does not download/read every full PDF or claim to count distinct algorithms.

For a local prompt-based website with automatic counts, an algorithm page and
downloads, see [Paper Atlas](conference_web.md).

## Run

From the repository root, with Paper Atlas installed in your Python environment:

```bash
python -m paper_atlas.conference_survey run \
  --years 2022 2023 2024 2025 2026 \
  --root data/literature/conference_survey_20260916 \
  --allow-incomplete
```

Equivalent entrypoint: `python bin/conference_survey.py` with the same arguments.
Without `--years`, the default is the current calendar year and four preceding
years. For five *completed* conference years, specify `2021 2022 2023 2024 2025`;
older source schemas may require adapters and will be audited rather than
silently assumed compatible. Default output root respects `paper_atlas.settings.DATA_DIR`
and `PAPER_ATLAS_DATA_DIR`.

The `run` command combines `collect` and `analyze`. Reanalyze without HTTP calls:

```bash
python -m paper_atlas.conference_survey analyze \
  --root data/literature/conference_survey_20260916 --allow-incomplete
```

`collect --offline` regenerates the corpus from cached HTTP bodies. `--refresh`
refetches metadata, retaining old content-addressed bodies. Do not refresh while
another process is using the same root. To retain multiple analyses, give each
an independent `--output` directory. Analysis snapshots both configuration and
source manifest. Sources are public static conference JSON, not the OpenReview
API; fetching uses bounded retries and three concurrent workers by default.

## What is counted?

- **Publication unit:** one conference/year paper; oral/poster/spotlight
  presentations are deduplicated using OpenReview or submission IDs and title
  fallback. Conference year is used, **not arXiv upload year**. Cross-venue
  publications are separate publication events, not globally unique algorithms.
- **Scope:** research-track papers by default. The corpus also retains position,
  datasets-and-benchmarks, journal-to-conference, blog and reproducibility-challenge
  metadata where present. Choose, for example,
  `--tracks research position datasets-and-benchmarks` to expand the analysis.
  Workshops and arXiv-only work are not covered. Historical catalogs can differ
  from final proceedings (withdrawals, revisions, late changes).
- **Coverage:** `complete` means the official catalog's declared event count was
  fetched and reconciled, selected-track records have abstracts, and no unresolved
  in-scope events were found. It is **not an independent proof of proceedings
  completeness or topic recall**. Empty/failed catalogs are unavailable, not
  evidence that no research exists. Missing abstracts and unknown tracks are
  exposed. The analysis status is scoped to selected tracks; missing abstracts
  in an excluded journal track do not invalidate research-track coverage.
- **Partial year:** bars are hatched and starred when a required source is
  unavailable/partial. Totals then cover available papers only. Individual CSV
  rows always carry source status: counts for `unavailable`/`failed` sources
  are blank, **not zero**. The program exits 2 for
  incomplete coverage unless `--allow-incomplete` is explicitly supplied.
- **Denominator:** `counts.csv` includes papers screened and keyword candidates
  per 1,000 papers. Raw growth can reflect a growing conference as well as topic
  popularity; the plot deliberately shows absolute counts, not prevalence.
- **Screening:** literal keyword phrases in title + abstract, Unicode normalized,
  case insensitive, with word boundaries. No opaque classification step. Candidate
  categories can overlap, so adding four counts is not meaningful.

At the September 16, 2026 snapshot, NeurIPS 2026's catalog was empty. Its official
[notification date](https://neurips.cc/Conferences/2026/Dates) is September 24.
The collector does not permanently hard-code that year as unavailable: refresh
later to retrieve the published catalog.

## Change topics

Edit/copy [the JSON configuration](../paper_atlas/configs/conference_rl_topics.json), then use
`--topics path/to/my_topics.json`. There is no need to change Python code.

Each topic has an ID, display label, description, optional semantic query, and
`all_of`: every group must match, while phrases *inside* a group are OR-ed.
`none_of` excludes literal phrases anywhere in the title/abstract.

```json
{
  "version": 1,
  "groups": {
    "rl": ["reinforcement learning", "GRPO", "policy gradient"],
    "model": ["masked diffusion", "discrete diffusion", "LLaDA"]
  },
  "topics": [{
    "id": "my_topic",
    "label": "RL for masked diffusion",
    "description": "RL that updates the language generator, not an external controller.",
    "all_of": ["rl", "model", {"any": ["language", "text"]}],
    "none_of": ["robot control"],
    "query": "Reinforcement learning fine-tuning of masked diffusion language models."
  }]
}
```

The supplied four topics are deliberately **broad retrieval queries**:

1. Autoregressive LM RL: retrieves generic LLM RL too; do not assume every LLM
   abstract explicitly specifies the architecture.
2. Discrete diffusion LM RL: categorical/masked state space, even if *time* is continuous.
3. Continuous diffusion/flow LM RL: continuous embedding/latent state space.
   A paper merely mentioning `continuous`, `embedding`, or AR baselines needs review.
4. Flow-matching image RL: the trained generator must use flow matching/rectified
   flow. Generic image diffusion or a robot flow policy is not sufficient.

Default queries target RL and policy optimization, not all preference alignment.
DPO-only papers without RL terms are not systematically included. Add preference
optimization terms if that is your intended scope, and label the expanded scope.
Lexical rules miss synonyms and can match background/comparison text; **no keyword
system establishes that a field is empty**, especially the continuous-LM category.

## Review and audit

`results/index.html` is a browsable report with the plot, coverage, paper links,
abstracts and matched evidence. `matches.csv` and `review_template.csv` contain one
row per candidate/topic. Save a **separate** working review CSV and fill `decision`
with `include` or `exclude`, and `reason` with your justification.

```bash
python -m paper_atlas.conference_survey analyze \
  --root data/literature/conference_survey_20260916 \
  --reviews path/to/my_reviews.csv --output data/literature/reviewed_survey \
  --allow-incomplete
```

To correct a false negative, append a row with a valid `paper_id` from
`papers.jsonl`, a topic ID, and `include`. Duplicate/unknown IDs or invalid decisions
fail loudly. Manual choices do **not** rewrite the raw keyword figure:
The additional `reviewed_trends.png/.svg/.pdf` figure plots human-confirmed includes.
`counts.csv` separately reports `keyword_candidates`, `confirmed_included`,
`reviewed_excluded` and `pending_review`. Confirmed includes are a lower bound
until the intended corpus/retrieval pool is reviewed. Review all candidates and
sample non-matches before making field-size/novelty claims; changing queries can
change both precision and recall. This software deliberately does not turn a
cosine threshold or an LLM classification into verified relevance.

## Optional vector retrieval

Keyword collection/analysis only needs Python plus the existing NumPy/Matplotlib
dependencies. An optional `sentence-transformers` installation enables CPU
embedding retrieval. Install it in an appropriate environment only if wanted;
the tool does not install dependencies automatically.

```bash
python -m paper_atlas.conference_survey analyze \
  --root data/literature/conference_survey_20260916 \
  --semantic-top-k 100 --output data/literature/semantic_survey \
  --allow-incomplete
```

Default embedding: `BAAI/bge-small-en-v1.5`, pinned commit
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Set `--embedding-model` and
`--embedding-revision` together to change it. The first semantic run downloads
model weights and can take substantially longer than keyword screening; it is
**CPU-only**. Long abstracts are token-chunked, vectors token-weighted pooled and
normalized, then compared to the topic query by cosine similarity. Chunk
round-tripping is checked against context limits; it never silently truncates
long documents. Scores are cached using a hash of model, revision, queries,
pooling version and paper texts.

The top K papers **per topic across all selected years** are exported to
`semantic_ranking.csv` and added to the review queue, including keyword misses.
They are **not automatically counted as relevant**, and K is not an estimate of
field size. Keyword plots remain directly comparable between runs. `--offline`
rejects semantic retrieval to preserve its no-network guarantee. Semantic code
has an offline fake-encoder/cache unit test; a real embedding-model run is not
part of the default keyword survey validation.

## Output files

| File | Purpose |
|---|---|
| `cache/bodies/*.json`, `cache/urls/*.json` | Raw responses, SHA256, fetch time and HTTP provenance |
| `papers.jsonl` | Deduplicated catalog publications, text, track and links |
| `manifest.json` | Per-source event reconciliation, issues, corpus hash |
| `results/index.html` | Figure and paper evidence browser |
| `results/topic_trends.png`, `.svg`, `.pdf` | Four-panel keyword screening figure |
| `results/coverage.csv` | Per-venue/year selected scope and missing data |
| `results/counts.csv` | Per-topic/year/venue automatic and reviewed counts |
| `results/matches.csv`, `review_template.csv` | Review queue with lexical evidence |
| `results/topics.used.json`, `manifest.used.json`, `summary.json` | Reproducibility metadata |

No random sampling, training or GPU is used in the keyword workflow; there is no
random seed. Retrieval results depend on source snapshot, topic configuration,
selected tracks and, optionally, pinned embedding model and top K. Generated
data and figures stay under `data/` and should not be committed.

Run focused tests:

```bash
python -m pytest -q tests/test_conference_survey.py
```
