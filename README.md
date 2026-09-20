# Paper Trending

Search ICLR, ICML, and NeurIPS papers, plot publication trends, and compare ICLR
topic acceptance rates with the overall rate from the same submission pool.

**[Try the live demo →](https://ageyy.github.io/paper-trending/)**

> The demo supports keyword search, charts, and downloads. It downloads a public
> metadata snapshot and searches in your browser. AI-assisted and semantic search
> require the local Python app below. No accounts or saved search history.

## Run locally

Requires Python 3.10+. No GPU needed.

```bash
git clone https://github.com/AgeYY/paper-trending.git
cd paper-trending
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.

Collect public paper metadata, then start the app:

```bash
python -m paper_atlas.conference_survey collect \
  --years 2022 2023 2024 2025 2026 \
  --root data/literature/conference_survey --allow-incomplete
python -m paper_atlas --corpus data/literature/conference_survey --port 8765
```

Open **http://localhost:8765**. Enter comma-separated phrases; all must match the
title or abstract. Keyword search needs no API key. Keep the server local; for
VS Code Remote SSH, forward port 8765 privately.

Topic acceptance rates require separate [OpenReview decision downloads](docs/conference_web.md).
Missing data is shown as unavailable, never zero. Matches are automatic screening
results, not verified relevance.

## Enable AI search

Install the extra dependencies and set the `OPENAI_API_KEY` environment variable:

```bash
python -m pip install -e '.[embeddings]'
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python -m paper_atlas --corpus data/literature/conference_survey --port 8765 --enable-ai
```

Stop the previous server first. In PowerShell, set the variable with
`$env:OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"`.

Select **AI**, describe your topic, review the generated concepts, then search.
GPT-5.4 nano interprets the description; keyword and local embedding search find
the papers. First startup downloads the embedding model and builds its indexes.

Only your description is sent to OpenAI, not the paper corpus. API charges apply;
the app caps outbound AI requests at 1,000 per week. Never commit your key or put
it in browser code.

## More

- [Data collection and analysis](docs/conference_survey.md)
- [Local app, decision downloads, and testing](docs/conference_web.md)
- [Build and publish the browser demo](docs/STATIC_DEMO.md)
- [Data provenance and reuse](docs/DATA.md)
- [Security](SECURITY.md)

[MIT license](LICENSE) for the code. Third-party paper data retains its original rights.
