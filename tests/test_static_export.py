import gzip
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from paper_atlas import static_export
from paper_atlas.conference_corpus import sha256
from paper_atlas.conference_search import phrase_text
from test_conference_web import engine


def node_search(output, queries):
    if not shutil.which("node"):
        pytest.skip("Node is required for browser/Python parity tests")
    script = """
import fs from 'node:fs';
import zlib from 'node:zlib';
import {pathToFileURL} from 'node:url';
const root=process.argv[1];
const {search,phraseText}=await import(pathToFileURL(root+'/search.mjs'));
const m=JSON.parse(fs.readFileSync(root+'/manifest.json'));
const shards=m.shards.map(s=>({...s,data:JSON.parse(zlib.gunzipSync(fs.readFileSync(root+'/'+s.file)))}));
for(const s of shards)if(s.kind==='papers')for(const p of s.data){p.text=' '+phraseText(p.title+'. '+p.abstract)+' ';p.title_text=' '+phraseText(p.title)+' ';}
const input=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(input.map(q=>search(m,shards,q))));
"""
    process = subprocess.run(["node", "--input-type=module", "-e", script, str(output)],
                             input=json.dumps(queries), capture_output=True, text=True, check=True)
    return json.loads(process.stdout)


def test_export_parity_and_privacy(engine, tmp_path, monkeypatch):
    # Synthetic pool tests same-population rates, including withdrawn records.
    key = (2025, "iclr")
    engine.acceptance.public_keys.add(key)
    engine.acceptance.pools[key] = [
        (" " + phrase_text(text) + " ", status) for text, status in [
            ("Reinforcement learning for language models", "accepted"),
            ("Reinforcement learning for language models", "rejected"),
            ("Reinforcement learning for language models", "withdrawn"),
            ("Graph learning", "accepted")]]
    engine.acceptance.provenance[key] = {"source": "https://openreview.net/group?id=ICLR.cc/2025/Conference",
        "population": "Official public pool", "retrieved_at": "2026-09-19", "sha256": "a" * 64}
    monkeypatch.setattr(static_export, "SearchEngine", lambda *_: engine)
    output = tmp_path / "site"
    manifest = static_export.build(engine.root, output, "https://github.com/example/project")
    assert not (output / "unused-state").exists()
    assert (output / "index.html").exists()
    assert "input_file" not in (output / "manifest.json").read_text()
    assert str(tmp_path) not in (output / "manifest.json").read_text()
    for desc in manifest["shards"]:
        assert sha256((output / desc["file"]).read_bytes()) == desc["sha256"]
        rows = json.loads(gzip.decompress((output / desc["file"]).read_bytes()))
        assert len(rows) == desc["count"]
        if desc["kind"] == "papers":
            assert set(rows[0]) == {"id", "title", "abstract", "authors", "track", "url"}
    queries = [{"prompt": q, "years": [2024, 2025, 2026], "venues": ["iclr", "icml", "neurips"], "tracks": tracks}
               for tracks in [["research"], ["research", "position"]]
               for q in ["learning", "reinforcement learning, language models", "graph neural networks", "image generation", "no-such-topic", "graph,,, learning", "ＧＲＰＯ"]]
    for query, browser in zip(queries, node_search(output, queries)):
        python = engine.search({**query, "search_mode": "keyword"})
        assert {p["id"] for p in browser["matches"]} == {p["id"] for p in python["papers"]}
        assert [(r["year"], r["venue"], r["candidates"], r["status"]) for r in browser["counts"]] == [
            (r["year"], r["venue"], r["candidates"], r["status"]) for r in python["counts"]]
        for b, p in zip(browser["acceptance"], python["acceptance"]["rows"]):
            for field in ("topic_accepted", "topic_submitted", "topic_rate"):
                assert b[field] == p[field]
            if b["available"] and query["tracks"] == ["research"]:
                assert b["conference_rate"] == p["conference_rate"]
    with pytest.raises(ValueError, match="empty"):
        static_export.build(engine.root, output, "https://github.com/example/project")


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///private", "https://user:password@example.org", "//example.org", ""])
def test_export_links_reject_unsafe(url):
    assert static_export.public_url(url) == ""


def test_static_resources_packaged():
    for name in ("index.html", "site.css", "site.mjs", "search.mjs", "worker.mjs"):
        assert (static_export.ASSETS / name).is_file()


def test_demo_no_remote_api_or_history():
    source = "\n".join(p.read_text() for p in static_export.ASSETS.glob("*.mjs"))
    for forbidden in ("localStorage", "sessionStorage", "indexedDB", "/api/", "api.openai.com", "OPENAI_API_KEY"):
        assert forbidden not in source
