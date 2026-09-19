"""Explainable core-concept retrieval with optional local hybrid matching.

Legacy keyword requests remain fully local and do not load embedding models.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time

from paper_atlas.conference_corpus import VENUES, normalized_text, sha256
from paper_atlas.conference_survey import DEFAULT_TOPICS
from paper_atlas.transient_store import TransientStore

RULE_VERSION = "core-support-hybrid-v2"
STOP = set("a an the and or for of to in on with by from that this these those how what which when where why can could would should is are was were be been being has have had do does did i we you me us our my their its it they them please find search show tell looking look interested interest about relevant related relevance topic topics paper papers research study studies work works method methods approach approaches technique techniques using use used recent last past five year years number count conference conferences neurips iclr icml within between across specifically especially also all some any into through want like learn understand explore investigate focuses focus aiming based application applications".split())


def words(text):
    tokens = re.findall(r"[a-z0-9]+", normalized_text(text))
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")) else t for t in tokens]


def phrase_text(text):
    return " ".join(words(text))


def vocabulary():
    defaults = json.loads(DEFAULT_TOPICS.read_text())["groups"]
    return [
        ("Reinforcement learning", ["reinforcement learning", "rl", "rlhf", "rlvr", "grpo", "ppo", "policy gradient", "policy optimization"], defaults["rl"] + ["RL"]),
        ("Autoregressive / general LMs", ["autoregressive", "auto-regressive"], ["autoregressive", "auto-regressive", "large language model", "LLM", "GPT", "Llama", "Qwen"]),
        ("Discrete state / masking", ["discrete", "masked", "masking", "llada", "mdlm", "d3pm"], defaults["discrete"]),
        ("Continuous / latent state", ["continuous", "continuous-state", "latent-space", "embedding-space"], defaults["continuous"]),
        ("Flow matching", ["flow matching", "flow-matching", "rectified flow", "flux"], defaults["flow"]),
        ("Diffusion", ["diffusion", "denoising", "diffusion models"], defaults["diffusion"]),
        ("Language generation", ["language model", "language models", "llm", "llms", "text generation", "language generation", "text-to-text"], defaults["language"]),
        ("Image generation", ["image generation", "image synthesis", "text-to-image", "image", "images"], defaults["image"] + ["image"]),
        ("Fine-tuning", ["fine-tuning", "finetuning", "fine tuning", "fine-tune", "finetune", "fine tune", "post-training", "posttraining"], ["fine tuning", "fine tune", "finetuning", "finetune", "post training", "adaptation", "alignment"]),
        ("Preference alignment", ["preference optimization", "preference optimisation", "alignment", "dpo", "human preference"], ["preference optimization", "preference optimisation", "DPO", "alignment", "human preference", "RLHF"]),
        ("Graph neural networks", ["graph neural network", "graph neural networks", "gnn", "gnns"], ["graph neural network", "graph convolution", "graph transformer", "GNN"]),
        ("Drug discovery", ["drug discovery", "drug design"], ["drug discovery", "drug design", "molecular generation", "molecule generation", "drug candidate"]),
        ("Privacy", ["differential privacy", "differentially private"], ["differential privacy", "differentially private", "DP-SGD"]),
        ("Machine unlearning", ["machine unlearning", "unlearning"], ["unlearning", "data deletion", "machine forgetting"]),
        ("Retrieval augmented generation", ["retrieval augmented generation", "rag"], ["retrieval augmented", "RAG"]),
        ("Recommendation", ["recommendation", "recommender", "recommender systems"], ["recommendation", "recommender", "collaborative filtering"]),
        ("Robotics", ["robotics", "robot", "robots", "robotic"], ["robot", "robotic", "robotics", "manipulation"]),
    ]


def validate_groups(groups):
    if not isinstance(groups, list) or not 1 <= len(groups) <= 16:
        raise ValueError("Use between 1 and 16 search concepts.")
    result = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Each concept needs a label and terms.")
        terms = group.get("terms")
        if not isinstance(terms, list) or not 1 <= len(terms) <= 30:
            raise ValueError("Each concept needs 1–30 alternative phrases.")
        if any(not isinstance(t, str) or not 1 <= len(t.strip()) <= 120 or not words(t) for t in terms):
            raise ValueError("Search phrases must be nonempty text, at most 120 characters.")
        label = group.get("label", terms[0])
        if not isinstance(label, str) or not 1 <= len(label) <= 120:
            raise ValueError("Concept labels must contain 1–120 characters.")
        result.append({"label": label, "terms": list(dict.fromkeys(t.strip() for t in terms))})
    return result


def interpret_prompt(prompt):
    if not isinstance(prompt, str) or not 3 <= len(prompt.strip()) <= 1500:
        raise ValueError("Describe a topic in 3–1,500 characters.")
    parts = re.split(r"\b(?:excluding|exclude|without|but not)\b", prompt, maxsplit=1, flags=re.I)
    positive = parts[0]
    exclusions = []
    warnings = []
    if len(parts) > 1:
        exclusions = [part.strip(" .") for part in re.split(r",|;|\band\b|\bor\b", parts[1]) if part.strip(" .")]
        warnings.append("Exclusions remove papers when the phrase occurs anywhere, even in background discussion.")
    if re.search(r"\b(?:not|except|rather than)\b", positive, flags=re.I):
        raise ValueError("For exclusions, use ‘excluding …’ at the end of your topic description.")
    remaining = " " + phrase_text(positive) + " "
    groups = []
    # Recognize each concept independently, then remove all recognized spans.
    recognized = []
    for label, triggers, alternatives in vocabulary():
        hits = [t for t in triggers if " " + phrase_text(t) + " " in remaining]
        if hits:
            recognized.extend(hits)
            groups.append({"label": label, "terms": alternatives})
    for term in sorted(recognized, key=len, reverse=True):
        remaining = remaining.replace(" " + phrase_text(term) + " ", " ")
    if any(g["label"] == "Reinforcement learning" for g in groups):
        groups = [g for g in groups if g["label"] != "Fine-tuning"]
    residual = list(dict.fromkeys(t for t in remaining.split() if t not in STOP and not t.isdigit() and len(t) > 1))
    groups.extend({"label": t, "terms": [t]} for t in residual)
    if residual:
        warnings.append("Unrecognized terms are literal concepts. Try a shorter description if results are too narrow.")
    if any(g["label"] == "Continuous / latent state" for g in groups):
        warnings.append("Continuous time does not imply a continuous state space. Inspect papers before assigning the model family.")
    if any(g["label"] == "Autoregressive / general LMs" for g in groups):
        warnings.append("Generic LLM terms are included for recall; autoregressive architecture still needs verification.")
    return {"groups": validate_groups(groups), "exclusions": exclusions, "warnings": warnings,
            "method": "Local keyword concepts + synonym expansion. No LLM or embedding model is called."}


class SearchEngine:
    def __init__(self, corpus_root, state_root):
        self.root, self.state_root = Path(corpus_root), Path(state_root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        raw = (self.root / "papers.jsonl").read_bytes()
        if sha256(raw) != self.manifest["corpus_sha256"]:
            raise ValueError("Corpus checksum differs from manifest; recollect before searching.")
        self.papers = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if len(self.papers) != self.manifest["paper_count"] or len({p["id"] for p in self.papers}) != len(self.papers):
            raise ValueError("Corpus count or unique paper IDs disagree.")
        self.by_id = {p["id"]: i for i, p in enumerate(self.papers)}
        self.texts, self.titles, self.index = [], [], defaultdict(set)
        for i, paper in enumerate(self.papers):
            text = phrase_text(paper["title"] + ". " + paper["abstract"])
            self.texts.append(" " + text + " ")
            self.titles.append(" " + phrase_text(paper["title"]) + " ")
            for token in set(text.split()):
                self.index[token].add(i)
        self._results = TransientStore()
        self._reviews = TransientStore()
        self._instance_nonce = secrets.token_hex(16)
        self.lock = threading.RLock()
        self.embeddings = None
        self.document_vectors = None
        self.documents = [p["title"] + ". " + p["abstract"] for p in self.papers]
        from paper_atlas.conference_acceptance import AcceptanceStore
        self.acceptance = AcceptanceStore(self.root)

    def enable_semantics(self, cache_root, embeddings=None):
        from paper_atlas.semantic_search import LocalEmbeddings
        encoder = embeddings or LocalEmbeddings(cache_root)
        vectors = encoder.documents(self.documents)
        for texts in self.acceptance.documents.values():
            encoder.documents(texts)
        # Publish the backend only after every required pool is prepared.
        self.embeddings, self.document_vectors = encoder, vectors
        self.acceptance.embeddings = encoder
        self.acceptance._compute.cache_clear()

    @lru_cache(maxsize=2048)
    def phrase_matches(self, phrase):
        tokens = words(phrase)
        if not tokens:
            return frozenset()
        postings = [self.index.get(t, set()) for t in tokens]
        possible = set.intersection(*sorted(postings, key=len))
        needle = " " + " ".join(tokens) + " "
        return frozenset(i for i in possible if needle in self.texts[i])

    def coverage(self, years, venues, tracks):
        rows = []
        lookup = {(r["year"], r["venue"]): r for r in self.manifest["coverage"]}
        sizes = Counter((p["year"], p["venue"]) for p in self.papers if p["track"] in tracks)
        missing = Counter((p["year"], p["venue"]) for p in self.papers if p["track"] in tracks and not p["abstract"])
        for year in years:
            for venue in venues:
                original = lookup.get((year, venue), {"status": "unavailable", "reason": "Not in this corpus snapshot"})
                status = original["status"]
                if status not in ("failed", "unavailable"):
                    issues = [x for x in original.get("issues", []) if x.get("track", "unknown") in (*tracks, "unknown")]
                    status = "partial" if issues or missing[(year, venue)] else "complete"
                rows.append({"year": year, "venue": venue, "status": status, "papers": sizes[(year, venue)],
                             "missing_abstracts": missing[(year, venue)], "reason": original.get("reason", "")})
        return rows

    def metadata(self):
        from paper_atlas.semantic_search import POLICY
        years = list(range(date.today().year - 4, date.today().year + 1))
        return {"years": years, "available_years": self.manifest["years"], "venues": VENUES,
                "snapshot": self.manifest["created_at"], "corpus_hash": self.manifest["corpus_sha256"],
                "papers": sum(p["track"] == "research" for p in self.papers),
                "semantic": {"enabled": self.embeddings is not None, "policy": dict(POLICY)},
                "coverage": self.coverage(years, list(VENUES), ["research"]), "rule_version": RULE_VERSION,
                "examples": ["Reinforcement learning for autoregressive language models",
                             "Reinforcement learning for discrete diffusion language models",
                             "Reinforcement learning for continuous diffusion language models",
                             "Reinforcement learning for flow matching image generation"]}

    def search(self, request, interpretation=None):
        started = time.perf_counter()
        prompt = request.get("prompt", "")
        search_mode = request.get("search_mode")
        if search_mode not in (None, "keyword", "ai"):
            raise ValueError("Search mode must be keyword or ai.")
        if search_mode == "keyword":
            if any(k in request for k in ("groups", "supporting_groups", "exclusions", "hybrid", "mode", "interpretation_id")) or interpretation is not None:
                raise ValueError("Keyword mode accepts comma-separated phrases, not AI or custom criteria.")
            if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1500:
                raise ValueError("Enter comma-separated keywords in 1–1,500 characters.")
            phrases = list(dict.fromkeys(t.strip() for t in prompt.split(",") if t.strip()))
            if not phrases or any(not phrase_text(t) for t in phrases):
                raise ValueError("Enter at least one keyword or phrase containing letters or numbers.")
            request = {**request, "groups": validate_groups([{"label": t, "terms": [t]} for t in phrases])}
        elif not isinstance(prompt, str) or not 3 <= len(prompt.strip()) <= 1500:
            raise ValueError("Describe a topic in 3–1,500 characters.")
        if search_mode == "ai" and interpretation is None:
            raise ValueError("AI mode requires a reviewed interpretation.")
        interpreted = ({"groups": request["groups"], "exclusions": [],
                        "method": "Reviewed search concepts; local lexical matching",
                        "warnings": ["Concepts are combined with AND; alternatives within each concept use OR."]}
                       if "groups" in request else interpret_prompt(prompt))
        groups = validate_groups(request["groups"]) if "groups" in request else interpreted["groups"]
        supporting = request.get("supporting_groups", [])
        if not isinstance(supporting, list):
            raise ValueError("Supporting concepts must be a list.")
        supporting = validate_groups(supporting) if supporting else []
        hybrid = request.get("hybrid", False)
        if type(hybrid) is not bool:
            raise ValueError("Hybrid must be a boolean.")
        if hybrid and self.embeddings is None:
            from paper_atlas.semantic_search import SemanticUnavailable
            raise SemanticUnavailable("Local semantic index unavailable. Prepare and enable it before hybrid search; no keyword fallback was used.")
        exclusions = request.get("exclusions", interpreted["exclusions"])
        if not isinstance(exclusions, list) or len(exclusions) > 30 or any(not isinstance(t, str) or not 1 <= len(t.strip()) <= 120 for t in exclusions):
            raise ValueError("Exclusions must be at most 30 nonempty phrases, each at most 120 characters.")
        years = request.get("years", self.metadata()["years"])
        venues = request.get("venues", list(VENUES))
        tracks = request.get("tracks", ["research"])
        if not isinstance(years, list) or not 1 <= len(years) <= 10 or any(type(y) is not int or not 2000 <= y <= 2100 for y in years):
            raise ValueError("Select 1–10 valid conference years.")
        if not isinstance(venues, list) or not venues or any(v not in VENUES for v in venues):
            raise ValueError("Select at least one supported conference.")
        if not isinstance(tracks, list) or not tracks or any(t not in ("research", "position", "datasets-and-benchmarks") for t in tracks):
            raise ValueError("Unsupported paper track.")
        mode = request.get("mode", "all")
        if mode not in ("all", "half", "any"):
            raise ValueError("Match mode must be all, half or any.")
        if hybrid and mode != "all":
            raise ValueError("Hybrid search requires all core concepts.")
        config = {"prompt": prompt.strip(), "groups": groups, "exclusions": exclusions,
                  "years": sorted(set(years)), "venues": sorted(set(venues)), "tracks": sorted(set(tracks)), "mode": mode}
        config["supporting_groups"] = supporting
        if search_mode is not None:
            config["search_mode"] = search_mode
        if search_mode == "keyword":
            interpreted["method"] = "Comma-separated keyword phrases; all phrases required in title/abstract; no AI, synonyms or embeddings"
        if hybrid:
            from paper_atlas.semantic_search import POLICY
            config["retrieval"] = dict(POLICY)
        if interpretation is not None:
            config["interpretation"] = interpretation
            interpreted["method"] = "AI-assisted, user-reviewed concepts; local lexical matching"
        if hybrid:
            interpreted["method"] = "AI-assisted core concepts + optional evidence; local keyword/embedding hybrid search"
            interpreted["warnings"] = [
                "Every core concept must match by keyword or embedding similarity. Supporting concepts boost rank only.",
                "Semantic thresholds are experimental, not relevance probabilities. Counts include all qualifying candidates, not a top-k list.",
                "Generic LLM matches do not establish autoregressive architecture; matching terms may occur only in comparisons.",
                "Embeddings use the title and the first 256 wordpieces of title plus abstract. Keyword matching uses the full abstract."]
        elif supporting:
            interpreted["warnings"].append("Supporting concepts affect ranking, never membership or counts.")
        if exclusions and "groups" in request:
            interpreted["warnings"].append("Exclusions also remove papers mentioning the phrase only in background discussion.")
        signature = {**config, "corpus": self.manifest["corpus_sha256"], "rules": RULE_VERSION,
                     "instance": self._instance_nonce}
        search_id = sha256(json.dumps(signature, sort_keys=True).encode())[:24]
        term_sets = [[(term, self.phrase_matches(term)) for term in group["terms"]] for group in groups]
        group_sets = [set().union(*(indices for _, indices in terms)) for terms in term_sets]
        required = len(groups) if mode == "all" else math.ceil(len(groups) / 2) if mode == "half" else 1
        if mode == "all":
            candidates = set.intersection(*sorted(group_sets, key=len))
        else:
            hits = Counter(i for group in group_sets for i in group)
            candidates = {i for i, count in hits.items() if count >= required}
        excluded = set().union(*(self.phrase_matches(t) for t in exclusions))
        similarities = None
        if hybrid:
            from paper_atlas.semantic_search import evaluate_hybrid
            membership, lexical, similarities = evaluate_hybrid(self.texts, self.documents, config,
                self.embeddings, self.document_vectors)
            candidates = set(membership.nonzero()[0])
        supporting_sets = [[(term, self.phrase_matches(term)) for term in group["terms"]] for group in supporting]
        matches = []
        for i in candidates - excluded:
            paper = self.papers[i]
            if paper["year"] not in years or paper["venue"] not in venues or paper["track"] not in tracks:
                continue
            evidence, score = [], 0.
            for group, terms, indices in zip(groups, term_sets, group_sets):
                found = [term for term, hits in terms if i in hits]
                if found:
                    title_hit = any(" " + phrase_text(t) + " " in self.titles[i] for t in found)
                    score += math.log(1 + len(self.papers) / max(1, len(indices))) * (2 if title_hit else 1)
                    evidence.append({"concept": group["label"], "terms": found, "in_title": title_hit, "role": "core", "source": "keyword"})
                elif hybrid:
                    column = groups.index(group) + 1
                    evidence.append({"concept": group["label"], "terms": [], "role": "core", "source": "semantic",
                                     "cosine": round(float(similarities[i, column]), 4)})
            supporting_score = 0.
            for group, terms in zip(supporting, supporting_sets):
                found = [term for term, hits in terms if i in hits]
                if found:
                    supporting_score += 1.
                    evidence.append({"concept": group["label"], "terms": found, "role": "supporting", "source": "keyword"})
            match = {"paper_id": paper["id"], "score": round(score + supporting_score, 4), "evidence": evidence}
            if hybrid:
                match.update(lexical_score=score, supporting_score=supporting_score,
                    semantic_similarity=round(float(similarities[i, 0]), 6),
                    match_kind="keyword" if lexical[i].all() else "semantic-expanded")
            matches.append(match)
        if hybrid and matches:
            from paper_atlas.semantic_search import rrf_scores
            fused = rrf_scores([m["lexical_score"] for m in matches], [m["semantic_similarity"] for m in matches],
                [m["supporting_score"] for m in matches], [m["paper_id"] for m in matches])
            for match, score in zip(matches, fused):
                match["score"] = round(float(score), 8)
        matches.sort(key=lambda m: (-m["score"], -self.papers[self.by_id[m["paper_id"]]]["year"], m["paper_id"]))
        result = {"id": search_id, "config": config, "matches": matches, "warnings": interpreted["warnings"],
                  "method": interpreted["method"], "coverage": self.coverage(config["years"], config["venues"], config["tracks"]),
                  "elapsed_ms": round(1000 * (time.perf_counter() - started)), "corpus_hash": self.manifest["corpus_sha256"],
                  "snapshot": self.manifest["created_at"], "rule_version": RULE_VERSION}
        self._results.put(search_id, result)
        return self.page(search_id)

    def load(self, search_id):
        if not re.fullmatch(r"[0-9a-f]{24}", search_id):
            raise ValueError("Invalid search ID.")
        try:
            return self._results.get(search_id)
        except KeyError:
            raise KeyError("Result expired or server restarted. Run the search again.") from None

    def decisions(self, search_id):
        try:
            return self._reviews.get(search_id)
        except KeyError:
            return {}

    def page(self, search_id, page=1, sort="relevance", review="all"):
        if type(page) is not int or page < 1 or sort not in ("relevance", "newest", "oldest") or review not in ("all", "unreviewed", "include", "exclude"):
            raise ValueError("Invalid pagination or result filter.")
        result = self.load(search_id)
        decisions = self.decisions(search_id)
        matches = result.pop("matches")
        if result["config"].get("retrieval"):
            result["retrieval_summary"] = {
                "keyword_core": sum(m.get("match_kind") == "keyword" for m in matches),
                "semantic_expanded": sum(m.get("match_kind") == "semantic-expanded" for m in matches)}
        totals = Counter((self.papers[self.by_id[m["paper_id"]]]["year"], self.papers[self.by_id[m["paper_id"]]]["venue"]) for m in matches)
        included = Counter((self.papers[self.by_id[m["paper_id"]]]["year"], self.papers[self.by_id[m["paper_id"]]]["venue"]) for m in matches if decisions.get(m["paper_id"]) == "include")
        counts = []
        for row in result["coverage"]:
            available = row["status"] in ("complete", "partial")
            key = row["year"], row["venue"]
            counts.append({**row, "candidates": totals[key] if available else None,
                           "confirmed": included[key] if available else None,
                           "per_1000": round(1000 * totals[key] / row["papers"], 3) if row["papers"] else None})
        counts_review = Counter(decisions.get(m["paper_id"], "unreviewed") for m in matches)
        result.update({"total": len(matches), "screened": sum(r["papers"] for r in result["coverage"]),
                       "review_counts": dict(counts_review), "counts": counts,
                       "incomplete": any(r["status"] != "complete" for r in result["coverage"])})
        if review != "all":
            matches = [m for m in matches if decisions.get(m["paper_id"], "unreviewed") == review]
        if sort != "relevance":
            matches.sort(key=lambda m: self.papers[self.by_id[m["paper_id"]]]["year"], reverse=sort == "newest")
        pages = max(1, math.ceil(len(matches) / 20))
        page = min(page, pages)
        result.update({"page": page, "pages": pages, "filtered_total": len(matches), "sort": sort, "review_filter": review})
        result["papers"] = [{**self.papers[self.by_id[m["paper_id"]]], **m, "decision": decisions.get(m["paper_id"], "unreviewed")}
                            for m in matches[(page - 1) * 20:page * 20]]
        result["acceptance"] = self.acceptance.compute(result["config"])
        return result

    def review(self, search_id, paper_id, decision):
        result = self.load(search_id)
        if decision not in ("include", "exclude", "unreviewed") or paper_id not in {m["paper_id"] for m in result["matches"]}:
            raise ValueError("Invalid decision or paper outside this search.")
        with self.lock:
            decisions = self.decisions(search_id)
            if decision == "unreviewed":
                decisions.pop(paper_id, None)
            else:
                decisions[paper_id] = decision
            self._reviews.put(search_id, decisions)
