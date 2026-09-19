"""Deterministic hybrid correctness tests: no downloads, GPU or paid API calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from paper_atlas.ai_interpreter import normalize_concepts, validate_output
from paper_atlas.semantic_search import (POLICY, LocalEmbeddings, SemanticUnavailable,
    evaluate_hybrid, hybrid_membership, query_texts, rrf_scores)
from paper_atlas.conference_acceptance import AcceptanceStore
from test_conference_web import engine
from test_conference_acceptance import cohort, config


class FakeEmbeddings:
    def __init__(self, semantic_phrase=None):
        self.semantic_phrase = semantic_phrase

    def documents(self, texts):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        if self.semantic_phrase:
            for i, text in enumerate(texts):
                if self.semantic_phrase in text:
                    result[i, :3] = [.7, .5, .5]
        return result

    def queries(self, texts):
        return np.eye(384, dtype=np.float32)[:len(texts)]


def test_membership_preserves_core_requirements_and_expands_paraphrases():
    lexical = [[True, True], [False, False], [True, False], [False, False], [True, True]]
    sims = [[0, 0, 0], [.7, .4, .4], [.7, .4, .1], [.2, .4, .4], [.9, .9, .9]]
    assert hybrid_membership(lexical, sims, [False, False, False, False, True]).tolist() == [True, True, False, False, False]


def test_hybrid_adds_real_search_candidates_without_top_k(engine, tmp_path):
    engine.enable_semantics(tmp_path, FakeEmbeddings("Graph neural networks for drug discovery"))
    request = {"prompt": "rare paraphrase topic", "groups": [{"label": "A", "terms": ["xyz"]},
        {"label": "B", "terms": ["qwerty"]}], "hybrid": True}
    found = engine.search(request)
    assert found["total"] == 1
    assert found["papers"][0]["id"] == "paper:2"
    assert found["papers"][0]["match_kind"] == "semantic-expanded"
    assert all(e["source"] == "semantic" for e in found["papers"][0]["evidence"])
    assert found["config"]["retrieval"] == POLICY
    assert found["retrieval_summary"] == {"keyword_core": 0, "semantic_expanded": 1}
    assert engine.search({**request, "years": [2025]})["total"] == 0
    assert engine.search({**request, "venues": ["icml"]})["total"] == 0
    assert engine.search({**request, "exclusions": ["drug discovery"]})["total"] == 0


def test_optional_support_changes_rank_not_counts_or_membership(engine, tmp_path):
    engine.enable_semantics(tmp_path, FakeEmbeddings())
    request = {"prompt": "reinforcement learning language models", "hybrid": True,
        "groups": [{"label": "RL", "terms": ["reinforcement learning"]},
                   {"label": "LM", "terms": ["language model"]}]}
    before = engine.search(request)
    after = engine.search({**request, "supporting_groups": [{"label": "AR", "terms": ["autoregressive"]}]})
    assert before["total"] == after["total"] == 2
    assert {p["id"] for p in before["papers"]} == {p["id"] for p in after["papers"]}
    assert after["papers"][0]["id"] == "paper:3"
    assert before["counts"] == after["counts"]
    assert before["id"] != after["id"]
    assert engine.page(after["id"])["papers"] == after["papers"]


def test_no_silent_fallback_and_legacy_stays_local(engine):
    assert engine.search({"prompt": "graph neural networks"})["total"] == 2
    with pytest.raises(SemanticUnavailable, match="no keyword fallback"):
        engine.search({"prompt": "graph neural networks", "hybrid": True})


def test_shared_membership_for_accepted_rejected_and_other_statuses(cohort):
    store = AcceptanceStore(cohort[0], cohort[1])
    store.embeddings = FakeEmbeddings("graph prediction")
    query = {**config(["zzzz"]), "retrieval": dict(POLICY)}
    result = store.compute(query)["rows"][0]
    assert result["topic_submitted"] == 3  # rejected, withdrawn and desk-rejected
    assert result["topic_accepted"] == 0
    assert result["topic_rate"] == 0
    assert result["conference_rate"] == 40
    query["supporting_groups"] = [{"label": "irrelevant optional", "terms": ["never matches"]}]
    assert store.compute(query)["rows"][0] == result
    query["retrieval"]["query_cosine_threshold"] = .99
    assert store.compute(query)["rows"][0]["topic_rate"] is None


def test_missing_semantic_pool_never_uses_lexical_rate(cohort):
    store = AcceptanceStore(cohort[0], cohort[1])
    row = store.compute({**config(), "retrieval": dict(POLICY)})["rows"][0]
    assert row["topic_rate"] is None
    assert row["conference_rate"] == 40
    assert "no lexical substitution" in row["topic_reason"]


def test_embedding_queries_respect_edits_and_ignore_optional_evidence():
    assert query_texts([{"label": "misleading", "terms": ["graphs", "networks"]}]) == ("graphs, networks", "graphs, networks")


def test_empty_index_and_policy_version():
    query = {"groups": [{"terms": ["graphs"]}], "exclusions": [], "retrieval": dict(POLICY)}
    mask, _, _ = evaluate_hybrid([], [], query, FakeEmbeddings())
    assert mask.shape == (0,)
    query["retrieval"]["version"] = "old"
    with pytest.raises(SemanticUnavailable):
        evaluate_hybrid([], [], query, FakeEmbeddings())


def test_embedding_cache_restart_content_invalidation_and_corruption(tmp_path, monkeypatch):
    encoder = LocalEmbeddings(tmp_path)
    calls = []
    def encode(texts, **kwargs):
        calls.append(texts)
        return np.zeros((len(texts), 384), dtype=np.float32)
    monkeypatch.setattr(encoder, "encode", encode)
    original = encoder.documents(["first"])
    assert len(calls) == 1
    other = LocalEmbeddings(tmp_path)
    monkeypatch.setattr(other, "encode", lambda *a, **k: pytest.fail("Cache must load without a model"))
    np.testing.assert_equal(other.documents(["first"]), original)
    encoder.documents(["changed"])
    assert len(calls) == 2
    for path in tmp_path.glob("*.npy"):
        path.write_bytes(b"corrupt fixture")
    with pytest.raises(SemanticUnavailable, match="invalid"):
        LocalEmbeddings(tmp_path).documents(["first"])


def test_recall_normalization_preserves_ai_exclusion_and_explanation():
    value = {"groups": [{"label": "too narrow", "terms": ["autoregressive language model"]}],
             "supporting_groups": [], "exclusions": ["diffusion"], "needs_clarification": False,
             "explanation": "Original"}
    result, notes = normalize_concepts(value, "Reinforcement learning for autoregressive language models")
    validate_output(result)
    assert len(result["groups"]) == 2
    assert {"RLHF", "RLVR", "PPO", "GRPO"} <= set(result["groups"][0]["terms"])
    assert {"language model", "LLM", "Qwen"} <= set(result["groups"][1]["terms"])
    assert result["supporting_groups"][0]["terms"][0] == "autoregressive"
    assert result["exclusions"] == ["diffusion"] and notes
    assert result["explanation"] == "Original"
    assert value["exclusions"] == ["diffusion"]  # preserve raw model output
    explicit, _ = normalize_concepts(value, "Reinforcement learning for autoregressive language models excluding diffusion")
    assert explicit["exclusions"] == ["diffusion"]


@pytest.mark.parametrize("prompt", [
    "Reinforcement learning for autoregressive language models",
    "RL for autoregressive language models in robotics",
    "Training causal language models",
])
def test_no_exclusion_added_when_ai_returns_empty_list(prompt):
    value = {"groups": [{"label": "LM", "terms": ["language model"]}],
             "supporting_groups": [], "exclusions": [], "needs_clarification": False,
             "explanation": "Original"}
    result, notes = normalize_concepts(value, prompt)
    validate_output(result)
    assert result["exclusions"] == []
    assert result["explanation"] == "Original"
    assert value["exclusions"] == []


@pytest.mark.parametrize("prompt", [
    "RL for discrete diffusion language models",
    "RL for language models",
    "RL for autoregressive image generation",
    "Compare autoregressive versus diffusion language models",
    "Autoregressive language models including diffusion comparisons",
    "Non-autoregressive language models",
    "Language models that are not autoregressive",
])
def test_normalization_does_not_override_ai_exclusions_for_any_family(prompt):
    value = {"groups": [{"label": "LM", "terms": ["language model"]}],
             "supporting_groups": [], "exclusions": ["diffusion"],
             "needs_clarification": False, "explanation": "Original"}
    result, _ = normalize_concepts(value, prompt)
    assert result["exclusions"] == ["diffusion"]


def test_inferred_exclusions_need_not_appear_literally_in_prompt():
    value = {"groups": [{"label": "LM", "terms": ["language model"]}],
             "supporting_groups": [], "exclusions": ["robotics", "vision"],
             "needs_clarification": False, "explanation": "Original"}
    prompt = "Autoregressive language models excluding robotics"
    result, _ = normalize_concepts(value, prompt)
    assert result["exclusions"] == ["robotics", "vision"]
    result, _ = normalize_concepts({**value, "needs_clarification": True}, prompt)
    assert result["exclusions"] == ["robotics", "vision"]


@pytest.mark.parametrize("prompt", ["RL for autoregressive image generation", "RL for autoregressive language models in robotics"])
def test_recall_policy_does_not_drop_additional_domains(prompt):
    value = {"groups": [{"label": "Domain", "terms": ["robotics", "image generation"]}],
        "supporting_groups": [], "exclusions": [], "needs_clarification": False, "explanation": "Domain"}
    result, _ = normalize_concepts(value, prompt)
    assert result["groups"] == value["groups"]


def test_strict_supporting_schema_required():
    from test_ai_interpreter import answer
    value = answer()
    value.pop("supporting_groups")
    with pytest.raises(ValueError):
        validate_output(value)


def test_rank_fusion_stable_ties():
    scores = rrf_scores([1, 1], [.5, .5], [0, 0], ["b", "a"])
    assert scores[1] > scores[0]


def test_hybrid_exports_are_labeled(engine, tmp_path):
    from paper_atlas.conference_web import chart_bytes, acceptance_chart_bytes
    engine.enable_semantics(tmp_path, FakeEmbeddings())
    result = engine.search({"prompt": "graph neural networks", "hybrid": True})
    assert b"Hybrid screening" in chart_bytes(result, "svg")
    assert b"Hybrid topic screening" in acceptance_chart_bytes(result, "svg")
