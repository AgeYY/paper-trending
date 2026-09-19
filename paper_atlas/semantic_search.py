"""CPU-only local embeddings and shared, exhaustive hybrid membership rules.

No top-k truncation is used for counts. The same predicate is applied to
publications and to every record in a verified submission pool.
"""
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
import tempfile
import threading

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_TOKENS = 256
POLICY = {"version": "hybrid-v1", "model": MODEL, "revision": REVISION,
          "max_tokens": MAX_TOKENS, "pooling": "attention-mask-mean-l2",
          "compute_dtype": "bfloat16", "stored_dtype": "float32",
          "core_cosine_threshold": 0.30, "query_cosine_threshold": 0.45,
          "rrf_k": 60, "calibration": "experimental; not a relevance probability"}

_worker_encoder = None


def _initialize_worker(root):
    global _worker_encoder
    _worker_encoder = LocalEmbeddings(root)
    _worker_encoder._load()


def _encode_worker(texts):
    return _worker_encoder.encode(texts)


class SemanticUnavailable(ValueError):
    pass


class LocalEmbeddings:
    def __init__(self, cache_root):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.model = self.tokenizer = None
        self.matrices = {}

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            torch.set_num_threads(1)
            options = {"revision": REVISION, "cache_dir": str(self.root / "models"),
                       "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(MODEL, **options)
            model = AutoModel.from_pretrained(MODEL, use_safetensors=True, **options)
            self.tokenizer, self.model = tokenizer, model.to("cpu").eval()
        except Exception:
            raise SemanticUnavailable("Local embedding model unavailable. Install paper-atlas[embeddings] and prepare the semantic index; no keyword fallback was used.") from None

    def encode(self, texts, progress=False):
        with self.lock:
            self._load()
            import torch
            vectors = []
            for start in range(0, len(texts), 64):
                batch = self.tokenizer(texts[start:start+64], padding=True, truncation=True,
                                       max_length=MAX_TOKENS, return_tensors="pt")
                with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                    output = self.model(**batch).last_hidden_state.float()
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
                    vectors.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
                if progress and (start % 2048 == 0 or start+64 >= len(texts)):
                    print(f"Local embeddings: {min(start+64,len(texts)):,}/{len(texts):,} documents (CPU)", flush=True)
            return np.concatenate(vectors) if vectors else np.empty((0, 384), dtype=np.float32)

    def documents(self, texts):
        digest = hashlib.sha256(json.dumps([MODEL, REVISION, MAX_TOKENS, POLICY["pooling"], POLICY["compute_dtype"], texts], ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            if digest in self.matrices:
                return self.matrices[digest]
            path = self.root / (digest + ".npy")
            if path.exists():
                try:
                    matrix = np.load(path, allow_pickle=False)
                    if matrix.shape != (len(texts), 384) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                        raise ValueError("Invalid embedding cache")
                except Exception:
                    raise SemanticUnavailable("The local embedding cache is invalid; rebuild the affected index before searching.") from None
            else:
                if len(texts) > 1024:
                    # Independent one-thread workers avoid a large intra-op
                    # thread pool on CPU. No CUDA device is ever selected.
                    chunks = [texts[i:i+512] for i in range(0, len(texts), 512)]
                    completed, pieces = 0, []
                    with ProcessPoolExecutor(max_workers=4,
                            mp_context=multiprocessing.get_context("spawn"),
                            initializer=_initialize_worker, initargs=(str(self.root),)) as pool:
                        for piece in pool.map(_encode_worker, chunks):
                            pieces.append(piece)
                            completed += len(piece)
                            print(f"Local embeddings: {completed:,}/{len(texts):,} documents (4 CPU workers)", flush=True)
                    matrix = np.concatenate(pieces)
                else:
                    matrix = self.encode(texts, progress=True)
                descriptor, temporary = tempfile.mkstemp(prefix="embedding-", suffix=".npy", dir=self.root)
                try:
                    with os.fdopen(descriptor, "wb") as out:
                        np.save(out, matrix, allow_pickle=False)
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            self.matrices[digest] = matrix
            return matrix

    @lru_cache(maxsize=128)
    def queries(self, texts):
        return self.encode(list(texts))


def query_texts(groups):
    """Use reviewed terms, not the original prompt or unedited AI labels."""
    concepts = [", ".join(g["terms"]) for g in groups]
    return tuple(["; ".join(concepts)] + concepts)


def hybrid_membership(lexical, similarities, excluded, policy=POLICY):
    """Rows=documents; columns=whole core query followed by each core concept.

    Optional evidence never changes membership. A missing lexical core may be
    satisfied semantically only when the whole core query also passes threshold.
    """
    lexical = np.asarray(lexical, dtype=bool)
    similarities = np.asarray(similarities)
    exact = lexical.all(axis=1)
    per_core = lexical | (similarities[:, 1:] >= policy["core_cosine_threshold"])
    semantic = per_core.all(axis=1) & (similarities[:, 0] >= policy["query_cosine_threshold"])
    return (exact | semantic) & ~np.asarray(excluded, dtype=bool)


def lexical_hits(texts, groups):
    from paper_atlas.conference_search import phrase_text
    columns = []
    for group in groups:
        terms = [" " + phrase_text(t) + " " for t in group["terms"]]
        columns.append([any(t in text for t in terms) for text in texts])
    return np.array(columns, dtype=bool).T.reshape(len(texts), len(groups))


def evaluate_hybrid(texts, documents, config, embeddings, matrix=None):
    policy = config["retrieval"]
    if policy != POLICY:
        raise SemanticUnavailable("This saved search uses different semantic rules. Rerun it with the current rules.")
    lexical = lexical_hits(texts, config["groups"])
    excluded = lexical_hits(texts, [{"terms": config["exclusions"]}])[:, 0]
    matrix = embeddings.documents(documents) if matrix is None else matrix
    similarities = matrix @ embeddings.queries(query_texts(config["groups"])).T
    membership = hybrid_membership(lexical, similarities, excluded, policy)
    return membership, lexical, similarities


def rrf_scores(lexical_scores, semantic_scores, supporting_scores, identifiers, k=60):
    """Fuse complete candidate rankings; IDs break ties deterministically."""
    count = len(identifiers)
    scores = np.zeros(count)
    for values in (np.asarray(lexical_scores) + np.asarray(supporting_scores), semantic_scores):
        order = sorted(range(count), key=lambda i: (-float(values[i]), identifiers[i]))
        for rank, i in enumerate(order, 1):
            scores[i] += 1 / (k + rank)
    # A small explicit optional-evidence boost also breaks identical-core ties.
    scores += np.minimum(supporting_scores, 4) * .001
    return scores


def main():
    import argparse
    from paper_atlas.conference_search import SearchEngine
    from paper_atlas.settings import DATA_DIR
    parser = argparse.ArgumentParser(description="Prepare local CPU embeddings without any OpenAI calls.")
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    engine = SearchEngine(args.corpus, Path(DATA_DIR)/"semantic-preparation")
    engine.enable_semantics(Path(DATA_DIR)/"semantic")
    print("Semantic indexes ready for publications and all verified submission pools.", flush=True)


if __name__ == "__main__":
    main()
