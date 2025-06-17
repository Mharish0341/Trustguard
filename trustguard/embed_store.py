"""
trustguard.embed_store
FAISS vector DB backed by a local Sentence-Transformer model
(no external API calls).
• LFU-caches embeddings.
• Skips malformed vectors (<64-d).
• Never mixes dimensions in one FAISS index.
"""

from __future__ import annotations

import collections
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────────
# Load a light-and-fast SBERT model
# ────────────────────────────────────────────────────────────────
# you can swap in any HuggingFace-compatible model here
_EMBED_MODEL = "all-MiniLM-L6-v2"
_sbert = SentenceTransformer(_EMBED_MODEL)

# ────────────────────────────────────────────────────────────────
# Tiny LFU cache
# ────────────────────────────────────────────────────────────────
class LFUCache:
    def __init__(self, cap: int = 4096):
        self.cap      = cap
        self.data     : dict[str, np.ndarray] = {}
        self.freq     : collections.Counter[str] = collections.Counter()

    def get(self, k: str):
        return self.data.get(k)

    def put(self, k: str, v: np.ndarray):
        if k in self.data:
            return
        if len(self.data) >= self.cap:
            victim, _ = self.freq.most_common()[-1]
            self.data.pop(victim, None)
            self.freq.pop(victim, None)
        self.data[k] = v

    def bump(self, k: str):
        self.freq[k] += 1

    def clear(self):
        self.data.clear()
        self.freq.clear()

_cache = LFUCache()

# ────────────────────────────────────────────────────────────────
# Embedding helper (local SBERT)
# ────────────────────────────────────────────────────────────────
def _embed(texts: List[str]) -> np.ndarray:
    """
    Returns a (n × dim) float32 matrix, normalised for cosine similarity.
    """
    if not texts:
        return np.empty((0, 0), dtype="float32")

    # encode all at once
    vecs = _sbert.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # sanity-check: MiniLM is 384-dim; any <64-dim is bogus
    if vecs.ndim != 2 or vecs.shape[1] < 64:
        raise ValueError(f"Bad embedding shape: {vecs.shape}")

    return vecs.astype("float32")

# ────────────────────────────────────────────────────────────────
# Dimension-safe FAISS wrapper
# ────────────────────────────────────────────────────────────────
class EmbedDB:
    def __init__(self):
        self.idx : faiss.IndexFlatIP | None = None
        self.dim : int | None               = None
        self.text: List[str]                = []

    def _reset(self, dim: int):
        print(f"[EmbedDB] Resetting FAISS & cache to dim={dim}")
        self.idx = faiss.IndexFlatIP(dim)
        self.dim = dim
        _cache.clear()
        self.text.clear()

    def _ensure(self, dim: int):
        if self.idx is None or dim != self.dim:
            self._reset(dim)

    def add(self, texts: List[str]):
        if not texts:
            return

        # 1) split cache hits vs needs-embedding
        hits, to_embed = [], []
        for t in texts:
            v = _cache.get(t)
            if v is None:
                to_embed.append(t)
            else:
                hits.append((t, v))
                _cache.bump(t)

        # 2) embed misses
        fresh = []
        if to_embed:
            try:
                vecs = _embed(to_embed)
                fresh = list(zip(to_embed, vecs))
                for txt, vec in fresh:
                    _cache.put(txt, vec)
            except ValueError as e:
                print("[EmbedDB]", e)

        all_pairs = hits + fresh
        if not all_pairs:
            return

        # 3) pick canonical dim
        canon_dim = all_pairs[0][1].shape[0]
        self._ensure(canon_dim)

        # 4) filter mismatches
        vecs, txts = [], []
        for txt, vec in all_pairs:
            if vec.shape[0] == canon_dim:
                vecs.append(vec); txts.append(txt)

        if not vecs:
            return

        # 5) add to FAISS
        arr = vecs[0].reshape(1, -1) if len(vecs) == 1 else np.vstack(vecs)
        self.idx.add(arr)
        self.text.extend(txts)

    def similar(self, query: str, k: int = 5):
        if self.idx is None or self.idx.ntotal == 0:
            return []

        v = _cache.get(query)
        if v is None:
            try:
                v = _embed([query])[0]
                _cache.put(query, v)
            except ValueError as e:
                print("[EmbedDB]", e)
                return []

        if v.shape[0] != self.dim:
            return []

        D, I = self.idx.search(v.reshape(1, -1), k)
        return [
            (float(d), self.text[i])
            for d, i in zip(D[0], I[0])
            if 0 <= i < len(self.text)
        ]
