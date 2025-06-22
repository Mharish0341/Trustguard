from __future__ import annotations
import collections
from typing import Any, List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_sbert = SentenceTransformer(EMBED_MODEL)

class LFUCache:
    def __init__(self, cap: int = 4096):
        self.cap  = cap
        self.data : dict[str, np.ndarray]      = {}
        self.freq : collections.Counter[str]   = collections.Counter()

    def get(self, k: str) -> np.ndarray | None:
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

def _embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    vecs = _sbert.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    if vecs.ndim != 2 or vecs.shape[1] < 64:
        raise ValueError(f"Bad embedding shape: {vecs.shape}")
    return vecs.astype("float32")

class EmbedDB:
    def __init__(self):
        self.idx  : faiss.IndexFlatIP | None = None
        self.dim  : int | None               = None
        self.text : List[str]                = []

    def _reset(self, dim: int):
        self.idx = faiss.IndexFlatIP(dim)
        self.dim = dim
        _cache.clear()
        self.text.clear()

    def _ensure(self, dim: int):
        if self.idx is None or dim != self.dim:
            self._reset(dim)

    def add(self, items: List[Any]):
        if not items:
            return

        hits, to_embed = [], []
        for item in items:
            # coerce to a string key
            if isinstance(item, str):
                txt = item
            else:
                try:
                    txt = item.get("body") or item.get("text") or repr(item)
                except Exception:
                    txt = repr(item)
            vec = _cache.get(txt)
            if vec is None:
                to_embed.append(txt)
            else:
                hits.append((txt, vec))
                _cache.bump(txt)

        fresh: List[Tuple[str, np.ndarray]] = []
        if to_embed:
            try:
                vecs = _embed(to_embed)
                for txt, vec in zip(to_embed, vecs):
                    _cache.put(txt, vec)
                    fresh.append((txt, vec))
            except ValueError:
                pass

        all_pairs = hits + fresh
        if not all_pairs:
            return

        canon_dim = all_pairs[0][1].shape[0]
        self._ensure(canon_dim)

        valid_vecs, valid_texts = [], []
        for txt, vec in all_pairs:
            if vec.shape[0] == canon_dim:
                valid_vecs.append(vec)
                valid_texts.append(txt)

        if not valid_vecs:
            return

        arr = (valid_vecs[0].reshape(1, -1)
               if len(valid_vecs) == 1
               else np.vstack(valid_vecs))
        self.idx.add(arr)
        self.text.extend(valid_texts)

    def similar(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        if self.idx is None or self.idx.ntotal == 0:
            return []

        vec = _cache.get(query)
        if vec is None:
            try:
                vec = _embed([query])[0]
                _cache.put(query, vec)
            except ValueError:
                return []

        if vec.shape[0] != self.dim:
            return []

        D, I = self.idx.search(vec.reshape(1, -1), k)
        result: List[Tuple[float, str]] = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.text):
                result.append((float(dist), self.text[idx]))
        return result
