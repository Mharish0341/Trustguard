from __future__ import annotations
import functools
import io
import time
from typing import List
import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError
import clip
from .config import CLIP_VARIANT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load(CLIP_VARIANT, device=DEVICE)


if DEVICE == "cuda":
    clip_model = clip_model.half()

@functools.lru_cache(maxsize=4096)
def _download(url: str) -> bytes | None:
    """Return raw bytes or None if fetch fails / times out."""
    try:
        with requests.get(url, timeout=10) as r:
            if r.ok:
                return r.content
    except requests.exceptions.RequestException:
        pass
    return None


def _safe_similarity(title: str, raw: bytes) -> float | None:
    """Return CLIP similarity or None on any failure."""
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(DEVICE)
        txt_t = clip.tokenize([title]).to(DEVICE)

        with torch.no_grad():
            v = clip_model.encode_image(img_t)
            t = clip_model.encode_text(txt_t)
            v, t = v / v.norm(dim=-1, keepdim=True), t / t.norm(dim=-1, keepdim=True)
            return float((v @ t.T).item())
    except UnidentifiedImageError:
        return None

def worst_clip_score(title: str, image_urls: List[str], top_n: int = 3) -> float:
    sims: list[float] = []

    for url in image_urls[:top_n]:
        raw = _download(url)
        if not raw:
            continue
        sim = _safe_similarity(title, raw)
        if sim is not None:
            sims.append(sim)

    if not sims:               # nothing worked → high risk
        return 1.0

    worst = min(sims)          # lowest similarity is the riskiest
    return 1.0 - worst         # invert: low sim → high risk
