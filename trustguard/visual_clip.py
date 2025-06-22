import time
import functools
import io
import re
from typing import Optional, List

import requests
import torch, clip
from PIL import Image, UnidentifiedImageError
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from .config import CLIP_VARIANT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load(CLIP_VARIANT, device=DEVICE)
if DEVICE == "cuda":
    clip_model = clip_model.half()

@functools.lru_cache(maxsize=4096)
def _download(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=10)
        return r.content if r.ok else None
    except requests.RequestException:
        return None

def _safe_similarity(title: str, raw: bytes) -> Optional[float]:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(DEVICE)
        txt_t = clip.tokenize([title]).to(DEVICE)
        with torch.no_grad():
            v = clip_model.encode_image(img_t)
            t = clip_model.encode_text(txt_t)
            v = v / v.norm(dim=-1, keepdim=True)
            t = t / t.norm(dim=-1, keepdim=True)
            return float((v @ t.T).item())
    except UnidentifiedImageError:
        return None

def worst_clip_score(title: str, image_urls: List[str], top_n: int = 3) -> float:
    sims: List[float] = []
    for url in image_urls[:top_n]:
        raw = _download(url)
        if raw is None:
            continue
        sim = _safe_similarity(title, raw)
        if sim is not None:
            sims.append(sim)
    return 1.0 if not sims else 1.0 - min(sims)


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
blip2     = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b"
).to(DEVICE).eval()

MAX_CALLS_PER_MIN = 15
MIN_INTERVAL    = 60.0 / MAX_CALLS_PER_MIN
_last_call_time = 0.0

def blip2_vision_risk(title: str, raw: bytes) -> float:
    global _last_call_time
    since = time.time() - _last_call_time
    if since < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - since)

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    prompt = (
        "On a scale 0.0 (no relation) → 1.0 (perfect match), how well does "
        "this product title describe the image? Answer with single decimal.\n\n"
        f"Product Title: {title}"
    )

    inputs = processor(img, prompt, return_tensors="pt").to(DEVICE)
    out_ids = blip2.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(out_ids[0], skip_special_tokens=True)

    m = re.search(r"[01]\.[0-9]+|0|1", answer)
    score = float(m.group(0)) if m else 0.5

    _last_call_time = time.time()
    return 1.0 - max(0.0, min(score, 1.0))


def weighted_visual_risk(title: str, image_urls: List[str],clip_n:  int = 3,blip_n:  int = 1,strictness_factor: float = 0.5,  # <1.0 → less strict (lower risk)
) -> float:
    clip_r = worst_clip_score(title, image_urls, top_n=clip_n)

    blip_r = 0.0
    for url in image_urls[:blip_n]:
        raw = _download(url)
        if raw is None:
            continue
        blip_r = max(blip_r, blip2_vision_risk(title, raw))

    base = 0.5 * clip_r + 0.5 * blip_r
    # downscale the risk to make it less strict:
    risk = base * strictness_factor

    # clamp to [0,1]
    return min(1.0, max(0.0, risk))
