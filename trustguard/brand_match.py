import functools
import io
import re
from typing import Optional
import numpy as np
import requests
from PIL import Image
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
from transformers import pipeline

_paddle = PaddleOCR(lang="en")

_flan_extractor = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=-1
)

@functools.lru_cache(maxsize=512)
def _download(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=10)
        return r.content if r.ok else None
    except:
        return None

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _ocr_with_paddle(raw: bytes) -> str:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)
    try:
        results = _paddle.ocr(arr, det=True, rec=True)
    except:
        return ""
    lines = []
    for item in results:
        if isinstance(item, dict):
            txt = item.get("text") or item.get("ocr_text") or ""
        else:
            rec = item[1]
            if isinstance(rec, (list, tuple)):
                txt = rec[0]
            else:
                txt = rec
        if txt:
            lines.append(txt)
    return " ".join(lines)

def _extract_brand_from_text(text: str) -> str:
    if not text.strip():
        return ""
    prompt = (
        "Extract the brand name from this product image OCR text.\n"
        "Respond with ONLY the brand, in lowercase, no extra words.\n\n"
        f"OCR text: {text}"
    )
    out = _flan_extractor(prompt, max_new_tokens=8, do_sample=False)
    gen = out[0]["generated_text"]
    if ":" in gen:
        gen = gen.split(":", 1)[1]
    return gen.strip().lower()

def brand_mismatch(image_url: str, title: str, threshold: float = 0.80) -> bool:
    title_brand = _normalize(title.split()[0] if title else "")
    if not title_brand:
        return False
    raw = _download(image_url)
    if not raw:
        return False
    ocr_text = _ocr_with_paddle(raw)
    ocr_brand = _normalize(ocr_text)
    if not ocr_brand:
        extracted = _extract_brand_from_text(ocr_text)
        ocr_brand = _normalize(extracted)
    if not ocr_brand:
        return False
    sim = _fuzzy_ratio(title_brand, ocr_brand)
    return sim < threshold
