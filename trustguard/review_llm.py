from __future__ import annotations
import inspect
import json
import re
import textwrap
import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from .config import GOOGLE_API_KEY, LLM_MODEL

genai.configure(api_key=GOOGLE_API_KEY)
_model = genai.GenerativeModel(LLM_MODEL)

_MAX_RPM      = 15
_MIN_INTERVAL = 60.0 / _MAX_RPM
_last_call    = 0.0


@lru_cache(maxsize=512)
def _query_llm(prompt: str) -> Dict[str, Any]:
    global _last_call
    wait = _MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)

    for _ in range(2):
        try:
            resp = _model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 64},
            )
            _last_call = time.time()
            raw = resp.text.strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                block = re.search(r"\{.*?\}", raw, re.S)
                return json.loads(block.group(0)) if block else {}
        except ResourceExhausted as e:
            back = getattr(e, "retry_delay", None)
            time.sleep(back.seconds if back and hasattr(back, "seconds") else 60)

    return {"score": 0.5, "why": "rate-limit fallback"}


def _bodies(rev: Union[List[Dict[str, Any]], List[str]], cap: int = 20) -> List[str]:
    out: List[str] = []
    for r in rev:
        body = r.get("body") if isinstance(r, dict) else r
        if isinstance(body, str) and body.strip():
            out.append(body.strip())
        if len(out) >= cap:
            break
    return out


def review_fraud_score(
    reviews: Union[List[Dict[str, Any]], List[str]], sample: int = 20
) -> Tuple[float, str]:
    texts = _bodies(reviews, sample)
    n = len(texts)

    if n == 0:
        return 0.8, "no reviews"
    if n < 3:
        return 0.6, "few reviews"

    snippet = "\n".join(texts)
    prompt = f"""
You are a fraud-detection analyst for an e-commerce marketplace.

Return ONLY one JSON object with two keys:
  "score": float from 0.0 (certainly genuine) to 1.0 (certainly fake)
  "why":   ≤ 20 words explaining the MAIN signal used.

Special cases
-------------
• If no review text is provided, set "score": 0.8 and "why": "no reviews".
• If fewer than 3 distinct reviews are given, set "score": 0.6 and "why": "few reviews".

Otherwise
---------
Analyse linguistic patterns, repetitiveness and sentiment consistency.
Be strict about identical or templated language across reviewers.

Reviews:
{snippet}
""".strip()

    obj    = _query_llm(prompt)
    score  = float(obj.get("score", 0.5))
    reason = str(obj.get("why",  "no reason"))[:200]
    return score, reason
