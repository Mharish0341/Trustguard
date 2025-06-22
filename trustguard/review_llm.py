import inspect
import json
import re
import textwrap
import time
from functools import lru_cache
from typing import Dict, List, Any, Union, Tuple

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from .config import GOOGLE_API_KEY, LLM_MODEL

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(LLM_MODEL)

MAX_CALLS_PER_MIN = 15
MIN_INTERVAL      = 60.0 / MAX_CALLS_PER_MIN
_last_call_time   = 0.0

@lru_cache(maxsize=1024)
def _query_llm(prompt: str) -> Dict[str, Any]:
    global _last_call_time

    # throttle between calls
    since = time.time() - _last_call_time
    if since < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - since)

    # try up to two times (back off on 429)
    for _ in (1, 2):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 64},
            )
            raw = resp.text
            _last_call_time = time.time()
            break
        except ResourceExhausted as e:
            retry = getattr(e, "retry_delay", None)
            wait  = retry.seconds if retry and hasattr(retry, "seconds") else 60
            time.sleep(wait)
    else:
        return {"score": 0.5, "why": "rate-limit fallback"}

    text = raw.strip()
    # first attempt: parse entire response as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: find first {...}
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    # last‐ditch: extract SCORE/WHY labeled portions
    m = re.search(r"SCORE:\s*([0-9.]+)", text)
    return {
        "score": float(m.group(1)) if m else 0.5,
        "why":   text.split("WHY:", 1)[-1].strip()[:200]
                 if "WHY:" in text else "unparsable fallback"
    }

def review_fraud_score(
    reviews: Union[List[Dict[str, Any]], List[str]],
    sample: int = 20
) -> Tuple[float, str]:
    # build a list of up to `sample` plain‐text review bodies
    texts: List[str] = []
    for r in reviews:
        if isinstance(r, dict):
            body = r.get("body") or r.get("text") or ""
            if isinstance(body, str) and body.strip():
                texts.append(body.strip())
        elif isinstance(r, str) and r.strip():
            texts.append(r.strip())
        if len(texts) >= sample:
            break

    snippet = "\n".join(texts)
    prompt = textwrap.dedent(f"""
        You are a fraud-detection specialist for an online marketplace.
        Given the following customer reviews, return ONLY a JSON object with:
          • "score": a number between 0.0 and 1.0 indicating the likelihood
                     the reviews are fake (0 = genuine, 1 = definitely fake)
          • "why":   a brief (≤20 words) explanation of your key signal

        Reviews:
        {snippet}
    """.strip())

    obj = _query_llm(prompt)
    score = float(obj.get("score", 0.5))
    reason = str(obj.get("why", "no reason"))[:200]
    return score, reason
