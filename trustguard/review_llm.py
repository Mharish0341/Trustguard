import inspect
import json
import re
import textwrap
import time
from functools import lru_cache
from typing import Dict

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from .config import GOOGLE_API_KEY, LLM_MODEL

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(LLM_MODEL)

# ---- Rate-limit settings (adjust to your plan) ----
MAX_CALLS_PER_MIN = 15
MIN_INTERVAL    = 60.0 / MAX_CALLS_PER_MIN
_last_call_time = 0.0

# ---- JSON support detection ----
SUPPORTS_JSON_FLAG = "response_format" in inspect.signature(
    model.generate_content
).parameters


@lru_cache(maxsize=1024)
def _query_llm(prompt: str) -> Dict:
    global _last_call_time

    # 1) throttle to MIN_INTERVAL between calls
    since = time.time() - _last_call_time
    if since < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - since)

    # 2) attempt + one retry on 429
    for attempt in (1, 2):
        try:
            if SUPPORTS_JSON_FLAG:
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0, "max_output_tokens": 64},
                    response_format={"type": "json_object"},
                )
                raw = resp.text
            else:
                safe = (
                    "Return only JSON inside triple backticks.\n" + prompt
                )
                resp = model.generate_content(
                    safe,
                    generation_config={"temperature": 0.0, "max_output_tokens": 64},
                )
                raw = resp.text
            _last_call_time = time.time()
            break

        except ResourceExhausted as e:
            # backoff per API’s suggested delay
            retry = getattr(e, "retry_delay", None)
            wait  = retry.seconds if retry and hasattr(retry, "seconds") else 60
            print(f"[review_llm] Rate limit hit, sleeping for {wait}s…")
            time.sleep(wait)

    else:
        # both attempts failed → neutral fallback
        return {"score": 0.5, "why": "rate‐limit fallback"}

    # 3) parse JSON out of raw (first {...})
    text = raw.strip()
    try:
        # if JSON‐flag used
        return json.loads(text)
    except Exception:
        # regex‐find the first { … }
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    # 4) last‐ditch fallback parsing SCORE/WHY
    m = re.search(r"SCORE:\s*([0-9.]+)", text)
    return {
        "score": float(m.group(1)) if m else 0.5,
        "why":   text.split("WHY:", 1)[-1].strip()[:200]
                if "WHY:" in text else "LLM unparsable fallback"
    }


def review_fraud_score(reviews: list[str], sample: int = 20) -> tuple[float, str]:
    """Returns (score 0–1, short reason) for a list of review strings."""
    snippet = "\n".join(reviews[:sample])
    base_prompt = textwrap.dedent(f"""
        ROLE: Marketplace-trust analyst.
        TASK: Return ONLY a JSON object with keys:
          • score  (float 0–1) – likelihood reviews are fake
          • why    (≤20 words) – short rationale

        REVIEWS:
        {snippet}
    """)
    obj = _query_llm(base_prompt)
    return float(obj.get("score", 0.5)), str(obj.get("why", "no reason"))[:200]
