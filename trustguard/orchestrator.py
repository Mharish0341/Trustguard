from .review_llm import review_fraud_score
from .visual_clip import worst_clip_score
from .brand_match import brand_mismatch
from .rules import anomaly_score
from .scoring import aggregate
from .embed_store import EmbedDB

vecdb = EmbedDB()

def analyse_listing(lst: dict):
    txt_s, txt_reason = review_fraud_score(lst["reviews"])
    vecdb.add(lst["reviews"])    # continually enrich DB (optional)

    vis_s = worst_clip_score(lst["title"], lst["images"])
    bm_flag = any(brand_mismatch(url, lst["title"]) for url in lst["images"][:2])
    if bm_flag:
        vis_s = 1.0

    rule_s = anomaly_score(lst["ratings"], lst["returns"])

    score, verdict = aggregate(txt_s, vis_s, rule_s)
    return {
        "id": lst["id"],
        "url": lst["url"],
        "trust_score": score,
        "verdict": verdict,
        "explain": {
            "text": txt_reason,
            "visual": f"clip_risk={vis_s:.2f} brand_flag={bm_flag}",
            "rule": rule_s,
        }
    }