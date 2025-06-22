from typing import Any, Dict
from .review_llm    import review_fraud_score
from .visual_clip   import weighted_visual_risk
from .brand_match   import brand_mismatch
from .rules         import anomaly_score
from .scoring       import aggregate
from .embed_store   import EmbedDB

vecdb = EmbedDB()

def analyse_listing(listing: Dict[str, Any]) -> Dict[str, Any]:
    text_score, text_reason = review_fraud_score(listing["reviews"])
    vecdb.add(listing["reviews"])

    visual_score = weighted_visual_risk(listing["title"], listing["images"])

    brand_mismatch_flag = any(
        brand_mismatch(url, listing["title"])
        for url in listing["images"][:2]
    )
    if brand_mismatch_flag:
        visual_score = 1.0

    rule_score = anomaly_score(listing.get("ratings", []), listing.get("returns", 0))

    trust_score, verdict = aggregate(
        text_score,
        visual_score,
        rule_score,
        brand_mismatch_flag,
    )

    return {
        "asin":         listing.get("id", ""),
        "title":        listing.get("title", ""),
        "product_url":  listing.get("url", ""),
        "trust_score":  trust_score,
        "verdict":      verdict,
        "breakdown": {
            "text_score":     text_score,
            "visual_score":   visual_score,
            "brand_mismatch": brand_mismatch_flag,
            "rule_score":     rule_score,
        },
        "explanation": {
            "text": text_reason,
        },
    }
