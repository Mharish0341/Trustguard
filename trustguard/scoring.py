def aggregate(text_s: float,visual_s: float,rule_s: float,brand_flag: bool,threshold: int = 70,) -> tuple[int, bool]:
    # weights must sum to 1.0
    w_text, w_visual, w_rule, w_brand = 0.20, 0.20, 0.20, 0.40
    risk_brand = 1.0 if brand_flag else 0.0
    trust_frac = ((1.0 - text_s)   * w_text   +(1.0 - visual_s) * w_visual +(1.0 - rule_s) * w_rule +(1.0 - risk_brand) * w_brand)
    score   = int(trust_frac * 100)
    listable = score >= threshold
    return score, listable
