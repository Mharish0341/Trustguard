def aggregate(text_s: float, visual_s: float, rule_s: float):
    trust = (1 - text_s)*0.4 + (1 - visual_s)*0.4 + (1 - rule_s)*0.2
    score = int(trust * 100)
    verdict = "Pass" if score >= 80 else "Flag" if score >= 40 else "Manual"
    return score, verdict