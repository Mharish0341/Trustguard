def anomaly_score(ratings: list[str], returns: int) -> float:
    hi = ratings.count("5") / max(len(ratings), 1)
    if hi > 0.9 and returns > 10:
        return 0.9
    if returns > 20:
        return 0.7
    return 0.1