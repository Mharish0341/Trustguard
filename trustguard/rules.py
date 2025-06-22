def anomaly_score(ratings: list[float], returns: int) -> float:
    n = len(ratings)
    if n == 0:
        return 0.1
    high_ratio = sum(1 for r in ratings if r >= 4.5) / n
    low_ratio  = sum(1 for r in ratings if r <= 1.5) / n
    if low_ratio > 0.3:
        return 0.9
    if high_ratio > 0.9 and returns > 10:
        return 0.9
    if returns > 20:
        return 0.7
    return 0.1
