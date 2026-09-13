def score_signal(row, prev):
    score = 0

    if row["ema9"] > row["ema21"] > row["ema50"]:
        score += 2

    if row["low"] <= row["ema21"] * 1.01:
        score += 2

    if 45 <= row["rsi"] <= 70:
        score += 1

    if row["volume"] > row["vol_avg"]:
        score += 1

    if row["close"] > prev["close"]:
        score += 1

    if row["close"] > row["ema50"]:
        score += 1

    return score
