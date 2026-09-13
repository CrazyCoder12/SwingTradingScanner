from .scoring import score_signal


def scan(df):
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        score = score_signal(row, prev)

        if score >= 6:
            signals.append(
                {
                    "date": df.index[i],
                    "close": row["close"],
                    "score": score,
                    "ema21": row["ema21"],
                    "ema50": row["ema50"],
                    "rsi": row["rsi"],
                }
            )

    return signals
