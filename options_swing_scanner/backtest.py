def backtest(signals, df):
    results = []

    for sig in signals:
        if sig["date"] not in df.index:
            continue

        idx = df.index.get_loc(sig["date"])
        entry = df.iloc[idx]["close"]

        atr = df.iloc[idx]["atr"]
        stop = entry - 1.5 * atr
        target = entry + 2 * atr

        outcome = 0

        for j in range(idx+1, min(idx+15, len(df))):
            if df.iloc[j]["low"] <= stop:
                outcome = -1
                break
            if df.iloc[j]["high"] >= target:
                outcome = 1
                break

        results.append(outcome)

    if not results:
        return {"trades": 0, "win_rate": 0}

    win_rate = len([r for r in results if r == 1]) / len(results)

    return {"trades": len(results), "win_rate": win_rate}