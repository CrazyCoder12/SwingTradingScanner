from datetime import datetime, timedelta
from data import fetch_bars
from indicators import add_indicators

def get_regime():
    end = datetime.now()
    start = end - timedelta(days=200)

    spy = add_indicators(fetch_bars("SPY", start, end))
    qqq = add_indicators(fetch_bars("QQQ", start, end))

    if spy.empty or qqq.empty:
        return True

    return (
        spy["close"].iloc[-1] > spy["ema50"].iloc[-1]
        and qqq["close"].iloc[-1] > qqq["ema50"].iloc[-1]
    )