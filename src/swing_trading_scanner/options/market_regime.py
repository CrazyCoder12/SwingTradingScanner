"""Report whether both benchmark ETFs are above their 50-day EMAs."""

from datetime import datetime, timedelta

from .data import fetch_bars
from .indicators import add_indicators


def get_regime():
    end = datetime.now()
    start = end - timedelta(days=200)
    benchmarks = [fetch_bars(symbol, start, end) for symbol in ("SPY", "QQQ")]
    if any(df.empty for df in benchmarks):
        return None
    return all(
        bool(df["close"].iloc[-1] > df["ema50"].iloc[-1])
        for df in map(add_indicators, benchmarks)
    )
