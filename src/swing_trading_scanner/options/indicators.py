import pandas as pd
from .config import EMA_FAST, EMA_SLOW, EMA_TREND, RSI_PERIOD, ATR_PERIOD


def add_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(span=EMA_FAST).mean()
    df["ema21"] = df["close"].ewm(span=EMA_SLOW).mean()
    df["ema50"] = df["close"].ewm(span=EMA_TREND).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["vol_avg"] = df["volume"].rolling(20).mean()

    return df
