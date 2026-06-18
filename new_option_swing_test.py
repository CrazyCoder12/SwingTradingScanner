"""
Options Swing Trading Signal Scanner  (Alpaca API)
===================================================
Designed for options swing trading — multi-indicator confluence required.

Indicators
----------
  EMA 9/21          Primary trend crossover
  EMA 50/200        Major trend filter (Scanner 2)
  MACD (12/26/9)    Trend + momentum confirmation
  RSI (14)          Entry zone oscillator (avoids chasing extended moves)
  Stochastic (14)   Second oscillator — %K/%D crossover in entry zone
  Bollinger Bands   Squeeze detection (volatility coil)
  ATR (14)          Options viability filter + stop/target sizing
  Volume            Breakout confirmation (volume > N-day avg)

Scanners
--------
  Scanner 1 — EMA 9/21 Crossover  (short-term momentum)
      Bullish : EMA-9 crosses ABOVE EMA-21  AND  close > EMA-9
      Bearish : EMA-9 crosses BELOW EMA-21  AND  close < EMA-9

  Scanner 2 — EMA 21/50 with EMA 9/200 filter  (major trend shift)
      Bullish : EMA-21 crosses ABOVE EMA-50  AND  EMA-9 > EMA-200
      Bearish : EMA-21 crosses BELOW EMA-50  AND  EMA-9 < EMA-200

Confirmation Filters (each adds +1 to confluence score, max 5)
---------------------------------------------------------------
  +1  MACD confirmed      MACD line aligned + histogram building
  +1  RSI entry zone      Not overbought/oversold, momentum present
  +1  Stochastic zone     %K/%D crossover in the right zone
  +1  BB squeeze active   Band width in bottom 20% of 1yr range
  +1  Volume confirmed    Volume > 20-day average (breakout is real)

  Score 3+ = Standard signal
  Score 5  = HIGH CONFLUENCE ★★★★★  (strongest options setups)

ATR options viability (not a score — acts as a hard filter if enabled):
  ATR% (ATR/close) must be above min_atr_pct to be worth trading options.
  Low ATR% = premium will be too cheap or theta will eat the position.

Data source : Alpaca Markets v2 daily bars (IEX feed, split-adjusted)
API keys    : loaded from .env  →  ALPACA_API_KEY, ALPACA_SECRET_KEY

Usage
-----
    python swing_scanner.py --start-date 2026-01-01
    python swing_scanner.py --start-date 2026-01-01 --symbol NVDA
    python swing_scanner.py --start-date 2026-01-01 --scanner 1
    python swing_scanner.py --start-date 2026-01-01 --scanner 2
    python swing_scanner.py --start-date 2026-01-01 --min-score 5      # high confluence only
    python swing_scanner.py --start-date 2026-01-01 --min-atr 1.5      # only stocks moving >1.5%/day
    python swing_scanner.py --start-date 2026-01-01 --no-atr-filter    # skip ATR viability gate
"""

import os
import requests
import argparse
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pandas as pd

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Alpaca credentials
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://data.alpaca.markets"

_HEADERS = {
    "APCA-API-KEY-ID":     API_KEY    or "",
    "APCA-API-SECRET-KEY": SECRET_KEY or "",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Watchlist
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "APP",
    "AVGO", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM",
    "TXN", "INTU", "NOW", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
    "PANW", "CRWD", "DDOG", "NET", "ZS", "SNOW", "PLTR", "TEAM",
    "WDAY", "ADSK", "UBER", "ABNB", "CRCL", "ZM", "RBLX", "ONON",
    "COIN", "PYPL", "MSTR", "APLD", "BIDU", "BABA", "FUTU", "SMCI",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "TTD", "BULL", "HIMS", "CELH",
    "CMG", "MCD", "SBUX", "NKE", "LULU", "LOW", "HD", "TGT", "WMT", "COST",
    "PG", "KO", "PEP", "PM",'HOOD',
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "GILD",
    "REGN", "VRTX", "ISRG", "BSX", "SYK", "MDT", "ZTS",
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "C", "SCHW", "AXP",
    "COF", "BX", "KKR",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
    "CAT", "BA", "HON", "UNP", "RTX", "GE", "LMT", "DE", "UPS", "FDX", "UPST",
    "ETN", "GD", "NOC", "CSX",
    "LIN", "APD", "SHW", "NEM", "FCX", "NUE",
    "AMT", "PLD", "EQIX", "PSA", "O",
    "NEE", "DUK", "SO",
    "TSM", "ASML", "TER", "ENTG", "MPWR",
    "RIVN", "ENPH", "FSLR",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

EMA_FAST   = 9
EMA_SLOW   = 21
EMA_MID    = 50
EMA_TREND  = 200

MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

RSI_PERIOD    = 14
RSI_BULL_LOW  = 40   # RSI must be above this — trend momentum present
RSI_BULL_HIGH = 65   # RSI must be below this — not overbought
RSI_BEAR_LOW  = 35   # RSI must be above this — not washed out
RSI_BEAR_HIGH = 60   # RSI must be below this — trend momentum present

STOCH_K      = 14    # %K period
STOCH_D      = 3     # %D smoothing
STOCH_SMOOTH = 3     # %K smoothing (slow stochastic)
STOCH_BULL_ZONE = 50  # %K must cross above %D below this level for bullish
STOCH_BEAR_ZONE = 50  # %K must cross below %D above this level for bearish

BB_PERIOD        = 20
BB_STD           = 2.0
BB_SQUEEZE_PCT   = 20   # squeeze = width in bottom N% of 1yr range

ATR_PERIOD       = 14
ATR_MIN_PCT      = 1.5  # default: stock must move >1.5% per day (ATR/close)
                         # low ATR = options premium won't be worthwhile

VOL_MA_PERIOD    = 20   # volume must be above this N-day average

WARMUP_DAYS = 300   # extra calendar days before start_date for indicator warmup

# ─────────────────────────────────────────────────────────────────────────────
#  Alpaca data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_daily_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    fetch_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    fetch_end   = end_dt.strftime("%Y-%m-%d")

    url    = f"{BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe":  "1Day",
        "start":      fetch_start,
        "end":        fetch_end,
        "feed":       "iex",
        "adjustment": "split",
        "limit":      10000,
    }
    all_bars, next_token = [], None

    while True:
        if next_token:
            params["page_token"] = next_token
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
            resp.raise_for_status()
        except Exception:
            return pd.DataFrame()

        data       = resp.json()
        all_bars.extend(data.get("bars", []))
        next_token = data.get("next_page_token")
        if not next_token:
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize()
    df = (
        df.rename(columns={"t": "date", "o": "open", "h": "high",
                            "l": "low",  "c": "close", "v": "volume"})
          .set_index("date")
          .sort_index()
    )
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")]
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Indicator computation
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    for span in (EMA_FAST, EMA_SLOW, EMA_MID, EMA_TREND):
        df[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema_f          = close.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_s          = close.ewm(span=MACD_SLOW,   adjust=False).mean()
    df["macd"]     = ema_f - ema_s
    df["macd_sig"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_h"]   = df["macd"] - df["macd_sig"]

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    al    = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + ag / al.replace(0, float("nan"))))

    # ── Stochastic (slow) ─────────────────────────────────────────────────────
    low_n  = low.rolling(STOCH_K).min()
    high_n = high.rolling(STOCH_K).max()
    raw_k  = 100 * (close - low_n) / (high_n - low_n).replace(0, float("nan"))
    df["stoch_k"] = raw_k.rolling(STOCH_SMOOTH).mean()   # smoothed %K
    df["stoch_d"] = df["stoch_k"].rolling(STOCH_D).mean()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df["bb_mid"]   = close.rolling(BB_PERIOD).mean()
    bb_std         = close.rolling(BB_PERIOD).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_wpct"]  = df["bb_width"].rolling(252, min_periods=50).rank(pct=True) * 100

    # ── ATR ───────────────────────────────────────────────────────────────────
    prev_close      = close.shift(1)
    tr              = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(com=ATR_PERIOD - 1, adjust=False).mean()
    df["atr_pct"] = df["atr"] / close * 100   # ATR as % of price (options viability)

    # ── Volume MA ─────────────────────────────────────────────────────────────
    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Per-bar confirmation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _macd_bull(prev, cur) -> bool:
    return cur["macd"] > cur["macd_sig"] and cur["macd_h"] > prev["macd_h"]

def _macd_bear(prev, cur) -> bool:
    return cur["macd"] < cur["macd_sig"] and cur["macd_h"] < prev["macd_h"]

def _rsi_bull(cur) -> bool:
    return RSI_BULL_LOW <= cur["rsi"] <= RSI_BULL_HIGH

def _rsi_bear(cur) -> bool:
    return RSI_BEAR_LOW <= cur["rsi"] <= RSI_BEAR_HIGH

def _stoch_bull(prev, cur) -> bool:
    """Slow %K crosses above %D while both are below STOCH_BULL_ZONE (not overbought)."""
    cross = prev["stoch_k"] <= prev["stoch_d"] and cur["stoch_k"] > cur["stoch_d"]
    zone  = cur["stoch_k"] < STOCH_BULL_ZONE and cur["stoch_d"] < STOCH_BULL_ZONE
    return cross and zone

def _stoch_bear(prev, cur) -> bool:
    """Slow %K crosses below %D while both are above STOCH_BEAR_ZONE (not oversold)."""
    cross = prev["stoch_k"] >= prev["stoch_d"] and cur["stoch_k"] < cur["stoch_d"]
    zone  = cur["stoch_k"] > STOCH_BEAR_ZONE and cur["stoch_d"] > STOCH_BEAR_ZONE
    return cross and zone

def _squeeze(cur) -> bool:
    pct = cur.get("bb_wpct", float("nan"))
    return not pd.isna(pct) and pct <= BB_SQUEEZE_PCT

def _vol_confirmed(cur) -> bool:
    """Volume spike above the N-day moving average."""
    vm = cur.get("vol_ma", float("nan"))
    return not pd.isna(vm) and cur["volume"] > vm

def _atr_viable(cur, min_atr_pct: float) -> bool:
    """ATR% check — is this stock moving enough to justify buying options premium?"""
    ap = cur.get("atr_pct", float("nan"))
    return not pd.isna(ap) and ap >= min_atr_pct

def _score(macd_ok, rsi_ok, stoch_ok, squeeze_ok, vol_ok) -> int:
    return sum([macd_ok, rsi_ok, stoch_ok, squeeze_ok, vol_ok])

def _build_row(direction, date, cur, prev, e9, e21,
               macd_ok, rsi_ok, stoch_ok, squeeze_ok, vol_ok,
               extra_emas=None):
    score = _score(macd_ok, rsi_ok, stoch_ok, squeeze_ok, vol_ok)
    row = {
        "date":    date.strftime("%Y-%m-%d"),
        "signal":  direction,
        "score":   score,
        "close":   round(float(cur["close"]), 3),
        f"ema{EMA_FAST}": round(e9,  3),
        f"ema{EMA_SLOW}": round(e21, 3),
        "macd":    round(float(cur["macd"]),   4),
        "macd_h":  round(float(cur["macd_h"]), 4),
        "rsi":     round(float(cur["rsi"]),    1),
        "stk":     round(float(cur["stoch_k"]),1),
        "std":     round(float(cur["stoch_d"]),1),
        "bb_wpct": round(float(cur["bb_wpct"]),1) if not pd.isna(cur["bb_wpct"]) else "—",
        "atr_pct": round(float(cur["atr_pct"]),2),
        "vol_ratio": round(float(cur["volume"]) / float(cur["vol_ma"]), 2) if not pd.isna(cur["vol_ma"]) else "—",
        "macd_ok":  "Y" if macd_ok  else "n",
        "rsi_ok":   "Y" if rsi_ok   else "n",
        "stoch_ok": "Y" if stoch_ok else "n",
        "sqz_ok":   "Y" if squeeze_ok else "n",
        "vol_ok":   "Y" if vol_ok   else "n",
    }
    if extra_emas:
        for k, v in extra_emas.items():
            row[k] = round(v, 3)
    return row


# ─────────────────────────────────────────────────────────────────────────────
#  Scanner 1 — EMA 9/21 crossover
# ─────────────────────────────────────────────────────────────────────────────

def get_signals_s1(df, start_dt, min_score, min_atr_pct, atr_filter):
    signals, sim_mask = [], df.index >= pd.Timestamp(start_dt)
    if not sim_mask.any():
        return signals
    start_pos = df.index.get_loc(df[sim_mask].index[0])

    for i in range(start_pos, len(df)):
        if i < 1:
            continue
        prev, cur = df.iloc[i - 1], df.iloc[i]
        date  = df.index[i]
        e9    = float(cur[f"ema{EMA_FAST}"])
        e21   = float(cur[f"ema{EMA_SLOW}"])
        close = float(cur["close"])

        if atr_filter and not _atr_viable(cur, min_atr_pct):
            continue

        bull_cross = prev[f"ema{EMA_FAST}"] <= prev[f"ema{EMA_SLOW}"] and e9 > e21
        bear_cross = prev[f"ema{EMA_FAST}"] >= prev[f"ema{EMA_SLOW}"] and e9 < e21

        if bull_cross and close > e9:
            mk, rk = _macd_bull(prev, cur), _rsi_bull(cur)
            sk, sq, vk = _stoch_bull(prev, cur), _squeeze(cur), _vol_confirmed(cur)
            if _score(mk, rk, sk, sq, vk) >= min_score:
                row = _build_row("BULLISH", date, cur, prev, e9, e21, mk, rk, sk, sq, vk)
                row["symbol"] = None
                signals.append(row)

        elif bear_cross and close < e9:
            mk, rk = _macd_bear(prev, cur), _rsi_bear(cur)
            sk, sq, vk = _stoch_bear(prev, cur), _squeeze(cur), _vol_confirmed(cur)
            if _score(mk, rk, sk, sq, vk) >= min_score:
                row = _build_row("BEARISH", date, cur, prev, e9, e21, mk, rk, sk, sq, vk)
                row["symbol"] = None
                signals.append(row)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  Scanner 2 — EMA 21/50 crossing, EMA 9/200 filter
# ─────────────────────────────────────────────────────────────────────────────

def get_signals_s2(df, start_dt, min_score, min_atr_pct, atr_filter):
    signals, sim_mask = [], df.index >= pd.Timestamp(start_dt)
    if not sim_mask.any():
        return signals
    start_pos = df.index.get_loc(df[sim_mask].index[0])

    for i in range(start_pos, len(df)):
        if i < 1:
            continue
        prev, cur = df.iloc[i - 1], df.iloc[i]
        date  = df.index[i]
        e9    = float(cur[f"ema{EMA_FAST}"])
        e21   = float(cur[f"ema{EMA_SLOW}"])
        e50   = float(cur[f"ema{EMA_MID}"])
        e200  = float(cur[f"ema{EMA_TREND}"])
        close = float(cur["close"])

        if atr_filter and not _atr_viable(cur, min_atr_pct):
            continue

        bull_cross = prev[f"ema{EMA_SLOW}"] <= prev[f"ema{EMA_MID}"] and e21 > e50
        bear_cross = prev[f"ema{EMA_SLOW}"] >= prev[f"ema{EMA_MID}"] and e21 < e50

        if bull_cross and e9 > e200:
            mk, rk = _macd_bull(prev, cur), _rsi_bull(cur)
            sk, sq, vk = _stoch_bull(prev, cur), _squeeze(cur), _vol_confirmed(cur)
            if _score(mk, rk, sk, sq, vk) >= min_score:
                row = _build_row("BULLISH", date, cur, prev, e9, e21, mk, rk, sk, sq, vk,
                                 {f"ema{EMA_MID}": e50, f"ema{EMA_TREND}": e200})
                row["symbol"] = None
                signals.append(row)

        elif bear_cross and e9 < e200:
            mk, rk = _macd_bear(prev, cur), _rsi_bear(cur)
            sk, sq, vk = _stoch_bear(prev, cur), _squeeze(cur), _vol_confirmed(cur)
            if _score(mk, rk, sk, sq, vk) >= min_score:
                row = _build_row("BEARISH", date, cur, prev, e9, e21, mk, rk, sk, sq, vk,
                                 {f"ema{EMA_MID}": e50, f"ema{EMA_TREND}": e200})
                row["symbol"] = None
                signals.append(row)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_W = 118

def _header(title):
    print(f"\n{'='*_W}\n  {title}\n{'='*_W}")

def _score_label(score):
    return {5: "★★★★★ MAX", 4: "★★★★  HIGH", 3: "★★★   MED",
            2: "★★    LOW", 1: "★     MIN", 0: "      NONE"}.get(score, "?")

def _confirms(r):
    return (f"MACD:{r['macd_ok']}  RSI:{r['rsi_ok']}  "
            f"STCH:{r['stoch_ok']}  SQZ:{r['sqz_ok']}  VOL:{r['vol_ok']}")

def _table(rows, direction, scanner):
    emoji = "🟢" if direction == "BULLISH" else "🔴"
    if scanner == 1:
        label = f"EMA-{EMA_FAST} crossed {'ABOVE' if direction=='BULLISH' else 'BELOW'} EMA-{EMA_SLOW}"
    else:
        label = f"EMA-{EMA_SLOW} crossed {'ABOVE' if direction=='BULLISH' else 'BELOW'} EMA-{EMA_MID}"
    print(f"\n{emoji} SCANNER {scanner} — {direction}  ({label})")
    if not rows:
        print("  No signals.")
        return

    hdr = (f"  {'Symbol':<8}  {'Date':<12}  {'Score':<12}  {'Close':>9}  "
           f"{'EMA9':>9}  {'EMA21':>9}  "
           f"{'MACD':>8}  {'Hist':>8}  {'RSI':>6}  {'Stch%K':>7}  "
           f"{'BB W%':>6}  {'ATR%':>5}  {'VolR':>5}  Confirms")
    sep = (f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*9}  "
           f"{'-'*9}  {'-'*9}  "
           f"{'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  "
           f"{'-'*6}  {'-'*5}  {'-'*5}  {'-'*38}")
    print(hdr); print(sep)
    for r in rows:
        print(
            f"  {r['symbol']:<8}  {r['date']:<12}  {_score_label(r['score']):<12}  "
            f"${r['close']:>8.3f}  "
            f"${r[f'ema{EMA_FAST}']:>8.3f}  ${r[f'ema{EMA_SLOW}']:>8.3f}  "
            f"{r['macd']:>8.4f}  {r['macd_h']:>8.4f}  "
            f"{r['rsi']:>6.1f}  {r['stk']:>7.1f}  "
            f"{str(r['bb_wpct']):>6}  {r['atr_pct']:>5.2f}  {str(r['vol_ratio']):>5}  "
            f"{_confirms(r)}"
        )
    high = sum(1 for r in rows if r["score"] >= 4)
    print(f"\n  Total: {len(rows)} signal(s)   ★★★★+ High confluence: {high}")


def _print_legend(min_atr_pct, atr_filter):
    print(f"\n{'─'*_W}")
    print("  INDICATOR LEGEND")
    print(f"{'─'*_W}")
    print(f"  EMA {EMA_FAST}/{EMA_SLOW}      Primary crossover trigger (S1) / fast cluster (S2)")
    print(f"  EMA {EMA_MID}/{EMA_TREND}    Major trend filter (S2 only)")
    print(f"  MACD ({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL})  Line vs Signal + histogram direction  (+1)")
    print(f"  RSI ({RSI_PERIOD})       Bull zone {RSI_BULL_LOW}–{RSI_BULL_HIGH}  |  Bear zone {RSI_BEAR_LOW}–{RSI_BEAR_HIGH}  (+1)")
    print(f"  Stoch ({STOCH_K})    Slow %K/%D crossover in non-extreme zone  (+1)")
    print(f"  BB Width%    Squeeze = bottom {BB_SQUEEZE_PCT}% of 1yr range (volatility coil)  (+1)")
    print(f"  VolRatio     Volume / {VOL_MA_PERIOD}-day avg — breakout confirmation  (+1)")
    print(f"  ATR%         ATR({ATR_PERIOD}) / close — options viability (move size relative to price)")
    if atr_filter:
        print(f"               Filter ON: signals require ATR% >= {min_atr_pct}%  (enough daily move to trade options)")
    else:
        print(f"               Filter OFF: ATR% shown for reference only")
    print(f"  Score        Max 5  (★★★★★ = all 5 confirmed — strongest options setups)")
    print(f"{'─'*_W}")


# ─────────────────────────────────────────────────────────────────────────────
#  Core scanner loop
# ─────────────────────────────────────────────────────────────────────────────

def run_scanner(symbols, start_date_str, run_s1, run_s2,
                min_score, min_atr_pct, atr_filter):
    start_dt  = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt    = datetime.now()
    today_str = end_dt.strftime("%Y-%m-%d")

    _header(
        f"OPTIONS SWING TRADING SCANNER  |  {start_date_str} → {today_str}  "
        f"|  min score: {min_score}/5  |  ATR filter: {'≥'+str(min_atr_pct)+'%' if atr_filter else 'OFF'}"
    )
    _print_legend(min_atr_pct, atr_filter)

    s1_bull, s1_bear, s2_bull, s2_bear = [], [], [], []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] {sym:<6} ...", end="  ", flush=True)
        df = fetch_daily_bars(sym, start_dt, end_dt)
        if df.empty or len(df) < 80:
            print("no data"); continue
        df = add_indicators(df)
        parts = []

        if run_s1:
            sigs = get_signals_s1(df, start_dt, min_score, min_atr_pct, atr_filter)
            for s in sigs: s["symbol"] = sym
            bull = [s for s in sigs if s["signal"] == "BULLISH"]
            bear = [s for s in sigs if s["signal"] == "BEARISH"]
            s1_bull.extend(bull); s1_bear.extend(bear)
            if sigs:
                h = sum(1 for s in sigs if s["score"] >= 4)
                parts.append(f"S1: 🟢{len(bull)} 🔴{len(bear)} ★★★★+{h}")

        if run_s2:
            sigs = get_signals_s2(df, start_dt, min_score, min_atr_pct, atr_filter)
            for s in sigs: s["symbol"] = sym
            bull = [s for s in sigs if s["signal"] == "BULLISH"]
            bear = [s for s in sigs if s["signal"] == "BEARISH"]
            s2_bull.extend(bull); s2_bear.extend(bear)
            if sigs:
                h = sum(1 for s in sigs if s["score"] >= 4)
                parts.append(f"S2: 🟢{len(bull)} 🔴{len(bear)} ★★★★+{h}")

        print("  |  ".join(parts) if parts else "no signals")

    key = lambda r: (r["date"], r["symbol"])
    for lst in (s1_bull, s1_bear, s2_bull, s2_bear):
        lst.sort(key=key)

    print(f"\n{'─'*_W}\n  RESULTS  ({start_date_str} → {today_str})\n{'─'*_W}")

    if run_s1:
        print(f"\n{'─'*_W}\n  SCANNER 1 — EMA {EMA_FAST}/{EMA_SLOW} Crossover\n{'─'*_W}")
        _table(s1_bull, "BULLISH", 1)
        _table(s1_bear, "BEARISH", 1)

    if run_s2:
        print(f"\n{'─'*_W}\n  SCANNER 2 — EMA {EMA_SLOW}/{EMA_MID} x EMA {EMA_TREND}\n{'─'*_W}")
        _table(s2_bull, "BULLISH", 2)
        _table(s2_bear, "BEARISH", 2)

    print(f"\n{'='*_W}\n  SUMMARY\n{'='*_W}")
    if run_s1:
        h1 = sum(1 for r in s1_bull+s1_bear if r["score"] >= 4)
        print(f"  Scanner 1 (EMA {EMA_FAST}/{EMA_SLOW})      :  "
              f"{len(s1_bull)+len(s1_bear):>4} signals  🟢 {len(s1_bull)}  🔴 {len(s1_bear)}  ★★★★+ {h1}")
    if run_s2:
        h2 = sum(1 for r in s2_bull+s2_bear if r["score"] >= 4)
        print(f"  Scanner 2 (EMA {EMA_SLOW}/{EMA_MID} x {EMA_TREND})  :  "
              f"{len(s2_bull)+len(s2_bear):>4} signals  🟢 {len(s2_bull)}  🔴 {len(s2_bear)}  ★★★★+ {h2}")
    print(f"{'='*_W}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Single-symbol mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single(symbol, start_date_str, run_s1, run_s2,
               min_score, min_atr_pct, atr_filter):
    start_dt  = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt    = datetime.now()
    today_str = end_dt.strftime("%Y-%m-%d")

    _header(f"OPTIONS SWING SCAN: {symbol}  |  {start_date_str} → {today_str}")
    _print_legend(min_atr_pct, atr_filter)

    df = fetch_daily_bars(symbol, start_dt, end_dt)
    if df.empty or len(df) < 80:
        print("  No data available.\n"); return

    df = add_indicators(df)

    if run_s1:
        sigs = get_signals_s1(df, start_dt, min_score, min_atr_pct, atr_filter)
        for s in sigs: s["symbol"] = symbol
        _table([s for s in sigs if s["signal"] == "BULLISH"], "BULLISH", 1)
        _table([s for s in sigs if s["signal"] == "BEARISH"], "BEARISH", 1)

    if run_s2:
        sigs = get_signals_s2(df, start_dt, min_score, min_atr_pct, atr_filter)
        for s in sigs: s["symbol"] = symbol
        _table([s for s in sigs if s["signal"] == "BULLISH"], "BULLISH", 2)
        _table([s for s in sigs if s["signal"] == "BEARISH"], "BEARISH", 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        print("\n[ERROR] ALPACA_API_KEY / ALPACA_SECRET_KEY not found in .env\n")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="Options Swing Trading Signal Scanner — Alpaca API"
    )
    parser.add_argument("--start-date",  required=True,
                        help="Scan start date  YYYY-MM-DD")
    parser.add_argument("--symbol",      default=None,
                        help="Single symbol (skips full watchlist)")
    parser.add_argument("--scanner",     type=int, choices=[1, 2], default=None,
                        help="Run only Scanner 1 or 2 (default: both)")
    parser.add_argument("--min-score",   type=int, default=2, choices=range(6),
                        help="Minimum confluence score 0–5 (default 2; use 4-5 for best setups)")
    parser.add_argument("--min-atr",     type=float, default=ATR_MIN_PCT,
                        help=f"Min ATR%% for options viability (default {ATR_MIN_PCT})")
    parser.add_argument("--no-atr-filter", action="store_true",
                        help="Disable ATR viability gate (show all signals regardless of move size)")
    args = parser.parse_args()

    run_s1      = args.scanner in (None, 1)
    run_s2      = args.scanner in (None, 2)
    atr_filter  = not args.no_atr_filter

    if args.symbol:
        run_single(args.symbol.upper(), args.start_date,
                   run_s1, run_s2, args.min_score, args.min_atr, atr_filter)
    else:
        run_scanner(WATCHLIST, args.start_date,
                    run_s1, run_s2, args.min_score, args.min_atr, atr_filter)