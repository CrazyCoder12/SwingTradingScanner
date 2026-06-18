"""
9/21 EMA Pullback Options Swing Scanner
Strategy: Pullback to 21 EMA after a 9/21 EMA crossover, confirmed by volume + SPY trend.
Usage:
    python ema_swing_scanner.py                        # scan today
    python ema_swing_scanner.py --from-date 2025-01-01 # scan from a date onward
    python ema_swing_scanner.py --tickers AAPL MSFT NVDA  # custom tickers
"""

import argparse
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Default watchlist ────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
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

# ─── EMA helpers ─────────────────────────────────────────────────────────────

def compute_emas(df):
    df["ema9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    return df


def detect_crossover(df):
    """Return the index of the most recent 9/21 EMA bullish or bearish crossover."""
    cross_idx = None
    cross_dir = None
    for i in range(1, len(df)):
        prev_bull = df["ema9"].iloc[i-1] > df["ema21"].iloc[i-1]
        curr_bull = df["ema9"].iloc[i]   > df["ema21"].iloc[i]
        if not prev_bull and curr_bull:
            cross_idx = i
            cross_dir = "bullish"
        elif prev_bull and not curr_bull:
            cross_idx = i
            cross_dir = "bearish"
    return cross_idx, cross_dir


def is_pullback_setup(df, cross_idx, direction):
    """
    After a crossover, look for:
      - Price pulled back to touch/slightly breach 21 EMA
      - A rejection candle: closes above 9 EMA (bull) or below 9 EMA (bear)
      - Volume on bounce > 20-day avg volume
    Returns True + details if setup is valid on the LAST candle.
    """
    if cross_idx is None or cross_idx >= len(df) - 1:
        return False, {}

    last = df.iloc[-1]
    close   = last["Close"]
    low     = last["Low"]
    high    = last["High"]
    ema9    = last["ema9"]
    ema21   = last["ema21"]
    volume  = last["Volume"]
    vol_avg = last["vol_avg20"]

    touched_21 = False
    # Check any candle after crossover touched/dipped to 21 EMA
    post_cross = df.iloc[cross_idx:]
    for _, row in post_cross.iterrows():
        if row["Low"] <= row["ema21"] * 1.005:   # within 0.5% tolerance
            touched_21 = True
            break

    vol_confirmed = volume > vol_avg * 1.1   # at least 10% above avg

    if direction == "bullish":
        rejection = close > ema9 and low <= ema21 * 1.005
        signal = touched_21 and rejection and vol_confirmed
        signal_type = "BULL PULLBACK"
    else:
        rejection = close < ema9 and high >= ema21 * 0.995
        signal = touched_21 and rejection and vol_confirmed
        signal_type = "BEAR PULLBACK"

    details = {
        "signal_type": signal_type,
        "close": round(close, 2),
        "ema9":  round(ema9, 2),
        "ema21": round(ema21, 2),
        "volume": int(volume),
        "vol_avg20": int(vol_avg),
        "vol_ratio": round(volume / vol_avg, 2) if vol_avg else 0,
        "touched_21": touched_21,
        "rejection_candle": rejection,
        "vol_confirmed": vol_confirmed,
    }
    return signal, details


def emas_fanning(df):
    """Check that 9/21 EMAs are angled and spreading (not tangling/flat)."""
    recent = df.tail(5)
    spread = (recent["ema9"] - recent["ema21"]).abs()
    return spread.is_monotonic_increasing or spread.iloc[-1] > spread.mean()


# ─── SPY macro filter ────────────────────────────────────────────────────────

def spy_is_bullish():
    spy = yf.download("SPY", period="3mo", interval="1d", progress=False, auto_adjust=True)
    if spy.empty or len(spy) < 50:
        return True  # assume OK if data unavailable
    # Flatten MultiIndex columns (yfinance >=0.2 returns ticker-level MultiIndex)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy["sma50"] = spy["Close"].rolling(50).mean()
    last_close = float(spy["Close"].iloc[-1])
    last_sma50 = float(spy["sma50"].iloc[-1])
    return last_close > last_sma50


# ─── Single ticker analysis ───────────────────────────────────────────────────

def scan_ticker(ticker, from_date=None, spy_bull=True):
    try:
        if from_date:
            # Pull extra history for EMA warmup (90 days before from_date)
            warmup_start = (from_date - timedelta(days=120)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=warmup_start, interval="1d",
                             progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, period="6mo", interval="1d",
                             progress=False, auto_adjust=True)

        if df is None or len(df) < 30:
            return None

        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        df = compute_emas(df.copy())

        # If a from_date filter is set, only look at signals on/after that date
        if from_date:
            scan_df = df[df.index >= pd.Timestamp(from_date)]
            if scan_df.empty:
                return None
            # Use full df for indicator calc, but last row must be in range
            df = df[df.index <= scan_df.index[-1]]

        cross_idx, cross_dir = detect_crossover(df)
        if not cross_dir:
            return None

        # Skip tangling EMAs
        if not emas_fanning(df):
            return None

        # Macro filter: only take bullish setups if SPY > 50 SMA
        if cross_dir == "bullish" and not spy_bull:
            return None

        signal, details = is_pullback_setup(df, cross_idx, cross_dir)
        if not signal:
            return None

        details["ticker"] = ticker
        details["crossover_direction"] = cross_dir
        details["last_date"] = df.index[-1].strftime("%Y-%m-%d")

        # Suggested option structure
        if cross_dir == "bullish":
            details["option_action"] = "BUY CALL"
        else:
            details["option_action"] = "BUY PUT"
        details["suggested_dte"] = "45–60 DTE"
        details["suggested_delta"] = "0.70–0.80 (ITM/ATM)"

        return details

    except Exception as e:
        print(f"  [skip] {ticker}: {e}")
        return None


# ─── Main scanner ─────────────────────────────────────────────────────────────

def run_scanner(tickers, from_date=None):
    print("\n" + "="*65)
    print("  9/21 EMA PULLBACK OPTIONS SWING SCANNER")
    if from_date:
        print(f"  Scanning signals from: {from_date.strftime('%Y-%m-%d')} onward")
    else:
        print(f"  Scanning for today:    {datetime.today().strftime('%Y-%m-%d')}")
    print("="*65)

    print("\n⏳ Checking SPY macro filter...")
    spy_bull = spy_is_bullish()
    print(f"   SPY above 50 SMA: {'✅ YES' if spy_bull else '❌ NO (bearish macro)'}\n")

    results = []
    print(f"🔍 Scanning {len(tickers)} tickers...\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"   [{i:02d}/{len(tickers)}] {ticker:<8}", end="  ")
        result = scan_ticker(ticker, from_date=from_date, spy_bull=spy_bull)
        if result:
            print(f"✅ SIGNAL: {result['signal_type']}")
            results.append(result)
        else:
            print("—")

    # ── Print results ──────────────────────────────────────────────────────
    if not results:
        print("\n📭 No setups found matching all filters.\n")
        return

    print(f"\n{'='*65}")
    print(f"  🎯 {len(results)} SETUP(S) FOUND")
    print(f"{'='*65}\n")

    for r in results:
        print(f"  {'─'*55}")
        print(f"  📌 {r['ticker']}  |  {r['signal_type']}  |  {r['last_date']}")
        print(f"  {'─'*55}")
        print(f"    Close:        ${r['close']}")
        print(f"    9 EMA:        ${r['ema9']}")
        print(f"    21 EMA:       ${r['ema21']}")
        print(f"    Volume Ratio: {r['vol_ratio']}x avg  {'✅' if r['vol_confirmed'] else '❌'}")
        print(f"    Touched 21:   {'✅' if r['touched_21'] else '❌'}")
        print(f"    Rejection ✅:  {'yes' if r['rejection_candle'] else 'no'}")
        print(f"\n  📊 Option Setup:")
        print(f"    Action:  {r['option_action']}")
        print(f"    DTE:     {r['suggested_dte']}")
        print(f"    Delta:   {r['suggested_delta']}")
        print(f"    Stop:    Below the pullback swing low / 21 EMA\n")

    # Save to CSV
    out_df = pd.DataFrame(results)
    fname = f"scan_results_{datetime.today().strftime('%Y%m%d')}.csv"
    out_df.to_csv(fname, index=False)
    print(f"💾 Results saved to: {fname}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="9/21 EMA Pullback Swing Scanner")
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Scan signals from this date onward (YYYY-MM-DD). Omit for today only.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Space-separated list of tickers. Omit to use default watchlist.",
    )
    args = parser.parse_args()

    tickers   = args.tickers or DEFAULT_TICKERS
    from_date = datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else None

    run_scanner(tickers, from_date=from_date)




# """
# Options Swing Trading Signal Scanner  (Alpaca API)
# ===================================================
# Designed for options swing trading — multi-indicator confluence required.
#
# Scanners
# --------
#   Scanner 1 — EMA 9/21 Crossover  (primary trend signal)
#       Bullish : EMA-9 crosses ABOVE EMA-21  AND  close > EMA-9
#       Bearish : EMA-9 crosses BELOW EMA-21  AND  close < EMA-9
#
#   Scanner 2 — EMA 9/21 crossing EMA 50/200  (major trend shift)
#       Bullish : EMA-21 crosses ABOVE EMA-50  AND  EMA-9 > EMA-200
#       Bearish : EMA-21 crosses BELOW EMA-50  AND  EMA-9 < EMA-200
#
# Confirmation Filters (applied to BOTH scanners)
# -------------------------------------------------
#   MACD Confirmation
#       Bullish : MACD line > Signal line  AND  histogram turning positive (momentum building)
#       Bearish : MACD line < Signal line  AND  histogram turning negative
#
#   RSI Entry Zone  (oscillator — avoids chasing extended moves)
#       Bullish : RSI(14) between 40–65  (momentum, not overbought)
#       Bearish : RSI(14) between 35–60  (momentum, not oversold)
#
#   Bollinger Band Squeeze  (volatility coil — identifies high-probability setups)
#       Active when BB width (upper-lower / middle) < squeeze_threshold
#       Squeeze = price coiling before expansion — ideal for options premium plays
#
# Signal strength scoring:
#   Each confirmation that passes adds to a "confluence score" (0–3):
#       +1 MACD confirmed
#       +1 RSI in entry zone
#       +1 BB squeeze active
#   Score 2+ = Standard signal
#   Score 3  = HIGH CONFLUENCE (strongest setups for options)
#
# Data source : Alpaca Markets v2 daily bars (IEX feed, split-adjusted)
# API keys    : loaded from .env  →  ALPACA_API_KEY, ALPACA_SECRET_KEY
#
# Usage
# -----
#     python swing_scanner.py --start-date 2026-01-01
#     python swing_scanner.py --start-date 2026-01-01 --symbol NVDA
#     python swing_scanner.py --start-date 2026-01-01 --scanner 1
#     python swing_scanner.py --start-date 2026-01-01 --scanner 2
#     python swing_scanner.py --start-date 2026-01-01 --min-score 3   # high confluence only
#     python swing_scanner.py --start-date 2026-01-01 --no-squeeze     # skip BB squeeze filter
# """
#
# import os
# import requests
# import argparse
# import warnings
# from datetime import datetime, timedelta
# from dotenv import load_dotenv
#
# import pandas as pd
#
# warnings.filterwarnings("ignore")
# load_dotenv()
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Alpaca credentials
# # ─────────────────────────────────────────────────────────────────────────────
#
# API_KEY    = os.getenv("ALPACA_API_KEY")
# SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# BASE_URL   = "https://data.alpaca.markets"
#
# _HEADERS = {
#     "APCA-API-KEY-ID":     API_KEY    or "",
#     "APCA-API-SECRET-KEY": SECRET_KEY or "",
# }
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Watchlist
# # ─────────────────────────────────────────────────────────────────────────────
#
# WATCHLIST = [
#     "SPY", "QQQ",
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "APP",
#     "AVGO", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM",
#     "TXN", "INTU", "NOW", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
#     "PANW", "CRWD", "DDOG", "NET", "ZS", "SNOW", "PLTR", "TEAM",
#     "WDAY", "ADSK", "UBER", "ABNB", "CRCL", "ZM", "RBLX", "ONON",
#     "COIN", "PYPL", "MSTR", "APLD", "BIDU", "BABA", "FUTU", "SMCI",
#     "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "TTD", "BULL", "HIMS", "CELH",
#     "CMG", "MCD", "SBUX", "NKE", "LULU", "LOW", "HD", "TGT", "WMT", "COST",
#     "PG", "KO", "PEP", "PM",
#     "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "GILD",
#     "REGN", "VRTX", "ISRG", "BSX", "SYK", "MDT", "ZTS",
#     "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "C", "SCHW", "AXP",
#     "COF", "BX", "KKR",
#     "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
#     "CAT", "BA", "HON", "UNP", "RTX", "GE", "LMT", "DE", "UPS", "FDX", "UPST",
#     "ETN", "GD", "NOC", "CSX",
#     "LIN", "APD", "SHW", "NEM", "FCX", "NUE",
#     "AMT", "PLD", "EQIX", "PSA", "O",
#     "NEE", "DUK", "SO",
#     "TSM", "ASML", "TER", "ENTG", "MPWR",
#     "RIVN", "ENPH", "FSLR",
# ]
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Configuration
# # ─────────────────────────────────────────────────────────────────────────────
#
# # EMA periods
# EMA_FAST   = 9
# EMA_SLOW   = 21
# EMA_MID    = 50
# EMA_TREND  = 200
#
# # MACD settings (standard: 12/26/9)
# MACD_FAST   = 12
# MACD_SLOW   = 26
# MACD_SIGNAL = 9
#
# # RSI
# RSI_PERIOD       = 14
# RSI_BULL_LOW     = 40   # RSI must be above this for bullish entry
# RSI_BULL_HIGH    = 65   # RSI must be below this (not overbought)
# RSI_BEAR_LOW     = 35   # RSI must be above this (not oversold) for bearish
# RSI_BEAR_HIGH    = 60   # RSI must be below this for bearish entry
#
# # Bollinger Bands
# BB_PERIOD        = 20
# BB_STD           = 2.0
# BB_SQUEEZE_PCTILE = 20  # squeeze = band width in bottom N% of historical range
#
# WARMUP_DAYS = 250       # calendar days before start_date to warm up all indicators
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Alpaca data fetch
# # ─────────────────────────────────────────────────────────────────────────────
#
# def fetch_daily_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
#     """
#     Fetch daily split-adjusted OHLCV from Alpaca v2.
#     Fetches WARMUP_DAYS extra before start_dt so all indicators are fully warmed up.
#     """
#     fetch_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
#     fetch_end   = end_dt.strftime("%Y-%m-%d")
#
#     url    = f"{BASE_URL}/v2/stocks/{symbol}/bars"
#     params = {
#         "timeframe":  "1Day",
#         "start":      fetch_start,
#         "end":        fetch_end,
#         "feed":       "iex",
#         "adjustment": "split",
#         "limit":      10000,
#     }
#     all_bars   = []
#     next_token = None
#
#     while True:
#         if next_token:
#             params["page_token"] = next_token
#         try:
#             resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
#             resp.raise_for_status()
#         except Exception:
#             return pd.DataFrame()
#
#         data       = resp.json()
#         bars       = data.get("bars", [])
#         all_bars.extend(bars)
#         next_token = data.get("next_page_token")
#         if not next_token:
#             break
#
#     if not all_bars:
#         return pd.DataFrame()
#
#     df = pd.DataFrame(all_bars)
#     df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize()
#     df = (
#         df.rename(columns={"t": "date", "o": "open", "h": "high",
#                             "l": "low",  "c": "close", "v": "volume"})
#           .set_index("date")
#           .sort_index()
#     )
#     df = df[["open", "high", "low", "close", "volume"]]
#     df = df[~df.index.duplicated(keep="first")]
#     return df
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Indicator computation
# # ─────────────────────────────────────────────────────────────────────────────
#
# def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """Compute all indicators in one pass."""
#     df = df.copy()
#     close = df["close"]
#
#     # ── EMAs ──────────────────────────────────────────────────────────────────
#     for span in (EMA_FAST, EMA_SLOW, EMA_MID, EMA_TREND):
#         df[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()
#
#     # ── MACD ──────────────────────────────────────────────────────────────────
#     ema_fast        = close.ewm(span=MACD_FAST, adjust=False).mean()
#     ema_slow        = close.ewm(span=MACD_SLOW, adjust=False).mean()
#     df["macd"]      = ema_fast - ema_slow
#     df["macd_sig"]  = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
#     df["macd_hist"] = df["macd"] - df["macd_sig"]
#
#     # ── RSI ───────────────────────────────────────────────────────────────────
#     delta  = close.diff()
#     gain   = delta.clip(lower=0)
#     loss   = (-delta).clip(lower=0)
#     avg_g  = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
#     avg_l  = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
#     rs     = avg_g / avg_l.replace(0, float("nan"))
#     df["rsi"] = 100 - (100 / (1 + rs))
#
#     # ── Bollinger Bands ───────────────────────────────────────────────────────
#     df["bb_mid"]   = close.rolling(BB_PERIOD).mean()
#     bb_std         = close.rolling(BB_PERIOD).std(ddof=0)
#     df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
#     df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
#     df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
#
#     # Rolling percentile of BB width (to detect squeeze vs expansion)
#     df["bb_width_pct"] = df["bb_width"].rolling(252, min_periods=50).rank(pct=True) * 100
#
#     return df
#
#
# def _squeeze_active(row: pd.Series) -> bool:
#     """True when BB width is in the bottom BB_SQUEEZE_PCTILE% of its 1yr history."""
#     pct = row.get("bb_width_pct", float("nan"))
#     if pd.isna(pct):
#         return False
#     return pct <= BB_SQUEEZE_PCTILE
#
#
# def _macd_bullish(prev: pd.Series, cur: pd.Series) -> bool:
#     """MACD line above signal AND histogram turning positive (increasing)."""
#     return (
#         cur["macd"] > cur["macd_sig"]
#         and cur["macd_hist"] > prev["macd_hist"]
#     )
#
#
# def _macd_bearish(prev: pd.Series, cur: pd.Series) -> bool:
#     """MACD line below signal AND histogram turning more negative (decreasing)."""
#     return (
#         cur["macd"] < cur["macd_sig"]
#         and cur["macd_hist"] < prev["macd_hist"]
#     )
#
#
# def _rsi_bull_zone(row: pd.Series) -> bool:
#     return RSI_BULL_LOW <= row["rsi"] <= RSI_BULL_HIGH
#
#
# def _rsi_bear_zone(row: pd.Series) -> bool:
#     return RSI_BEAR_LOW <= row["rsi"] <= RSI_BEAR_HIGH
#
#
# def _confluence_score(macd_ok: bool, rsi_ok: bool, squeeze: bool) -> int:
#     return int(macd_ok) + int(rsi_ok) + int(squeeze)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Scanner 1 — EMA 9/21 crossover
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_signals_s1(df: pd.DataFrame, start_dt: datetime,
#                    min_score: int, require_squeeze: bool) -> list:
#     """
#     Primary crossover: EMA-9 vs EMA-21
#       Bullish : EMA-9 crosses above EMA-21  AND  close > EMA-9
#       Bearish : EMA-9 crosses below EMA-21  AND  close < EMA-9
#     + MACD / RSI / BB squeeze confirmation
#     """
#     signals  = []
#     sim_mask = df.index >= pd.Timestamp(start_dt)
#     if not sim_mask.any():
#         return signals
#
#     start_pos = df.index.get_loc(df[sim_mask].index[0])
#
#     for i in range(start_pos, len(df)):
#         if i < 1:
#             continue
#         prev  = df.iloc[i - 1]
#         cur   = df.iloc[i]
#         date  = df.index[i]
#         close = float(cur["close"])
#         e9    = float(cur[f"ema{EMA_FAST}"])
#         e21   = float(cur[f"ema{EMA_SLOW}"])
#
#         bull_cross = prev[f"ema{EMA_FAST}"] <= prev[f"ema{EMA_SLOW}"] and e9 > e21
#         bear_cross = prev[f"ema{EMA_FAST}"] >= prev[f"ema{EMA_SLOW}"] and e9 < e21
#
#         if bull_cross and close > e9:
#             macd_ok  = _macd_bullish(prev, cur)
#             rsi_ok   = _rsi_bull_zone(cur)
#             squeeze  = _squeeze_active(cur)
#             score    = _confluence_score(macd_ok, rsi_ok, squeeze)
#
#             if require_squeeze and not squeeze:
#                 continue
#             if score < min_score:
#                 continue
#
#             signals.append({
#                 "date":    date.strftime("%Y-%m-%d"),
#                 "signal":  "BULLISH",
#                 "score":   score,
#                 "close":   round(close, 3),
#                 f"ema{EMA_FAST}":  round(e9, 3),
#                 f"ema{EMA_SLOW}":  round(e21, 3),
#                 "macd":    round(float(cur["macd"]), 4),
#                 "macd_h":  round(float(cur["macd_hist"]), 4),
#                 "rsi":     round(float(cur["rsi"]), 1),
#                 "bb_w%":   round(float(cur["bb_width_pct"]), 1) if not pd.isna(cur["bb_width_pct"]) else "—",
#                 "squeeze": "YES" if squeeze else "no",
#                 "macd_ok": "YES" if macd_ok else "no",
#                 "rsi_ok":  "YES" if rsi_ok else "no",
#             })
#
#         elif bear_cross and close < e9:
#             macd_ok  = _macd_bearish(prev, cur)
#             rsi_ok   = _rsi_bear_zone(cur)
#             squeeze  = _squeeze_active(cur)
#             score    = _confluence_score(macd_ok, rsi_ok, squeeze)
#
#             if require_squeeze and not squeeze:
#                 continue
#             if score < min_score:
#                 continue
#
#             signals.append({
#                 "date":    date.strftime("%Y-%m-%d"),
#                 "signal":  "BEARISH",
#                 "score":   score,
#                 "close":   round(close, 3),
#                 f"ema{EMA_FAST}":  round(e9, 3),
#                 f"ema{EMA_SLOW}":  round(e21, 3),
#                 "macd":    round(float(cur["macd"]), 4),
#                 "macd_h":  round(float(cur["macd_hist"]), 4),
#                 "rsi":     round(float(cur["rsi"]), 1),
#                 "bb_w%":   round(float(cur["bb_width_pct"]), 1) if not pd.isna(cur["bb_width_pct"]) else "—",
#                 "squeeze": "YES" if squeeze else "no",
#                 "macd_ok": "YES" if macd_ok else "no",
#                 "rsi_ok":  "YES" if rsi_ok else "no",
#             })
#
#     return signals
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Scanner 2 — EMA 9/21 crossing EMA 50/200
# # ─────────────────────────────────────────────────────────────────────────────
#
# def get_signals_s2(df: pd.DataFrame, start_dt: datetime,
#                    min_score: int, require_squeeze: bool) -> list:
#     """
#     Major trend shift: EMA-21 crosses EMA-50, confirmed by EMA-9 vs EMA-200
#       Bullish : EMA-21 crosses ABOVE EMA-50  AND  EMA-9 > EMA-200  (golden cross zone)
#       Bearish : EMA-21 crosses BELOW EMA-50  AND  EMA-9 < EMA-200  (death cross zone)
#     + MACD / RSI / BB squeeze confirmation
#     """
#     signals  = []
#     sim_mask = df.index >= pd.Timestamp(start_dt)
#     if not sim_mask.any():
#         return signals
#
#     start_pos = df.index.get_loc(df[sim_mask].index[0])
#
#     for i in range(start_pos, len(df)):
#         if i < 1:
#             continue
#         prev  = df.iloc[i - 1]
#         cur   = df.iloc[i]
#         date  = df.index[i]
#         close = float(cur["close"])
#         e9    = float(cur[f"ema{EMA_FAST}"])
#         e21   = float(cur[f"ema{EMA_SLOW}"])
#         e50   = float(cur[f"ema{EMA_MID}"])
#         e200  = float(cur[f"ema{EMA_TREND}"])
#
#         bull_cross = prev[f"ema{EMA_SLOW}"] <= prev[f"ema{EMA_MID}"] and e21 > e50
#         bear_cross = prev[f"ema{EMA_SLOW}"] >= prev[f"ema{EMA_MID}"] and e21 < e50
#
#         if bull_cross and e9 > e200:
#             macd_ok  = _macd_bullish(prev, cur)
#             rsi_ok   = _rsi_bull_zone(cur)
#             squeeze  = _squeeze_active(cur)
#             score    = _confluence_score(macd_ok, rsi_ok, squeeze)
#
#             if require_squeeze and not squeeze:
#                 continue
#             if score < min_score:
#                 continue
#
#             signals.append({
#                 "date":    date.strftime("%Y-%m-%d"),
#                 "signal":  "BULLISH",
#                 "score":   score,
#                 "close":   round(close, 3),
#                 f"ema{EMA_FAST}":   round(e9, 3),
#                 f"ema{EMA_SLOW}":   round(e21, 3),
#                 f"ema{EMA_MID}":    round(e50, 3),
#                 f"ema{EMA_TREND}":  round(e200, 3),
#                 "macd":    round(float(cur["macd"]), 4),
#                 "macd_h":  round(float(cur["macd_hist"]), 4),
#                 "rsi":     round(float(cur["rsi"]), 1),
#                 "bb_w%":   round(float(cur["bb_width_pct"]), 1) if not pd.isna(cur["bb_width_pct"]) else "—",
#                 "squeeze": "YES" if squeeze else "no",
#                 "macd_ok": "YES" if macd_ok else "no",
#                 "rsi_ok":  "YES" if rsi_ok else "no",
#             })
#
#         elif bear_cross and e9 < e200:
#             macd_ok  = _macd_bearish(prev, cur)
#             rsi_ok   = _rsi_bear_zone(cur)
#             squeeze  = _squeeze_active(cur)
#             score    = _confluence_score(macd_ok, rsi_ok, squeeze)
#
#             if require_squeeze and not squeeze:
#                 continue
#             if score < min_score:
#                 continue
#
#             signals.append({
#                 "date":    date.strftime("%Y-%m-%d"),
#                 "signal":  "BEARISH",
#                 "score":   score,
#                 "close":   round(close, 3),
#                 f"ema{EMA_FAST}":   round(e9, 3),
#                 f"ema{EMA_SLOW}":   round(e21, 3),
#                 f"ema{EMA_MID}":    round(e50, 3),
#                 f"ema{EMA_TREND}":  round(e200, 3),
#                 "macd":    round(float(cur["macd"]), 4),
#                 "macd_h":  round(float(cur["macd_hist"]), 4),
#                 "rsi":     round(float(cur["rsi"]), 1),
#                 "bb_w%":   round(float(cur["bb_width_pct"]), 1) if not pd.isna(cur["bb_width_pct"]) else "—",
#                 "squeeze": "YES" if squeeze else "no",
#                 "macd_ok": "YES" if macd_ok else "no",
#                 "rsi_ok":  "YES" if rsi_ok else "no",
#             })
#
#     return signals
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Display helpers
# # ─────────────────────────────────────────────────────────────────────────────
#
# _W = 110
#
#
# def _header(title: str):
#     print(f"\n{'='*_W}")
#     print(f"  {title}")
#     print(f"{'='*_W}")
#
#
# def _score_stars(score: int) -> str:
#     """Visual confluence indicator."""
#     stars = {3: "★★★ HIGH", 2: "★★  MED", 1: "★   LOW", 0: "    NONE"}
#     return stars.get(score, "?")
#
#
# def _table_s1(rows: list, direction: str):
#     emoji = "🟢" if direction == "BULLISH" else "🔴"
#     label = "ABOVE" if direction == "BULLISH" else "BELOW"
#     op    = ">" if direction == "BULLISH" else "<"
#     print(f"\n{emoji} SCANNER 1 — {direction}  "
#           f"(EMA-{EMA_FAST} crossed {label} EMA-{EMA_SLOW}, close {op} EMA-{EMA_FAST})")
#
#     if not rows:
#         print("  No signals.")
#         return
#
#     print(f"  {'Symbol':<8}  {'Date':<12}  {'Score':<10}  {'Close':>9}  "
#           f"{'EMA-9':>9}  {'EMA-21':>9}  "
#           f"{'MACD':>8}  {'Hist':>8}  {'RSI':>6}  {'BB W%':>7}  Confirms")
#     print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*9}  "
#           f"{'-'*9}  {'-'*9}  "
#           f"{'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*22}")
#
#     for r in rows:
#         confirms = (
#             f"MACD:{r['macd_ok']:<3}  RSI:{r['rsi_ok']:<3}  SQZ:{r['squeeze']}"
#         )
#         print(
#             f"  {r['symbol']:<8}  {r['date']:<12}  {_score_stars(r['score']):<10}  "
#             f"${r['close']:>8.3f}  "
#             f"${r[f'ema{EMA_FAST}']:>8.3f}  ${r[f'ema{EMA_SLOW}']:>8.3f}  "
#             f"{r['macd']:>8.4f}  {r['macd_h']:>8.4f}  "
#             f"{r['rsi']:>6.1f}  {str(r['bb_w%']):>7}  {confirms}"
#         )
#
#     bull_high = sum(1 for r in rows if r["score"] == 3)
#     print(f"\n  Total: {len(rows)} signal(s)   ★★★ High confluence: {bull_high}")
#
#
# def _table_s2(rows: list, direction: str):
#     emoji = "🟢" if direction == "BULLISH" else "🔴"
#     label = "ABOVE" if direction == "BULLISH" else "BELOW"
#     conf  = f"EMA-{EMA_FAST} > EMA-{EMA_TREND}" if direction == "BULLISH" else f"EMA-{EMA_FAST} < EMA-{EMA_TREND}"
#     print(f"\n{emoji} SCANNER 2 — {direction}  "
#           f"(EMA-{EMA_SLOW} crossed {label} EMA-{EMA_MID}  +  {conf})")
#
#     if not rows:
#         print("  No signals.")
#         return
#
#     print(f"  {'Symbol':<8}  {'Date':<12}  {'Score':<10}  {'Close':>9}  "
#           f"{'EMA-9':>9}  {'EMA-21':>9}  {'EMA-50':>9}  {'EMA-200':>10}  "
#           f"{'MACD':>8}  {'RSI':>6}  {'BB W%':>7}  Confirms")
#     print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*9}  "
#           f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}  "
#           f"{'-'*8}  {'-'*6}  {'-'*7}  {'-'*22}")
#
#     for r in rows:
#         confirms = (
#             f"MACD:{r['macd_ok']:<3}  RSI:{r['rsi_ok']:<3}  SQZ:{r['squeeze']}"
#         )
#         print(
#             f"  {r['symbol']:<8}  {r['date']:<12}  {_score_stars(r['score']):<10}  "
#             f"${r['close']:>8.3f}  "
#             f"${r[f'ema{EMA_FAST}']:>8.3f}  ${r[f'ema{EMA_SLOW}']:>8.3f}  "
#             f"${r[f'ema{EMA_MID}']:>8.3f}  ${r[f'ema{EMA_TREND}']:>9.3f}  "
#             f"{r['macd']:>8.4f}  {r['rsi']:>6.1f}  {str(r['bb_w%']):>7}  {confirms}"
#         )
#
#     high = sum(1 for r in rows if r["score"] == 3)
#     print(f"\n  Total: {len(rows)} signal(s)   ★★★ High confluence: {high}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Legend
# # ─────────────────────────────────────────────────────────────────────────────
#
# def _print_legend():
#     print(f"\n{'─'*_W}")
#     print("  INDICATOR LEGEND")
#     print(f"{'─'*_W}")
#     print(f"  EMA-{EMA_FAST}/{EMA_SLOW}     : Primary crossover — options entry trigger")
#     print(f"  EMA-{EMA_MID}/{EMA_TREND}   : Major trend filter (Scanner 2 only)")
#     print(f"  MACD ({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL})  : Trend confirmation  — YES = aligned, no = diverging")
#     print(f"  RSI  ({RSI_PERIOD})      : Entry zone oscillator")
#     print(f"               Bullish zone: {RSI_BULL_LOW}–{RSI_BULL_HIGH}  |  Bearish zone: {RSI_BEAR_LOW}–{RSI_BEAR_HIGH}")
#     print(f"  BB Width%    : Bollinger Band squeeze — percentile of 1yr range")
#     print(f"               SQZ YES = bottom {BB_SQUEEZE_PCTILE}%  (volatility coil, premium expansion likely)")
#     print(f"  Score        : Confluence count  (★★★=3 all confirmed, best options setups)")
#     print(f"{'─'*_W}")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Core scanner loop
# # ─────────────────────────────────────────────────────────────────────────────
#
# def run_scanner(symbols: list, start_date_str: str,
#                 run_s1: bool, run_s2: bool,
#                 min_score: int, require_squeeze: bool):
#
#     start_dt  = datetime.strptime(start_date_str, "%Y-%m-%d")
#     end_dt    = datetime.now()
#     today_str = end_dt.strftime("%Y-%m-%d")
#
#     _header(
#         f"OPTIONS SWING TRADING SIGNAL SCANNER  |  {start_date_str} → {today_str}  "
#         f"|  min score: {min_score}  |  squeeze filter: {'ON' if require_squeeze else 'OFF'}"
#     )
#     _print_legend()
#
#     s1_bull, s1_bear = [], []
#     s2_bull, s2_bear = [], []
#
#     for i, sym in enumerate(symbols, 1):
#         print(f"  [{i:>3}/{len(symbols)}] {sym:<6} ...", end="  ", flush=True)
#
#         df = fetch_daily_bars(sym, start_dt, end_dt)
#         if df.empty or len(df) < 60:
#             print("no data")
#             continue
#
#         df = add_indicators(df)
#         parts = []
#
#         if run_s1:
#             sigs = get_signals_s1(df, start_dt, min_score, require_squeeze)
#             for s in sigs:
#                 s["symbol"] = sym
#             bull = [s for s in sigs if s["signal"] == "BULLISH"]
#             bear = [s for s in sigs if s["signal"] == "BEARISH"]
#             s1_bull.extend(bull)
#             s1_bear.extend(bear)
#             if sigs:
#                 h = sum(1 for s in sigs if s["score"] == 3)
#                 parts.append(f"S1: 🟢{len(bull)} 🔴{len(bear)} ★★★{h}")
#
#         if run_s2:
#             sigs = get_signals_s2(df, start_dt, min_score, require_squeeze)
#             for s in sigs:
#                 s["symbol"] = sym
#             bull = [s for s in sigs if s["signal"] == "BULLISH"]
#             bear = [s for s in sigs if s["signal"] == "BEARISH"]
#             s2_bull.extend(bull)
#             s2_bear.extend(bear)
#             if sigs:
#                 h = sum(1 for s in sigs if s["score"] == 3)
#                 parts.append(f"S2: 🟢{len(bull)} 🔴{len(bear)} ★★★{h}")
#
#         print("  |  ".join(parts) if parts else "no signals")
#
#     key = lambda r: (r["date"], r["symbol"])
#     s1_bull.sort(key=key); s1_bear.sort(key=key)
#     s2_bull.sort(key=key); s2_bear.sort(key=key)
#
#     print(f"\n{'─'*_W}")
#     print(f"  RESULTS  ({start_date_str} → {today_str})")
#     print(f"{'─'*_W}")
#
#     if run_s1:
#         print(f"\n{'─'*_W}")
#         print(f"  SCANNER 1 — EMA {EMA_FAST}/{EMA_SLOW} Crossover  (short-term momentum)")
#         print(f"{'─'*_W}")
#         _table_s1(s1_bull, "BULLISH")
#         _table_s1(s1_bear, "BEARISH")
#
#     if run_s2:
#         print(f"\n{'─'*_W}")
#         print(f"  SCANNER 2 — EMA {EMA_SLOW}/{EMA_MID} crossing EMA {EMA_TREND}  (major trend shift)")
#         print(f"{'─'*_W}")
#         _table_s2(s2_bull, "BULLISH")
#         _table_s2(s2_bear, "BEARISH")
#
#     print(f"\n{'='*_W}")
#     print(f"  SUMMARY")
#     print(f"{'='*_W}")
#     if run_s1:
#         total_s1 = len(s1_bull) + len(s1_bear)
#         h1 = sum(1 for r in s1_bull + s1_bear if r["score"] == 3)
#         print(f"  Scanner 1 (EMA {EMA_FAST}/{EMA_SLOW})          :  "
#               f"{total_s1:>4} signals  —  🟢 {len(s1_bull)} bullish  🔴 {len(s1_bear)} bearish  ★★★ {h1} high")
#     if run_s2:
#         total_s2 = len(s2_bull) + len(s2_bear)
#         h2 = sum(1 for r in s2_bull + s2_bear if r["score"] == 3)
#         print(f"  Scanner 2 (EMA {EMA_SLOW}/{EMA_MID} x {EMA_TREND})      :  "
#               f"{total_s2:>4} signals  —  🟢 {len(s2_bull)} bullish  🔴 {len(s2_bear)} bearish  ★★★ {h2} high")
#     print(f"{'='*_W}\n")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Single-symbol mode
# # ─────────────────────────────────────────────────────────────────────────────
#
# def run_single(symbol: str, start_date_str: str,
#                run_s1: bool, run_s2: bool,
#                min_score: int, require_squeeze: bool):
#
#     start_dt  = datetime.strptime(start_date_str, "%Y-%m-%d")
#     end_dt    = datetime.now()
#     today_str = end_dt.strftime("%Y-%m-%d")
#
#     _header(f"OPTIONS SWING SCAN: {symbol}  |  {start_date_str} → {today_str}")
#     _print_legend()
#
#     df = fetch_daily_bars(symbol, start_dt, end_dt)
#     if df.empty or len(df) < 60:
#         print("  No data available.\n")
#         return
#
#     df = add_indicators(df)
#
#     if run_s1:
#         sigs = get_signals_s1(df, start_dt, min_score, require_squeeze)
#         for s in sigs:
#             s["symbol"] = symbol
#         bull = [s for s in sigs if s["signal"] == "BULLISH"]
#         bear = [s for s in sigs if s["signal"] == "BEARISH"]
#         print(f"\n{'─'*_W}")
#         print(f"  SCANNER 1 — EMA {EMA_FAST}/{EMA_SLOW} Crossover")
#         print(f"{'─'*_W}")
#         _table_s1(bull, "BULLISH")
#         _table_s1(bear, "BEARISH")
#
#     if run_s2:
#         sigs = get_signals_s2(df, start_dt, min_score, require_squeeze)
#         for s in sigs:
#             s["symbol"] = symbol
#         bull = [s for s in sigs if s["signal"] == "BULLISH"]
#         bear = [s for s in sigs if s["signal"] == "BEARISH"]
#         print(f"\n{'─'*_W}")
#         print(f"  SCANNER 2 — EMA {EMA_SLOW}/{EMA_MID} crossing EMA {EMA_TREND}")
#         print(f"{'─'*_W}")
#         _table_s2(bull, "BULLISH")
#         _table_s2(bear, "BEARISH")
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# #  Entry point
# # ─────────────────────────────────────────────────────────────────────────────
#
# if __name__ == "__main__":
#     if not API_KEY or not SECRET_KEY:
#         print("\n[ERROR] ALPACA_API_KEY / ALPACA_SECRET_KEY not found in .env\n")
#         raise SystemExit(1)
#
#     parser = argparse.ArgumentParser(
#         description="Options Swing Trading Signal Scanner — Alpaca API"
#     )
#     parser.add_argument("--start-date",    required=True,
#                         help="Scan start date  YYYY-MM-DD")
#     parser.add_argument("--symbol",        default=None,
#                         help="Single symbol (skips full watchlist)")
#     parser.add_argument("--scanner",       type=int, choices=[1, 2], default=None,
#                         help="Run only Scanner 1 or Scanner 2 (default: both)")
#     parser.add_argument("--min-score",     type=int, default=1,
#                         choices=[0, 1, 2, 3],
#                         help="Minimum confluence score 0-3 (default 1; use 3 for highest-quality only)")
#     parser.add_argument("--no-squeeze",    action="store_true",
#                         help="Disable Bollinger Band squeeze requirement (show all signals regardless)")
#
#     args = parser.parse_args()
#
#     run_s1         = args.scanner in (None, 1)
#     run_s2         = args.scanner in (None, 2)
#     require_squeeze = False  # squeeze is scored, not required, unless --no-squeeze changes nothing
#     # NOTE: to require squeeze always, set require_squeeze = True here
#
#     if args.symbol:
#         run_single(
#             args.symbol.upper(), args.start_date,
#             run_s1, run_s2, args.min_score, require_squeeze
#         )
#     else:
#         run_scanner(
#             WATCHLIST, args.start_date,
#             run_s1, run_s2, args.min_score, require_squeeze
#         )
#
#
