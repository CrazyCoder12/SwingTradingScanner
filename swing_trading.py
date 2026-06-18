"""
EMA Crossover Signal Scanner  (Alpaca API)
===========================================
Two scanners run in parallel over the watchlist:

  Scanner 1 — EMA 5/12 Crossover  (short-term momentum)
      Bullish : EMA-5 crosses ABOVE EMA-12  AND  close > EMA-5
      Bearish : EMA-5 crosses BELOW EMA-12  AND  close < EMA-5

  Scanner 2 — EMA 5/12 crossing EMA 34/50  (medium-term trend shift)
      Bullish : EMA-12 crosses ABOVE EMA-34  AND  EMA-5 > EMA-50
      Bearish : EMA-12 crosses BELOW EMA-34  AND  EMA-5 < EMA-50

Results are printed in separate tables:
  - Scanner 1 Bullish  /  Scanner 1 Bearish
  - Scanner 2 Bullish  /  Scanner 2 Bearish

Data source: Alpaca Markets v2 daily bars (IEX feed, split-adjusted).
API keys loaded from .env  →  ALPACA_API_KEY, ALPACA_SECRET_KEY

Usage:
    python swing_trading.py --start-date 2026-03-01
    python swing_trading.py --start-date 2026-03-01 --symbol NVDA
    python swing_trading.py --start-date 2026-01-01 --scanner 1
    python swing_trading.py --start-date 2026-01-01 --scanner 2
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
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA","APP",
    "AVGO", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM",
    "TXN", "INTU", "NOW", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
    "PANW", "CRWD", "DDOG", "NET", "ZS", "SNOW", "PLTR", "TEAM",
    "WDAY", "ADSK", "UBER", "ABNB","CRCL","ZM","RBLX","ONON",
    "COIN", "PYPL","MSTR","APLD","BIDU","BABA","FUTU","SMCI",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS","TTD","BULL","HIMS","CELH",
    "CMG", "MCD", "SBUX", "NKE", "LULU", "LOW", "HD", "TGT", "WMT", "COST",
    "PG", "KO", "PEP", "PM",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "AMGN", "GILD",
    "REGN", "VRTX", "ISRG", "BSX", "SYK", "MDT", "ZTS",
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "C", "SCHW", "AXP",
    "COF", "BX", "KKR",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
    "CAT", "BA", "HON", "UNP", "RTX", "GE", "LMT", "DE", "UPS", "FDX","UPST",
    "ETN", "GD", "NOC", "CSX",
    "LIN", "APD", "SHW", "NEM", "FCX", "NUE",
    "AMT", "PLD", "EQIX", "PSA", "O",
    "NEE", "DUK", "SO",
    "TSM", "ASML", "TER", "ENTG", "MPWR",
    "RIVN", "ENPH", "FSLR",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Alpaca data fetch
# ─────────────────────────────────────────────────────────────────────────────

WARMUP_DAYS = 120   # calendar days before start_date to warm up EMAs

def fetch_daily_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch daily split-adjusted OHLCV from Alpaca v2.
    Fetches WARMUP_DAYS extra before start_dt so all EMAs are fully warmed up.
    """
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
    all_bars   = []
    next_token = None

    while True:
        if next_token:
            params["page_token"] = next_token
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            return pd.DataFrame()

        data       = resp.json()
        bars       = data.get("bars", [])
        all_bars.extend(bars)
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

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for span in (5, 12, 34, 50):
        df[f"ema{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Scanner 1 — EMA 5/12 crossover
# ─────────────────────────────────────────────────────────────────────────────

def get_signals_s1(df: pd.DataFrame, start_dt: datetime) -> list:
    """
    Bullish : EMA-5 crosses above EMA-12  AND  close > EMA-5
    Bearish : EMA-5 crosses below EMA-12  AND  close < EMA-5
    """
    signals   = []
    sim_mask  = df.index >= pd.Timestamp(start_dt)
    if not sim_mask.any():
        return signals

    start_pos = df.index.get_loc(df[sim_mask].index[0])

    for i in range(start_pos, len(df)):
        if i < 1:
            continue
        prev  = df.iloc[i - 1]
        cur   = df.iloc[i]
        date  = df.index[i]
        close = float(cur["close"])
        e5    = float(cur["ema5"])
        e12   = float(cur["ema12"])

        bull_cross = prev["ema5"] <= prev["ema12"] and cur["ema5"] > cur["ema12"]
        bear_cross = prev["ema5"] >= prev["ema12"] and cur["ema5"] < cur["ema12"]

        if bull_cross and close > e5:
            signals.append({
                "date":   date.strftime("%Y-%m-%d"),
                "signal": "BULLISH",
                "close":  round(close, 3),
                "ema5":   round(e5, 3),
                "ema12":  round(e12, 3),
            })
        elif bear_cross and close < e5:
            signals.append({
                "date":   date.strftime("%Y-%m-%d"),
                "signal": "BEARISH",
                "close":  round(close, 3),
                "ema5":   round(e5, 3),
                "ema12":  round(e12, 3),
            })

    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  Scanner 2 — EMA 5/12 crossing EMA 34/50
# ─────────────────────────────────────────────────────────────────────────────

def get_signals_s2(df: pd.DataFrame, start_dt: datetime) -> list:
    """
    Bullish : EMA-12 crosses ABOVE EMA-34  AND  EMA-5 > EMA-50  (fast cluster above slow cluster)
    Bearish : EMA-12 crosses BELOW EMA-34  AND  EMA-5 < EMA-50  (fast cluster below slow cluster)
    """
    signals   = []
    sim_mask  = df.index >= pd.Timestamp(start_dt)
    if not sim_mask.any():
        return signals

    start_pos = df.index.get_loc(df[sim_mask].index[0])

    for i in range(start_pos, len(df)):
        if i < 1:
            continue
        prev  = df.iloc[i - 1]
        cur   = df.iloc[i]
        date  = df.index[i]
        close = float(cur["close"])
        e5    = float(cur["ema5"])
        e12   = float(cur["ema12"])
        e34   = float(cur["ema34"])
        e50   = float(cur["ema50"])

        # Primary crossover: EMA-12 vs EMA-34 (the "meeting point" of the two clusters)
        bull_cross = prev["ema12"] <= prev["ema34"] and cur["ema12"] > cur["ema34"]
        bear_cross = prev["ema12"] >= prev["ema34"] and cur["ema12"] < cur["ema34"]

        if bull_cross and e5 > e50:
            signals.append({
                "date":   date.strftime("%Y-%m-%d"),
                "signal": "BULLISH",
                "close":  round(close, 3),
                "ema5":   round(e5, 3),
                "ema12":  round(e12, 3),
                "ema34":  round(e34, 3),
                "ema50":  round(e50, 3),
            })
        elif bear_cross and e5 < e50:
            signals.append({
                "date":   date.strftime("%Y-%m-%d"),
                "signal": "BEARISH",
                "close":  round(close, 3),
                "ema5":   round(e5, 3),
                "ema12":  round(e12, 3),
                "ema34":  round(e34, 3),
                "ema50":  round(e50, 3),
            })

    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_W = 78

def _header(title: str):
    print(f"\n{'='*_W}")
    print(f"  {title}")
    print(f"{'='*_W}")


def _table_s1(rows: list, direction: str):
    """Print Scanner-1 table for one direction."""
    emoji  = "🟢" if direction == "BULLISH" else "🔴"
    label  = "ABOVE" if direction == "BULLISH" else "BELOW"
    print(f"\n{emoji} SCANNER 1 — {direction}  "
          f"(EMA-5 crossed {label} EMA-12, close {'>' if direction=='BULLISH' else '<'} EMA-5)")
    if not rows:
        print("  No signals.")
        return

    hdr = f"  {'Symbol':<8}  {'Date':<12}  {'Close':>9}  {'EMA-5':>9}  {'EMA-12':>9}"
    sep = f"  {'-'*8}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}"
    print(hdr)
    print(sep)
    for r in rows:
        print(f"  {r['symbol']:<8}  {r['date']:<12}  "
              f"${r['close']:>8.3f}  ${r['ema5']:>8.3f}  ${r['ema12']:>8.3f}")
    print(f"\n  Total: {len(rows)} signal(s)")


def _table_s2(rows: list, direction: str):
    """Print Scanner-2 table for one direction."""
    emoji  = "🟢" if direction == "BULLISH" else "🔴"
    label  = "ABOVE" if direction == "BULLISH" else "BELOW"
    conf   = "EMA-5 > EMA-50" if direction == "BULLISH" else "EMA-5 < EMA-50"
    print(f"\n{emoji} SCANNER 2 — {direction}  "
          f"(EMA-12 crossed {label} EMA-34  +  {conf})")
    if not rows:
        print("  No signals.")
        return

    hdr = (f"  {'Symbol':<8}  {'Date':<12}  {'Close':>9}  "
           f"{'EMA-5':>9}  {'EMA-12':>9}  {'EMA-34':>9}  {'EMA-50':>9}")
    sep = (f"  {'-'*8}  {'-'*12}  {'-'*9}  "
           f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    print(hdr)
    print(sep)
    for r in rows:
        print(f"  {r['symbol']:<8}  {r['date']:<12}  "
              f"${r['close']:>8.3f}  ${r['ema5']:>8.3f}  ${r['ema12']:>8.3f}  "
              f"${r['ema34']:>8.3f}  ${r['ema50']:>8.3f}")
    print(f"\n  Total: {len(rows)} signal(s)")


# ─────────────────────────────────────────────────────────────────────────────
#  Core scanner loop
# ─────────────────────────────────────────────────────────────────────────────

def run_scanner(symbols: list, start_date_str: str, run_s1: bool, run_s2: bool):
    start_dt  = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt    = datetime.now()
    today_str = end_dt.strftime("%Y-%m-%d")

    _header(f"EMA CROSSOVER SIGNAL SCANNER  |  {start_date_str} → {today_str}")

    s1_bull, s1_bear = [], []
    s2_bull, s2_bear = [], []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] {sym:<6} ...", end="  ", flush=True)

        df = fetch_daily_bars(sym, start_dt, end_dt)
        if df.empty or len(df) < 60:
            print("no data")
            continue

        df = add_emas(df)

        parts = []

        if run_s1:
            sigs = get_signals_s1(df, start_dt)
            for s in sigs:
                s["symbol"] = sym
            bull = [s for s in sigs if s["signal"] == "BULLISH"]
            bear = [s for s in sigs if s["signal"] == "BEARISH"]
            s1_bull.extend(bull)
            s1_bear.extend(bear)
            if sigs:
                parts.append(f"S1: 🟢{len(bull)} 🔴{len(bear)}")

        if run_s2:
            sigs = get_signals_s2(df, start_dt)
            for s in sigs:
                s["symbol"] = sym
            bull = [s for s in sigs if s["signal"] == "BULLISH"]
            bear = [s for s in sigs if s["signal"] == "BEARISH"]
            s2_bull.extend(bull)
            s2_bear.extend(bear)
            if sigs:
                parts.append(f"S2: 🟢{len(bull)} 🔴{len(bear)}")

        print("  |  ".join(parts) if parts else "no signals")

    # ── Sort by date then symbol ──────────────────────────────────────────────
    key = lambda r: (r["date"], r["symbol"])
    s1_bull.sort(key=key);  s1_bear.sort(key=key)
    s2_bull.sort(key=key);  s2_bear.sort(key=key)

    # ── Display ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*_W}")
    print(f"  RESULTS  ({start_date_str} → {today_str})")
    print(f"{'─'*_W}")

    if run_s1:
        print(f"\n{'─'*_W}")
        print(f"  SCANNER 1 — EMA 5/12 Crossover  (short-term momentum)")
        print(f"{'─'*_W}")
        _table_s1(s1_bull, "BULLISH")
        _table_s1(s1_bear, "BEARISH")

    if run_s2:
        print(f"\n{'─'*_W}")
        print(f"  SCANNER 2 — EMA 5/12 crossing EMA 34/50  (medium-term trend shift)")
        print(f"{'─'*_W}")
        _table_s2(s2_bull, "BULLISH")
        _table_s2(s2_bear, "BEARISH")

    # ── Grand summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*_W}")
    print(f"  SUMMARY")
    print(f"{'='*_W}")
    if run_s1:
        total_s1 = len(s1_bull) + len(s1_bear)
        print(f"  Scanner 1 (5/12 cross)      :  "
              f"{total_s1:>4} signals  —  🟢 {len(s1_bull)} bullish  🔴 {len(s1_bear)} bearish")
    if run_s2:
        total_s2 = len(s2_bull) + len(s2_bear)
        print(f"  Scanner 2 (5/12 x 34/50)   :  "
              f"{total_s2:>4} signals  —  🟢 {len(s2_bull)} bullish  🔴 {len(s2_bear)} bearish")
    print(f"{'='*_W}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Single-symbol mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single(symbol: str, start_date_str: str, run_s1: bool, run_s2: bool):
    start_dt  = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt    = datetime.now()
    today_str = end_dt.strftime("%Y-%m-%d")

    _header(f"SIGNAL SCAN: {symbol}  |  {start_date_str} → {today_str}")

    df = fetch_daily_bars(symbol, start_dt, end_dt)
    if df.empty or len(df) < 60:
        print("  No data available.\n")
        return

    df = add_emas(df)

    if run_s1:
        sigs  = get_signals_s1(df, start_dt)
        for s in sigs:
            s["symbol"] = symbol
        bull = [s for s in sigs if s["signal"] == "BULLISH"]
        bear = [s for s in sigs if s["signal"] == "BEARISH"]
        print(f"\n{'─'*_W}")
        print(f"  SCANNER 1 — EMA 5/12 Crossover")
        print(f"{'─'*_W}")
        _table_s1(bull, "BULLISH")
        _table_s1(bear, "BEARISH")

    if run_s2:
        sigs  = get_signals_s2(df, start_dt)
        for s in sigs:
            s["symbol"] = symbol
        bull = [s for s in sigs if s["signal"] == "BULLISH"]
        bear = [s for s in sigs if s["signal"] == "BEARISH"]
        print(f"\n{'─'*_W}")
        print(f"  SCANNER 2 — EMA 5/12 crossing EMA 34/50")
        print(f"{'─'*_W}")
        _table_s2(bull, "BULLISH")
        _table_s2(bear, "BEARISH")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        print("\n[ERROR] ALPACA_API_KEY / ALPACA_SECRET_KEY not found in .env\n")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(
        description="EMA Crossover Signal Scanner — Alpaca API"
    )
    parser.add_argument("--start-date", required=True,
                        help="Scan start date  YYYY-MM-DD")
    parser.add_argument("--symbol",     default=None,
                        help="Single symbol (skips full watchlist)")
    parser.add_argument("--scanner",    type=int, choices=[1, 2], default=None,
                        help="Run only Scanner 1 or Scanner 2 (default: both)")
    args = parser.parse_args()

    run_s1 = args.scanner in (None, 1)
    run_s2 = args.scanner in (None, 2)

    if args.symbol:
        run_single(args.symbol.upper(), args.start_date, run_s1, run_s2)
    else:
        run_scanner(WATCHLIST, args.start_date, run_s1, run_s2)
