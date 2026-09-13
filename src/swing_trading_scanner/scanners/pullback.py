"""
9/21 EMA Pullback Options Swing Scanner
Strategy: Pullback to 21 EMA after a 9/21 EMA crossover, confirmed by volume + SPY trend.
Usage:
    python -m swing_trading_scanner.scanners.pullback                        # scan today
    python -m swing_trading_scanner.scanners.pullback --from-date 2025-01-01 # scan from a date onward
    python -m swing_trading_scanner.scanners.pullback --tickers AAPL MSFT NVDA  # custom tickers
"""

from swing_trading_scanner.config import load_environment

load_environment()


import argparse
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Default watchlist ────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "APP",
    "AVGO",
    "ADBE",
    "CRM",
    "ORCL",
    "CSCO",
    "INTC",
    "AMD",
    "QCOM",
    "TXN",
    "INTU",
    "NOW",
    "AMAT",
    "MU",
    "LRCX",
    "KLAC",
    "SNPS",
    "CDNS",
    "PANW",
    "CRWD",
    "DDOG",
    "NET",
    "ZS",
    "SNOW",
    "PLTR",
    "TEAM",
    "WDAY",
    "ADSK",
    "UBER",
    "ABNB",
    "CRCL",
    "ZM",
    "RBLX",
    "ONON",
    "COIN",
    "PYPL",
    "MSTR",
    "APLD",
    "BIDU",
    "BABA",
    "FUTU",
    "SMCI",
    "NFLX",
    "DIS",
    "CMCSA",
    "T",
    "VZ",
    "TMUS",
    "TTD",
    "BULL",
    "HIMS",
    "CELH",
    "CMG",
    "MCD",
    "SBUX",
    "NKE",
    "LULU",
    "LOW",
    "HD",
    "TGT",
    "WMT",
    "COST",
    "PG",
    "KO",
    "PEP",
    "PM",
    "HOOD",
    "UNH",
    "JNJ",
    "LLY",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "PFE",
    "AMGN",
    "GILD",
    "REGN",
    "VRTX",
    "ISRG",
    "BSX",
    "SYK",
    "MDT",
    "ZTS",
    "JPM",
    "V",
    "MA",
    "BAC",
    "WFC",
    "MS",
    "GS",
    "BLK",
    "C",
    "SCHW",
    "AXP",
    "COF",
    "BX",
    "KKR",
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "OXY",
    "CAT",
    "BA",
    "HON",
    "UNP",
    "RTX",
    "GE",
    "LMT",
    "DE",
    "UPS",
    "FDX",
    "UPST",
    "ETN",
    "GD",
    "NOC",
    "CSX",
    "LIN",
    "APD",
    "SHW",
    "NEM",
    "FCX",
    "NUE",
    "AMT",
    "PLD",
    "EQIX",
    "PSA",
    "O",
    "NEE",
    "DUK",
    "SO",
    "TSM",
    "ASML",
    "TER",
    "ENTG",
    "MPWR",
    "RIVN",
    "ENPH",
    "FSLR",
]

# ─── EMA helpers ─────────────────────────────────────────────────────────────


def compute_emas(df):
    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    return df


def detect_crossover(df):
    """Return the index of the most recent 9/21 EMA bullish or bearish crossover."""
    cross_idx = None
    cross_dir = None
    for i in range(1, len(df)):
        prev_bull = df["ema9"].iloc[i - 1] > df["ema21"].iloc[i - 1]
        curr_bull = df["ema9"].iloc[i] > df["ema21"].iloc[i]
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
    close = last["Close"]
    low = last["Low"]
    high = last["High"]
    ema9 = last["ema9"]
    ema21 = last["ema21"]
    volume = last["Volume"]
    vol_avg = last["vol_avg20"]

    touched_21 = False
    # Check any candle after crossover touched/dipped to 21 EMA
    post_cross = df.iloc[cross_idx:]
    for _, row in post_cross.iterrows():
        if row["Low"] <= row["ema21"] * 1.005:  # within 0.5% tolerance
            touched_21 = True
            break

    vol_confirmed = volume > vol_avg * 1.1  # at least 10% above avg

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
        "ema9": round(ema9, 2),
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
    spy = yf.download(
        "SPY", period="3mo", interval="1d", progress=False, auto_adjust=True
    )
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
            df = yf.download(
                ticker,
                start=warmup_start,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        else:
            df = yf.download(
                ticker, period="6mo", interval="1d", progress=False, auto_adjust=True
            )

        if df is None or len(df) < 30:
            return None

        df.columns = (
            df.columns.get_level_values(0)
            if isinstance(df.columns, pd.MultiIndex)
            else df.columns
        )
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
    print("\n" + "=" * 65)
    print("  9/21 EMA PULLBACK OPTIONS SWING SCANNER")
    if from_date:
        print(f"  Scanning signals from: {from_date.strftime('%Y-%m-%d')} onward")
    else:
        print(f"  Scanning for today:    {datetime.today().strftime('%Y-%m-%d')}")
    print("=" * 65)

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

    print(f"\n{'=' * 65}")
    print(f"  🎯 {len(results)} SETUP(S) FOUND")
    print(f"{'=' * 65}\n")

    for r in results:
        print(f"  {'─' * 55}")
        print(f"  📌 {r['ticker']}  |  {r['signal_type']}  |  {r['last_date']}")
        print(f"  {'─' * 55}")
        print(f"    Close:        ${r['close']}")
        print(f"    9 EMA:        ${r['ema9']}")
        print(f"    21 EMA:       ${r['ema21']}")
        print(
            f"    Volume Ratio: {r['vol_ratio']}x avg  {'✅' if r['vol_confirmed'] else '❌'}"
        )
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

    tickers = args.tickers or DEFAULT_TICKERS
    from_date = (
        datetime.strptime(args.from_date, "%Y-%m-%d") if args.from_date else None
    )

    run_scanner(tickers, from_date=from_date)
