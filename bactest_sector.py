"""
Backtest: Sector Scanner EMA 5/12 Strategy (5-Minute Bars - SIMPLIFIED)
=======================================================================
SIMPLIFIED: Just picks top stocks in top ETF sectors.

Strategy:
  1. Find top 3 performing ETFs (sectors)
  2. For each, pick top 10 performing stocks
  3. Entry if 5EMA > 12EMA at 9:45

Usage:
  python bactest_sector.py --start 2026-06-15 --end 2026-06-20
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
import time

import pandas as pd
import numpy as np
from tabulate import tabulate

# Alpaca imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("ERROR: alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

RISK_PER_TRADE = 100.0
SLIPPAGE_PCT = 0.0005
TARGET1_R = 1.0
TARGET2_R = 2.0
TRAIL_HARD_R = 3.0
T1_SIZE = 0.50
T2_SIZE = 0.50
EMA_FAST = 5
EMA_SLOW = 12
TIMEFRAME = "5Min"

# Scan parameters
ETF_GAP_MIN = 0.5  # Minimum ETF gain to consider (0.5%)
STOCK_GAP_MIN = 0.5  # Minimum stock gain to consider (0.5%)
STOCK_VOL_MIN = 50_000  # Minimum volume
USE_RVOL = False  # Disabled for simplicity
TOP_ETF_SECTORS = 3  # Top ETFs to trade
TOP_STOCKS_SCAN = 10  # Top stocks per ETF
MAX_STOCKS_DAY = 10

# Alpaca API Keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_PAPER = True

DEBUG = False
EST = ZoneInfo("America/New_York")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── SECTOR MAP ────────────────────────────────────────────────────────────

SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM", "INTC", "QCOM", "TXN", "MU", "NOW", "AMAT",
                   "KLAC", "LRCX", "ADI", "MRVL", "PANW", "SNPS", "CDNS"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "PFE", "GILD", "ISRG",
                   "VRTX", "REGN", "CI", "CVS", "HCA", "MDT", "IQV"],
    "Financials": ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW", "CB", "PGR", "AON", "USB", "TFC",
                   "COF", "AIG", "MET", "PRU", "ICE"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "ORLY", "GM", "F",
                               "ABNB", "YUM", "DRI", "HLT", "MAR", "RCL", "CCL", "TSCO"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "CHTR", "EA", "TTWO", "TMUS", "FOXA",
                               "OMC", "AKAM", "NYT", "WBD", "NWSA", "LYV", "MTCH", "ZM"],
    "Industrials": ["RTX", "HON", "UNP", "CAT", "DE", "BA", "LMT", "GE", "MMM", "FDX", "UPS", "EMR", "ETN", "PH", "CMI",
                    "CTAS", "NSC", "CSX", "CARR", "TDG"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "GIS", "KHC", "KMB", "SYY", "HSY",
                         "KDP", "CAG", "CPB", "MKC", "CHD", "CLX"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL", "DVN", "BKR", "APA", "EQT", "KMI",
               "WMB", "TRGP", "NOG", "OKE", "FANG"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "ETR", "PCG", "WEC", "ES", "AWK", "DTE",
                  "CMS", "CNP", "NI", "AES", "PPL"],
    "Real Estate": ["PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB", "EQR", "VTR", "WY", "ARE",
                    "BXP", "KIM", "NNN", "HST", "EXR", "CUBE"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM", "CF", "MOS", "ALB", "RPM", "PKG",
                  "IP", "BALL", "SON", "GEF", "SLGN"],
}
SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Communication Services": "XLC",
    "Industrials": "XLI", "Consumer Staples": "XLP", "Energy": "XLE",
    "Utilities": "XLU", "Real Estate": "XLRE", "Materials": "XLB",
}
TICKER_SECTOR = {t: s for s, ts in SECTOR_TICKERS.items() for t in ts}
ALL_TICKERS = sorted(TICKER_SECTOR.keys())
ALL_ETFS = list(SECTOR_ETF.values())


# ─── ALPACA DATA FETCHER ──────────────────────────────────────────────────────

class AlpacaDataFetcher:
    """Fetches historical data from Alpaca API."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not api_key or not secret_key:
            raise ValueError("Alpaca API keys are required.")

        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def fetch_intraday_bars(self, symbols: List[str], start: date, end: date,
                            timeframe: str = "5Min") -> Dict[str, pd.DataFrame]:
        """Fetch intraday bars from Alpaca."""
        result: Dict[str, pd.DataFrame] = {}

        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "10Min": TimeFrame(10, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        }
        tf = tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

        CHUNK_SIZE = 50

        for i in range(0, len(symbols), CHUNK_SIZE):
            chunk = symbols[i:i + CHUNK_SIZE]

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=tf,
                    start=start,
                    end=end,
                    adjustment=Adjustment.ALL,
                    limit=10000
                )

                bars = self.data_client.get_stock_bars(request)

                for sym in chunk:
                    try:
                        sym_bars = bars.data.get(sym)
                        if sym_bars and len(sym_bars) > 0:
                            df = pd.DataFrame([{
                                'open': bar.open,
                                'high': bar.high,
                                'low': bar.low,
                                'close': bar.close,
                                'volume': bar.volume,
                                'timestamp': bar.timestamp
                            } for bar in sym_bars])

                            df.set_index('timestamp', inplace=True)
                            df.index = pd.to_datetime(df.index).tz_convert('America/New_York')
                            result[sym] = df

                    except Exception as e:
                        log.debug("Error processing %s: %s", sym, e)

            except Exception as e:
                log.warning("Error fetching chunk: %s", e)
                for sym in chunk:
                    try:
                        request = StockBarsRequest(
                            symbol_or_symbols=sym,
                            timeframe=tf,
                            start=start,
                            end=end,
                            adjustment=Adjustment.ALL,
                            limit=10000
                        )
                        bars = self.data_client.get_stock_bars(request)
                        sym_bars = bars.data.get(sym)
                        if sym_bars and len(sym_bars) > 0:
                            df = pd.DataFrame([{
                                'open': bar.open,
                                'high': bar.high,
                                'low': bar.low,
                                'close': bar.close,
                                'volume': bar.volume,
                                'timestamp': bar.timestamp
                            } for bar in sym_bars])
                            df.set_index('timestamp', inplace=True)
                            df.index = pd.to_datetime(df.index).tz_convert('America/New_York')
                            result[sym] = df
                    except Exception as e2:
                        log.debug("Individual fetch failed for %s: %s", sym, e2)

            time.sleep(0.5)

        return result


# ─── DAY DATA ────────────────────────────────────────────────────────────────

class DayData:
    """Holds all fetched data for one trading day."""

    def __init__(self):
        self.prior_closes: dict[str, float] = {}
        self.pm_ohlcv: dict[str, dict] = {}
        self.etf_gaps: dict[str, float] = {}
        self.intraday_bars: dict[str, pd.DataFrame] = {}


def fetch_day_data(scan_date: date, data_fetcher: AlpacaDataFetcher) -> DayData:
    """Fetch all data for one trading day using Alpaca."""
    dd = DayData()

    # Fetch 5 days of data to ensure enough bars for EMA calculation
    start_date = scan_date - timedelta(days=5)
    end_date = scan_date + timedelta(days=1)

    all_syms = ALL_TICKERS + ALL_ETFS
    log.debug("Fetching 5-min bars for %d symbols", len(all_syms))

    bars_data = data_fetcher.fetch_intraday_bars(all_syms, start_date, end_date, TIMEFRAME)

    for sym, df in bars_data.items():
        if df.empty:
            continue

        # Get all bars up to and including scan_date
        day_bars = df[df.index.date <= scan_date]

        # Filter to RTH: 9:30 - 16:00
        rth = day_bars[(day_bars.index.hour >= 9) & (day_bars.index.hour < 16)]

        if rth.empty:
            continue

        # Store FULL bars (including previous days) for EMA calculation
        dd.intraday_bars[sym] = rth

        # Get prior close (use first bar's open)
        if len(rth) > 0:
            dd.prior_closes[sym] = float(rth.iloc[0]['open'])

        # Get PM data (9:30 to 9:45)
        pm = rth[(rth.index.hour == 9) & (rth.index.minute >= 30) & (rth.index.minute <= 45)]
        if not pm.empty and len(pm) >= 3:
            dd.pm_ohlcv[sym] = {
                "pm_open": float(pm.iloc[0]['open']),
                "pm_high": float(pm['high'].max()),
                "pm_low": float(pm['low'].min()),
                "pm_close": float(pm.iloc[-1]['close']),
                "pm_volume": float(pm['volume'].sum()),
            }

    # ETF gaps
    for sector, etf in SECTOR_ETF.items():
        pc = dd.prior_closes.get(etf)
        pm = dd.pm_ohlcv.get(etf, {})
        pmc = pm.get("pm_close") if pm else None
        if pc and pmc:
            dd.etf_gaps[sector] = (pmc - pc) / pc * 100
        else:
            dd.etf_gaps[sector] = 0.0

    return dd


# ─── SCAN LOGIC ──────────────────────────────────────────────────────────────

def run_scan(scan_date: date, dd: DayData, list_mode: str) -> list[dict]:
    """Run scanner logic for one day."""
    candidates = []

    # List A: Stock catalyst plays (keep this if you want)
    if list_mode in ("a", "both"):
        results = []
        for sym in ALL_TICKERS:
            pc = dd.prior_closes.get(sym)
            pm = dd.pm_ohlcv.get(sym, {})
            pmc = pm.get("pm_close") if pm else None

            if not pc or not pmc or pc == 0:
                continue

            gap_pct = (pmc - pc) / pc * 100
            pm_vol = pm.get("pm_volume", 0) if pm else 0
            sector = TICKER_SECTOR.get(sym, "")

            # Basic filters
            if gap_pct < STOCK_GAP_MIN:
                continue
            if pm_vol < STOCK_VOL_MIN:
                continue

            results.append({
                "symbol": sym,
                "sector": sector,
                "gap_pct": round(gap_pct, 2),
                "pm_volume": int(pm_vol),
                "confluence": round(gap_pct, 2),
                "entry_source": "List A",
            })

        # Take top 10 overall from List A
        if results:
            top10 = (pd.DataFrame(results)
                     .sort_values("gap_pct", ascending=False)
                     .head(TOP_STOCKS_SCAN)
                     .to_dict("records"))
            candidates.extend(top10)

    # List B: ETF momentum - SIMPLIFIED
    # 1. Find top performing sectors via ETFs
    # 2. Pick top performing stocks within those sectors
    if list_mode in ("b", "both"):
        # Sort ETFs by gap (best performing first)
        top_etf = sorted(dd.etf_gaps.items(), key=lambda x: -x[1])[:TOP_ETF_SECTORS]

        log.debug("Top ETFs: %s", [f"{s}: {g:.2f}%" for s, g in top_etf])

        for sector, etf_gap_pct in top_etf:
            if etf_gap_pct < ETF_GAP_MIN:  # Skip if ETF is down or flat
                continue

            rows = []
            # For each stock in this sector
            for sym in SECTOR_TICKERS.get(sector, []):
                pc = dd.prior_closes.get(sym)
                pm = dd.pm_ohlcv.get(sym, {})
                pmc = pm.get("pm_close") if pm else None

                if not pc or not pmc or pc == 0:
                    continue

                # Calculate stock's performance
                gap_pct = (pmc - pc) / pc * 100
                pm_vol = pm.get("pm_volume", 0) if pm else 0

                # Basic filters
                if gap_pct < STOCK_GAP_MIN:  # Minimum gain threshold
                    continue
                if pm_vol < STOCK_VOL_MIN:  # Minimum volume
                    continue

                # Stock score = just its gap percentage
                rows.append({
                    "symbol": sym,
                    "sector": sector,
                    "gap_pct": round(gap_pct, 2),
                    "pm_volume": int(pm_vol),
                    "etf_gap": round(etf_gap_pct, 2),
                    "confluence": round(gap_pct, 2),  # Just the stock's performance
                    "entry_source": "List B",
                })

            # Take top 10 stocks from this sector (by gap percentage)
            if rows:
                top10 = (pd.DataFrame(rows)
                         .sort_values("gap_pct", ascending=False)
                         .head(TOP_STOCKS_SCAN)
                         .to_dict("records"))
                candidates.extend(top10)

    # Deduplicate (same stock could appear in both lists)
    seen = set()
    unique = []
    for c in candidates:
        if c["symbol"] not in seen:
            seen.add(c["symbol"])
            unique.append(c)

    return unique[:MAX_STOCKS_DAY]


# ─── EMA HELPERS ─────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def get_ema_at_bar(bars: pd.DataFrame, bar_idx: int, period: int) -> float:
    """Compute EMA using all bars up to and including bar_idx."""
    if bar_idx < 0 or bar_idx >= len(bars):
        return float("nan")
    if bar_idx < period - 1:
        return float("nan")
    return float(ema(bars["close"].iloc[:bar_idx + 1], period).iloc[-1])


# ─── TRADE SIMULATION ────────────────────────────────────────────────────────

@dataclass
class Trade:
    date: str
    symbol: str
    sector: str
    entry_source: str
    gap_pct: float
    confluence: float

    entry_time: str = ""
    entry_price: float = 0.0
    ema5_at_entry: float = 0.0
    ema12_at_entry: float = 0.0
    shares: int = 0
    risk_per_share: float = 0.0
    stop_price: float = 0.0

    t1_price: float = 0.0
    t2_price: float = 0.0
    trail_stop: float = 0.0

    t1_hit: bool = False
    t2_hit: bool = False
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""

    pnl: float = 0.0
    pnl_r: float = 0.0
    return_pct: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


def simulate_trade(sym: str, bars: pd.DataFrame, scan_date: date,
                   meta: dict) -> Trade:
    """Simulate one trade on intraday 5-min bars."""
    t = Trade(
        date=scan_date.isoformat(), symbol=sym,
        sector=meta.get("sector", ""),
        entry_source=meta.get("entry_source", ""),
        gap_pct=meta.get("gap_pct", 0),
        confluence=meta.get("confluence", 0),
    )

    if bars.empty or len(bars) < EMA_SLOW + 5:
        t.skipped, t.skip_reason = True, f"insufficient bars ({len(bars)} < {EMA_SLOW + 5})"
        return t

    # Find the 9:45 bar on the scan date
    entry_bar_idx = None
    for i, idx in enumerate(bars.index):
        if idx.date() == scan_date and idx.hour == 9 and idx.minute == 45:
            entry_bar_idx = i
            break

    if entry_bar_idx is None:
        t.skipped, t.skip_reason = True, "no 9:45 bar on scan date"
        return t

    if entry_bar_idx < EMA_SLOW - 1:
        t.skipped, t.skip_reason = True, f"only {entry_bar_idx + 1} bars before entry (need {EMA_SLOW})"
        return t

    # EMA at entry
    e5 = get_ema_at_bar(bars, entry_bar_idx, EMA_FAST)
    e12 = get_ema_at_bar(bars, entry_bar_idx, EMA_SLOW)

    if np.isnan(e5) or np.isnan(e12):
        t.skipped, t.skip_reason = True, "EMA calculation failed"
        return t

    t.ema5_at_entry = round(e5, 4)
    t.ema12_at_entry = round(e12, 4)

    # Entry condition: 5 EMA > 12 EMA
    if e5 <= e12:
        t.skipped, t.skip_reason = True, f"5EMA({e5:.2f}) ≤ 12EMA({e12:.2f})"
        return t

    # Entry price = close of 9:45 bar + slippage
    raw_entry = float(bars["close"].iloc[entry_bar_idx])
    entry = raw_entry * (1 + SLIPPAGE_PCT)

    # Stop = 12 EMA at entry bar
    stop = e12

    risk_per_share = entry - stop
    if risk_per_share <= 0:
        t.skipped, t.skip_reason = True, "stop >= entry"
        return t

    shares = int(RISK_PER_TRADE / risk_per_share)
    if shares < 1:
        t.skipped, t.skip_reason = True, f"0 shares"
        return t

    t.entry_price = round(entry, 4)
    t.entry_time = str(bars.index[entry_bar_idx])
    t.shares = shares
    t.risk_per_share = round(risk_per_share, 4)
    t.stop_price = round(stop, 4)
    t.t1_price = round(entry + TARGET1_R * risk_per_share, 4)
    t.t2_price = round(entry + TARGET2_R * risk_per_share, 4)
    t.trail_stop = round(entry - TRAIL_HARD_R * risk_per_share, 4)

    # Simulate bar by bar
    shares_remaining = shares
    shares_t1 = int(shares * T1_SIZE)
    shares_t2 = int((shares - shares_t1) * T2_SIZE)

    realized_pnl = 0.0

    for i in range(entry_bar_idx + 1, len(bars)):
        bar = bars.iloc[i]
        btime = bars.index[i]

        if btime.date() < scan_date:
            continue

        cur_e12 = get_ema_at_bar(bars, i, EMA_SLOW)

        if np.isnan(cur_e12):
            continue

        hi = float(bar["high"])
        lo = float(bar["low"])
        close = float(bar["close"])

        # EOD forced exit
        is_last = (i == len(bars) - 1) or (btime.hour >= 15 and btime.minute >= 55)

        if is_last and shares_remaining > 0:
            exit_price = close * (1 - SLIPPAGE_PCT)
            realized_pnl += shares_remaining * (exit_price - entry)
            t.exit_price = round(exit_price, 4)
            t.exit_time = str(btime)
            t.exit_reason = "EOD"
            break

        # Target 1
        if not t.t1_hit and hi >= t.t1_price and shares_t1 > 0:
            realized_pnl += shares_t1 * (t.t1_price * (1 - SLIPPAGE_PCT) - entry)
            shares_remaining -= shares_t1
            t.t1_hit = True
            t.trail_stop = max(t.trail_stop, entry)

        # Target 2
        if t.t1_hit and not t.t2_hit and hi >= t.t2_price and shares_t2 > 0:
            realized_pnl += shares_t2 * (t.t2_price * (1 - SLIPPAGE_PCT) - entry)
            shares_remaining -= shares_t2
            t.t2_hit = True
            t.trail_stop = max(t.trail_stop, t.t1_price)

        # Stop
        stop_triggered = (close < cur_e12) or (lo <= t.trail_stop)
        if stop_triggered and shares_remaining > 0:
            if lo <= t.trail_stop:
                exit_price = t.trail_stop * (1 - SLIPPAGE_PCT)
            else:
                exit_price = cur_e12 * (1 - SLIPPAGE_PCT)
            exit_price = max(lo, exit_price)
            realized_pnl += shares_remaining * (exit_price - entry)
            t.exit_price = round(exit_price, 4)
            t.exit_time = str(btime)
            t.exit_reason = "Trail Stop" if t.t1_hit else "Stop (12EMA)"
            shares_remaining = 0
            break

    t.pnl = round(realized_pnl, 2)
    t.pnl_r = round(realized_pnl / RISK_PER_TRADE, 3)
    t.return_pct = round(realized_pnl / (entry * shares) * 100, 3) if shares > 0 else 0
    return t


# ─── BACKTEST RUNNER ─────────────────────────────────────────────────────────

def run_backtest(start_date: date, end_date: date, list_mode: str) -> list[Trade]:
    all_trades: list[Trade] = []

    data_fetcher = AlpacaDataFetcher(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER)

    days = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    log.info("=" * 80)
    log.info("BACKTEST: Sector Scanner EMA 5/12 Strategy (5-Minute Bars)")
    log.info("=" * 80)
    log.info("Period: %s → %s", start_date, end_date)
    log.info("Timeframe: %s | List: %s | Risk: $%.0f/trade", TIMEFRAME, list_mode, RISK_PER_TRADE)
    log.info("Universe: %d stocks across %d sectors", len(ALL_TICKERS), len(SECTOR_TICKERS))
    log.info("")

    for scan_date in days:
        log.info("─" * 70)
        log.info("  %s", scan_date)

        try:
            dd = fetch_day_data(scan_date, data_fetcher)
        except Exception as exc:
            log.warning("  Data fetch failed: %s", exc)
            if DEBUG:
                import traceback
                traceback.print_exc()
            continue

        if not dd.pm_ohlcv:
            log.warning("  No PM data — skipping")
            continue

        candidates = run_scan(scan_date, dd, list_mode)
        log.info("  Scan → %d candidates", len(candidates))

        day_trades = 0
        for meta in candidates:
            sym = meta["symbol"]
            bars = dd.intraday_bars.get(sym, pd.DataFrame())

            trade = simulate_trade(sym, bars, scan_date, meta)
            all_trades.append(trade)

            if not trade.skipped:
                day_trades += 1
                status = f"P&L=${trade.pnl:+.2f} ({trade.pnl_r:+.2f}R)  exit={trade.exit_reason}"
            else:
                status = f"SKIP: {trade.skip_reason}"
            log.info("    %-8s  %s", sym, status)

        log.info("  Day trades taken: %d  |  skipped: %d",
                 day_trades, len(candidates) - day_trades)

    return all_trades


# ─── STATISTICS ──────────────────────────────────────────────────────────────

def build_report(trades: list[Trade], start_date: date, end_date: date,
                 list_mode: str) -> str:
    executed = [t for t in trades if not t.skipped]
    skipped = [t for t in trades if t.skipped]

    if not executed:
        return "No executed trades to report."

    df = pd.DataFrame([t.__dict__ for t in executed])
    df["date"] = pd.to_datetime(df["date"])

    total_pnl = df["pnl"].sum()
    total_r = df["pnl_r"].sum()
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / len(df) * 100
    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0
    avg_win_r = wins["pnl_r"].mean() if len(wins) else 0
    avg_loss_r = losses["pnl_r"].mean() if len(losses) else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else float("inf")
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
    t1_rate = df["t1_hit"].mean() * 100
    t2_rate = df["t2_hit"].mean() * 100

    cum_pnl = df.sort_values("date")["pnl"].cumsum()
    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl - roll_max
    max_dd = drawdown.min()

    daily_pnl = df.groupby("date")["pnl"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)

    best = df.loc[df["pnl"].idxmax()]
    worst = df.loc[df["pnl"].idxmin()]

    exit_counts = df["exit_reason"].value_counts()

    sector_stats = (df.groupby("sector")
                    .agg(trades=("pnl", "count"),
                         total_pnl=("pnl", "sum"),
                         win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                         avg_r=("pnl_r", "mean"))
                    .sort_values("total_pnl", ascending=False))

    lines = []
    W = 72

    def hdr(title):
        lines.append(f"\n{'═' * W}\n  {title}\n{'═' * W}")

    def sub(title):
        lines.append(f"\n── {title} {'─' * (W - len(title) - 4)}")

    hdr(f"BACKTEST REPORT  |  {start_date} → {end_date}  |  List: {list_mode.upper()}")
    lines.append(f"""
  Strategy : 5 EMA / 12 EMA crossover at 9:45 AM EST
  Timeframe: {TIMEFRAME} bars
  Scanner  : Top {TOP_ETF_SECTORS} ETFs → top {TOP_STOCKS_SCAN} stocks in each
  Risk     : ${RISK_PER_TRADE:.0f}/trade  |  Slippage: {SLIPPAGE_PCT * 100:.2f}% per side
  Exits    : T1={TARGET1_R}R (50%)  T2={TARGET2_R}R (25%)  Trail={TRAIL_HARD_R}R hard stop (25%)
  Stop     : Close below 12 EMA  OR  {TRAIL_HARD_R}R hard stop""")

    sub("OVERALL PERFORMANCE")
    lines.append(tabulate([
        ["Total P&L", f"${total_pnl:+,.2f}"],
        ["Total R", f"{total_r:+.2f}R"],
        ["Trades Executed", len(executed)],
        ["Trades Skipped", len(skipped)],
        ["Win Rate", f"{win_rate:.1f}%"],
        ["Profit Factor", f"{profit_factor:.2f}"],
        ["Expectancy/Trade", f"${expectancy:+.2f}"],
        ["Avg Win", f"${avg_win:+.2f}  ({avg_win_r:+.2f}R)"],
        ["Avg Loss", f"${avg_loss:+.2f}  ({avg_loss_r:+.2f}R)"],
        ["Max Drawdown", f"${max_dd:,.2f}"],
        ["Daily Sharpe", f"{sharpe:.2f}"],
        ["T1 Hit Rate", f"{t1_rate:.1f}%"],
        ["T2 Hit Rate", f"{t2_rate:.1f}%"],
        ["Best Trade", f"{best['symbol']} {best['date'].date()}  ${best['pnl']:+.2f}  ({best['pnl_r']:+.2f}R)"],
        ["Worst Trade", f"{worst['symbol']} {worst['date'].date()}  ${worst['pnl']:+.2f}  ({worst['pnl_r']:+.2f}R)"],
    ], tablefmt="simple", colalign=("left", "right")))

    sub("EXIT REASON BREAKDOWN")
    lines.append(tabulate(
        [[r, c, f"{c / len(df) * 100:.1f}%"] for r, c in exit_counts.items()],
        headers=["Exit Reason", "Count", "% of Trades"],
        tablefmt="simple",
    ))

    sub("PERFORMANCE BY LIST SOURCE")
    if "entry_source" in df.columns:
        src_stats = (df.groupby("entry_source")
                     .agg(trades=("pnl", "count"),
                          total_pnl=("pnl", "sum"),
                          win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                          avg_r=("pnl_r", "mean"))
                     .sort_values("total_pnl", ascending=False))
        lines.append(src_stats.to_string())

    sub("PERFORMANCE BY SECTOR")
    lines.append(sector_stats.to_string())

    sub("SKIP REASON BREAKDOWN")
    if skipped:
        skip_df = pd.DataFrame([{"reason": t.skip_reason, "symbol": t.symbol}
                                for t in skipped])
        lines.append(skip_df["reason"].value_counts().to_string())
    else:
        lines.append("  No skips.")

    sub("MONTHLY P&L")
    monthly = (df.groupby(df["date"].dt.to_period("M"))["pnl"]
               .agg(["sum", "count", lambda x: (x > 0).mean() * 100])
               .rename(columns={"sum": "P&L", "count": "Trades", "<lambda_0>": "WinRate%"}))
    lines.append(monthly.to_string())

    return "\n".join(lines)


# ─── CSV EXPORT ──────────────────────────────────────────────────────────────

def save_trades_csv(trades: list[Trade], path: str):
    rows = [t.__dict__ for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Trade ledger saved → %s", path)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Backtest: Sector Scanner EMA 5/12 Strategy")
    p.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--list", default="both", choices=["a", "b", "both"],
                   help="a=stock catalyst, b=ETF momentum, both (default)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    global DEBUG
    args = parse_args()
    DEBUG = args.debug
    if DEBUG: logging.getLogger().setLevel(logging.DEBUG)

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("\n" + "=" * 80)
        print("ERROR: Alpaca API credentials not set!")
        print("=" * 80)
        print("\nPlease set your Alpaca API keys as environment variables:")
        print("  export ALPACA_API_KEY='your_api_key'")
        print("  export ALPACA_SECRET_KEY='your_secret_key'")
        print("\nGet your API keys from: https://app.alpaca.markets/paper/dashboard/overview")
        print("=" * 80 + "\n")
        sys.exit(1)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    trades = run_backtest(start, end, args.list)

    report = build_report(trades, start, end, args.list)
    print(report)

    out_csv = f"backtest_alpaca_5min_{args.start}_{args.end}_{args.list}.csv"
    save_trades_csv(trades, out_csv)

    out_txt = f"backtest_alpaca_5min_{args.start}_{args.end}_{args.list}_report.txt"
    with open(out_txt, "w") as f:
        f.write(report)
    log.info("Report saved → %s", out_txt)


if __name__ == "__main__":
    main()