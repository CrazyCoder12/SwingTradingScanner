"""
S&P 500 Premarket Sector Scanner  (9:30–9:35 AM EST open window)
=================================================================
Data sources:
  • Alpaca Snapshot API  → prevDailyBar (prior close) + dailyBar (PM OHLCV)
    ─ dailyBar covers 4:00 AM EST of scan_date up to call time
    ─ prevDailyBar is the prior RTH session close (reliable)
  • 1-min premarket bars → accurate VWAP for tickers with PM volume
  • 1-min RTH bars       → 9:30–9:35 open-window performance
  • Sector ETF snapshot  → confirms sector move is broad, not just 1-2 stocks

Data access:
  Feed coverage and historical availability depend on your Alpaca subscription.
  Select ALPACA_DATA_FEED to match the feed enabled for your account.

Filters:
  ✓  PM gap    >= 2%     (pm_close vs prev_close)
  ✓  PM vol    >= 100K
  ✓  PM last   >= PM VWAP
  ✓  RVOL      >= 2x     [optional --no-rvol]
  ✓  Sector ETF gap >= 0% (sector itself must be green in PM) [optional --no-etf]

Scored output:
  Each stock gets a confluence_score (0–100) combining:
    gap %, RVOL, VWAP position, PM range extension, sector ETF alignment
  gap_quality: STRONG / MODERATE / WEAK based on confluence

Two watchlists per run:
  LIST A — Stock Catalyst Plays  : stocks with individual gap ≥2% + all filters
            Best for: earnings/news-driven options (directional, shorter DTE)
  LIST B — ETF Sector Momentum   : top 3 ETF sectors by PM gap, top 10 stocks
            per sector ranked by how much they're leading the ETF move
            Filters relaxed (gap ≥0.5%, vol ≥50K) — sector tide lifting all boats
            Best for: sector rotation plays, wider-strike options, 1–2 week DTE

Usage:
  python -m swing_trading_scanner.scanners.sector --date 2026-06-24
  python -m swing_trading_scanner.scanners.sector --start 2026-06-16 --end 2026-06-20
  python -m swing_trading_scanner.scanners.sector --date 2026-06-24 --no-rvol
  python -m swing_trading_scanner.scanners.sector --date 2026-06-24 --no-etf
  python -m swing_trading_scanner.scanners.sector --date 2026-06-24 --debug --csv

Env vars:  ALPACA_API_KEY  ALPACA_SECRET_KEY  ALPACA_DATA_FEED (iex|sip)
"""

from swing_trading_scanner.config import load_environment

load_environment()


import os, sys, argparse, logging
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import pytz
import requests
from tabulate import tabulate
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ─── CONFIG ──────────────────────────────────────────────────────────────────

API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

PREMARKET_GAP_MIN = 0.02  # 2% minimum gap vs prior close
PREMARKET_VOL_MIN = 100_000  # minimum premarket shares
USE_RVOL = True
RVOL_MIN = 2.0
USE_ETF_FILTER = True  # sector ETF must also be green in PM
TOP_SECTORS = 3
TOP_STOCKS_PER_SECT = 10
TOP_ETF_SECTORS = 3  # top N ETF sectors for List B
ETF_GAP_MIN = 0.005  # 0.5% min gap for ETF sector stocks (List B)
ETF_VOL_MIN = 50_000  # relaxed volume floor for List B

DEBUG = False

EST = pytz.timezone("US/Eastern")
UTC = pytz.utc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── SECTOR MAP ──────────────────────────────────────────────────────────────

SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "AMD",
        "ORCL",
        "CRM",
        "INTC",
        "QCOM",
        "TXN",
        "MU",
        "NOW",
        "AMAT",
        "KLAC",
        "LRCX",
        "ADI",
        "MRVL",
        "PANW",
        "SNPS",
        "CDNS",
    ],
    "Healthcare": [
        "UNH",
        "JNJ",
        "LLY",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "AMGN",
        "PFE",
        "GILD",
        "ISRG",
        "VRTX",
        "REGN",
        "CI",
        "CVS",
        "HCA",
        "MDT",
        "IQV",
    ],
    "Financials": [
        "JPM",
        "V",
        "MA",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "AXP",
        "BLK",
        "SCHW",
        "CB",
        "PGR",
        "MMC",
        "USB",
        "TFC",
        "COF",
        "AIG",
        "MET",
        "PRU",
        "ICE",
    ],
    "Consumer Discretionary": [
        "AMZN",
        "TSLA",
        "HD",
        "MCD",
        "NKE",
        "LOW",
        "SBUX",
        "BKNG",
        "TJX",
        "ORLY",
        "GM",
        "F",
        "ABNB",
        "YUM",
        "DRI",
        "HLT",
        "MAR",
        "RCL",
        "CCL",
        "TSCO",
    ],
    "Communication Services": [
        "GOOGL",
        "META",
        "NFLX",
        "DIS",
        "T",
        "VZ",
        "CMCSA",
        "CHTR",
        "EA",
        "TTWO",
        "TMUS",
        "FOXA",
        "OMC",
        "IPG",
        "NYT",
        "WBD",
        "PARA",
        "LYV",
        "MTCH",
        "ZM",
    ],
    "Industrials": [
        "RTX",
        "HON",
        "UNP",
        "CAT",
        "DE",
        "BA",
        "LMT",
        "GE",
        "MMM",
        "FDX",
        "UPS",
        "EMR",
        "ETN",
        "PH",
        "CMI",
        "CTAS",
        "NSC",
        "CSX",
        "CARR",
        "TDG",
    ],
    "Consumer Staples": [
        "PG",
        "KO",
        "PEP",
        "COST",
        "WMT",
        "PM",
        "MO",
        "MDLZ",
        "CL",
        "GIS",
        "KHC",
        "KMB",
        "SYY",
        "HSY",
        "K",
        "CAG",
        "CPB",
        "MKC",
        "CHD",
        "CLX",
    ],
    "Energy": [
        "XOM",
        "CVX",
        "COP",
        "EOG",
        "SLB",
        "MPC",
        "PSX",
        "VLO",
        "OXY",
        "HES",
        "HAL",
        "DVN",
        "BKR",
        "APA",
        "MRO",
        "EQT",
        "KMI",
        "WMB",
        "TRGP",
        "NOG",
    ],
    "Utilities": [
        "NEE",
        "DUK",
        "SO",
        "D",
        "AEP",
        "EXC",
        "SRE",
        "XEL",
        "ED",
        "ETR",
        "PCG",
        "WEC",
        "ES",
        "AWK",
        "DTE",
        "CMS",
        "CNP",
        "NI",
        "AES",
        "PPL",
    ],
    "Real Estate": [
        "PLD",
        "AMT",
        "EQIX",
        "CCI",
        "PSA",
        "SPG",
        "O",
        "WELL",
        "DLR",
        "AVB",
        "EQR",
        "VTR",
        "WY",
        "ARE",
        "BXP",
        "KIM",
        "NNN",
        "HST",
        "EXR",
        "CUBE",
    ],
    "Materials": [
        "LIN",
        "APD",
        "SHW",
        "ECL",
        "DD",
        "NEM",
        "FCX",
        "NUE",
        "VMC",
        "MLM",
        "CF",
        "MOS",
        "ALB",
        "RPM",
        "PKG",
        "IP",
        "SEE",
        "SON",
        "GEF",
        "SLGN",
    ],
}

# Sector ETFs — used to confirm sector-level PM momentum
SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

TICKER_SECTOR: dict[str, str] = {}
for _s, _ts in SECTOR_TICKERS.items():
    for _t in _ts:
        TICKER_SECTOR[_t] = _s

ALL_TICKERS = sorted(TICKER_SECTOR.keys())
ALL_ETFS = list(SECTOR_ETF.values())

# ─── CLIENT ──────────────────────────────────────────────────────────────────

client = None

# ─── HELPERS ─────────────────────────────────────────────────────────────────


def to_utc(d: date, hour: int, minute: int = 0) -> datetime:
    return EST.localize(datetime(d.year, d.month, d.day, hour, minute)).astimezone(UTC)


def prev_trading_day(d: date) -> date:
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def get_sym(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        try:
            return df.xs(sym, level="symbol")
        except KeyError:
            return pd.DataFrame()
    return df


def vwap_from_bars(bars: pd.DataFrame) -> Optional[float]:
    if bars.empty:
        return None
    vol = bars["volume"]
    total = vol.sum()
    if total == 0:
        return None
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    return float((typical * vol).sum() / total)


def fetch_bars_batch(
    symbols: list[str], start: datetime, end: datetime, timeframe: TimeFrame
) -> pd.DataFrame:
    CHUNK = 100
    frames = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=DATA_FEED,
            )
            df = client.get_stock_bars(req).df
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            log.debug("bars chunk error: %s", exc)
    return pd.concat(frames) if frames else pd.DataFrame()


# ─── SNAPSHOT API ────────────────────────────────────────────────────────────
#
#  Returns per ticker:
#    prevDailyBar → prior RTH close  (scan_date - 1 trading day)
#    dailyBar     → current day bars from 4:00 AM EST = PREMARKET data
#
#  Uses the configured data feed; account entitlements determine access.


def fetch_snapshots(symbols: list[str]) -> dict[str, dict]:
    BASE = "https://data.alpaca.markets/v2/stocks/snapshots"
    headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}
    result: dict[str, dict] = {}
    CHUNK = 100

    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        params = {"symbols": ",".join(chunk), "feed": DATA_FEED}
        try:
            resp = requests.get(BASE, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("snapshot error chunk %d: %s", i // CHUNK, exc)
            continue

        for sym, snap in data.items():
            try:
                prev = snap.get("prevDailyBar") or {}
                day = snap.get("dailyBar") or {}

                prev_close = prev.get("c") or prev.get("cl")
                if not prev_close:
                    continue

                pm_open = day.get("o") or day.get("op")
                pm_high = day.get("h") or day.get("hi")
                pm_low = day.get("l") or day.get("lo")
                pm_close = day.get("c") or day.get("cl")
                pm_volume = day.get("v") or day.get("vl") or 0

                result[sym] = {
                    "prev_close": float(prev_close),
                    "pm_open": float(pm_open) if pm_open else None,
                    "pm_high": float(pm_high) if pm_high else None,
                    "pm_low": float(pm_low) if pm_low else None,
                    "pm_close": float(pm_close) if pm_close else None,
                    "pm_volume": float(pm_volume),
                }
            except Exception as exc:
                log.debug("snapshot parse %s: %s", sym, exc)

    return result


def avg_daily_volume_20d(symbol: str, scan_date: date) -> Optional[float]:
    prev = prev_trading_day(scan_date)
    start_d = scan_date - timedelta(days=35)
    try:
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime(start_d.year, start_d.month, start_d.day, tzinfo=UTC),
            end=datetime(prev.year, prev.month, prev.day, 23, 59, tzinfo=UTC),
            feed=DATA_FEED,
        )
        df = client.get_stock_bars(req).df
        if df.empty:
            return None
        sym_df = get_sym(df, symbol)
        last20 = sym_df["volume"].tail(20)
        return float(last20.mean()) if len(last20) >= 5 else None
    except Exception:
        return None


# ─── GAP QUALITY SCORING ─────────────────────────────────────────────────────
#
#  A gap is BULLISH when:
#    1. Gap % is large relative to typical daily range  → big gap = real move
#    2. Price is above PM VWAP                          → buyers in control all session
#    3. PM volume is unusually high (RVOL)              → institutional participation
#    4. PM range is tight / not extended                → not exhausted at open
#    5. Sector ETF is also gapping up                   → macro tailwind, not just 1 stock
#
#  A gap is WEAK / FADE RISK when:
#    • Price gapped up but is trading BELOW PM VWAP     → sellers absorbed the gap
#    • PM high is far above PM close (faded during PM)  → early buyers trapped
#    • Low volume on the gap                            → no conviction
#    • Sector ETF is flat or red                        → idiosyncratic / no follow-through
#
#  confluence_score (0–100):
#    gap_pts    : 0–30  (scaled: 2%=5pts, 5%=15pts, 10%+=30pts)
#    rvol_pts   : 0–25  (scaled: 2x=10pts, 5x=20pts, 10x+=25pts)
#    vwap_pts   : 0–20  (above vwap = 20, below = 0)
#    range_pts  : 0–15  (tight PM range = higher score; extended = lower)
#    etf_pts    : 0–10  (sector ETF also up = 10, flat = 5, down = 0)


def confluence_score(
    gap_pct: float, rvol, above_vwap: bool, pm_range_pct: float, etf_gap_pct: float
) -> int:
    # Gap points (0–30)
    gap_pts = min(30, gap_pct / 10 * 30)  # 10% gap = full 30pts

    # RVOL points (0–25)
    if rvol is None or rvol == "N/A":
        rvol_pts = 10  # neutral if unknown
    else:
        rvol_pts = min(25, float(rvol) / 10 * 25)

    # VWAP position (0–20)
    vwap_pts = 20 if above_vwap else 0

    # PM range extension (0–15)
    # A tight range (< 2%) = good, extended (> 8%) = fade risk
    if pm_range_pct <= 2:
        range_pts = 15
    elif pm_range_pct <= 5:
        range_pts = 10
    elif pm_range_pct <= 8:
        range_pts = 5
    else:
        range_pts = 0

    # Sector ETF alignment (0–10)
    if etf_gap_pct > 0.5:
        etf_pts = 10
    elif etf_gap_pct > 0:
        etf_pts = 5
    else:
        etf_pts = 0

    return int(gap_pts + rvol_pts + vwap_pts + range_pts + etf_pts)


def gap_quality_label(score: int, above_vwap: bool, pm_range_pct: float) -> str:
    """
    STRONG  : high score + above VWAP + not over-extended
    MODERATE: decent score but some weakness
    WEAK    : below VWAP, or very extended, or low confluence
    """
    if not above_vwap:
        return "WEAK ⚠"
    if pm_range_pct > 10:
        return "EXTENDED ⚠"
    if score >= 65:
        return "STRONG 🟢"
    if score >= 40:
        return "MODERATE 🟡"
    return "WEAK ⚠"


# ─── LIST B: ETF SECTOR MOMENTUM STOCKS ─────────────────────────────────────
#
#  For the top N sectors by ETF PM gap, find stocks that are:
#    • Above their prior close (positive PM gap, even small)
#    • Leading the ETF — their individual gap > sector ETF gap  (outperforming)
#    • Minimum PM volume 50K (relaxed vs List A)
#    • Above PM VWAP
#
#  Ranking metric: "relative strength vs ETF" = stock_gap - etf_gap
#  A stock up 2% when XLI is up 1.2% has +0.8% RS — it's pulling the sector.
#  These are the names most likely to continue if the sector keeps running.


def build_etf_sector_list(
    snaps: dict,
    etf_gaps: dict[str, float],
    pm_bars: pd.DataFrame,
    rth_bars: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict: sector → DataFrame of top 10 stocks for the top ETF sectors.
    Uses relaxed filters vs List A.
    """
    # Top ETF sectors by PM gap (descending)
    top_etf_sectors = sorted(etf_gaps.items(), key=lambda x: -x[1])[:TOP_ETF_SECTORS]

    etf_list: dict[str, pd.DataFrame] = {}

    for sector, etf_gap_pct in top_etf_sectors:
        if etf_gap_pct <= 0:
            continue  # only truly green sectors

        tickers = SECTOR_TICKERS.get(sector, [])
        rows = []

        for sym in tickers:
            snap = snaps.get(sym)
            if not snap:
                continue

            pc = snap["prev_close"]
            pm_vol = snap["pm_volume"]
            pm_close = snap["pm_close"]
            pm_open = snap["pm_open"]
            pm_high = snap["pm_high"]
            pm_low = snap["pm_low"]

            if not pm_close or pc == 0:
                continue

            gap_pct = (pm_close - pc) / pc * 100

            # Relaxed gap floor — any positive move qualifies
            if gap_pct < ETF_GAP_MIN * 100:
                continue
            if pm_vol < ETF_VOL_MIN:
                continue

            # VWAP check
            sym_pm = get_sym(pm_bars, sym)
            if not sym_pm.empty:
                pm_vwap_val = vwap_from_bars(sym_pm)
            elif pm_high and pm_low and pm_close and pm_vol > 0:
                pm_vwap_val = (pm_high + pm_low + pm_close) / 3
            else:
                pm_vwap_val = None

            above_vwap = (pm_close >= pm_vwap_val) if pm_vwap_val else True
            if not above_vwap:
                continue

            # RTH bars
            rth_sym = get_sym(rth_bars, sym)
            rth_open = float(rth_sym["open"].iloc[0]) if not rth_sym.empty else None
            rth_close = float(rth_sym["close"].iloc[-1]) if not rth_sym.empty else None
            open_gain = (rth_close - pc) / pc * 100 if rth_close else None

            # RS vs ETF — key ranking metric
            rs_vs_etf = round(gap_pct - etf_gap_pct, 2)

            # PM range
            pm_range_pct = (
                (pm_high - pm_open) / pm_open * 100
                if pm_open and pm_open > 0 and pm_high
                else 0.0
            )

            rows.append(
                {
                    "symbol": sym,
                    "sector": sector,
                    "prev_close": round(pc, 2),
                    "pm_open": round(pm_open, 2) if pm_open else None,
                    "pm_close": round(pm_close, 2),
                    "pm_vwap": round(pm_vwap_val, 2) if pm_vwap_val else None,
                    "gap_pct": round(gap_pct, 2),
                    "etf_gap_pct": round(etf_gap_pct, 2),
                    "rs_vs_etf": rs_vs_etf,  # positive = leading sector
                    "pm_range_pct": round(pm_range_pct, 2),
                    "pm_volume": int(pm_vol),
                    "rth_open": round(rth_open, 2) if rth_open else None,
                    "rth_close": round(rth_close, 2) if rth_close else None,
                    "open_gain_pct": round(open_gain, 2)
                    if open_gain is not None
                    else None,
                }
            )

        if rows:
            sector_df = (
                pd.DataFrame(rows)
                .sort_values("rs_vs_etf", ascending=False)  # leaders first
                .head(TOP_STOCKS_PER_SECT)
                .reset_index(drop=True)
            )
            etf_list[sector] = sector_df

    return etf_list


# ─── CORE SCAN ───────────────────────────────────────────────────────────────


def scan_day(scan_date: date) -> dict:
    log.info("=" * 64)
    log.info(
        "Scanning %s  |  feed=%s  |  top_sectors=%d  |  rvol=%s  |  etf_filter=%s",
        scan_date,
        DATA_FEED,
        TOP_SECTORS,
        f">={RVOL_MIN}x" if USE_RVOL else "off",
        "on" if USE_ETF_FILTER else "off",
    )

    tf1m = TimeFrame(1, TimeFrameUnit.Minute)

    # ── Step 1: Snapshots (stocks + ETFs) ─────────────────────────
    all_snap_syms = ALL_TICKERS + ALL_ETFS
    log.info(
        "Step 1/3  Fetching snapshots for %d symbols (stocks + ETFs)…",
        len(all_snap_syms),
    )
    snaps = fetch_snapshots(all_snap_syms)
    log.info("  Got snapshots for %d symbols", len(snaps))

    # ETF gap map  sector → ETF gap %
    etf_gaps: dict[str, float] = {}
    for sector, etf in SECTOR_ETF.items():
        snap = snaps.get(etf)
        if snap and snap["prev_close"] and snap["pm_close"]:
            etf_gaps[sector] = (
                (snap["pm_close"] - snap["prev_close"]) / snap["prev_close"] * 100
            )
        else:
            etf_gaps[sector] = 0.0

    if DEBUG:
        log.debug("ETF gaps: %s", {k: f"{v:+.2f}%" for k, v in etf_gaps.items()})

    # ── Step 2: PM 1-min bars (for accurate VWAP) ─────────────────
    pm_candidates = [s for s in ALL_TICKERS if snaps.get(s, {}).get("pm_volume", 0) > 0]
    log.info(
        "Step 2/3  Fetching PM 1-min bars for %d tickers with PM volume…",
        len(pm_candidates),
    )
    pm_bars = pd.DataFrame()
    if pm_candidates:
        pm_bars = fetch_bars_batch(
            pm_candidates,
            to_utc(scan_date, 4, 0),
            to_utc(scan_date, 9, 29),
            tf1m,
        )
        pm_count = (
            pm_bars.index.get_level_values("symbol").nunique()
            if not pm_bars.empty and isinstance(pm_bars.index, pd.MultiIndex)
            else 0
        )
        log.info("  1-min bars for %d tickers", pm_count)

    # ── Step 3: RTH open-window bars 9:30–9:35 ────────────────────
    log.info("Step 3/3  Fetching RTH 9:30–9:35 AM bars…")
    rth_bars = fetch_bars_batch(
        ALL_TICKERS,
        to_utc(scan_date, 9, 30),
        to_utc(scan_date, 9, 36),
        tf1m,
    )
    rth_count = (
        rth_bars.index.get_level_values("symbol").nunique()
        if not rth_bars.empty and isinstance(rth_bars.index, pd.MultiIndex)
        else 0
    )
    log.info("  RTH bars for %d tickers", rth_count)

    if rth_bars.empty:
        log.warning("No RTH bars — market closed on %s?", scan_date)
        return {}

    # ── Per-ticker processing ──────────────────────────────────────
    results = []
    skip = {
        "no_snap": 0,
        "no_rth": 0,
        "gap": 0,
        "pmvol": 0,
        "vwap": 0,
        "rvol": 0,
        "etf": 0,
    }
    rvol_cache: dict[str, Optional[float]] = {}

    for sym in ALL_TICKERS:
        snap = snaps.get(sym)
        if snap is None:
            skip["no_snap"] += 1
            continue

        pc = snap["prev_close"]
        pm_vol = snap["pm_volume"]
        pm_close = snap["pm_close"]
        pm_open = snap["pm_open"]
        pm_high = snap["pm_high"]
        pm_low = snap["pm_low"]
        sector = TICKER_SECTOR[sym]

        if not pm_close or pc == 0:
            skip["gap"] += 1
            continue

        gap_pct = (pm_close - pc) / pc

        # VWAP: prefer bar-series, fall back to snapshot HLC/3
        sym_pm = get_sym(pm_bars, sym)
        if not sym_pm.empty:
            pm_vwap_val = vwap_from_bars(sym_pm)
        elif pm_high and pm_low and pm_close and pm_vol > 0:
            pm_vwap_val = (pm_high + pm_low + pm_close) / 3
        else:
            pm_vwap_val = None

        above_vwap = (pm_close >= pm_vwap_val) if pm_vwap_val else True

        # PM range extension: how far did PM high go above PM open?
        if pm_open and pm_open > 0 and pm_high:
            pm_range_pct = (pm_high - pm_open) / pm_open * 100
        else:
            pm_range_pct = 0.0

        # RTH bars
        rth_sym = get_sym(rth_bars, sym)
        if rth_sym.empty:
            skip["no_rth"] += 1
            continue

        rth_open = float(rth_sym["open"].iloc[0])
        rth_high = float(rth_sym["high"].max())
        rth_close = float(rth_sym["close"].iloc[-1])
        open_gain = (rth_close - pc) / pc

        # ── FILTERS ──────────────────────────────────────────────
        if gap_pct < PREMARKET_GAP_MIN:
            skip["gap"] += 1
            if DEBUG:
                log.debug("SKIP %-8s  gap %.2f%%", sym, gap_pct * 100)
            continue

        if pm_vol < PREMARKET_VOL_MIN:
            skip["pmvol"] += 1
            if DEBUG:
                log.debug("SKIP %-8s  pm_vol %,.0f", sym, pm_vol)
            continue

        if pm_vwap_val and not above_vwap:
            skip["vwap"] += 1
            if DEBUG:
                log.debug(
                    "SKIP %-8s  below VWAP %.2f (pm_close %.2f)",
                    sym,
                    pm_vwap_val,
                    pm_close,
                )
            continue

        # Sector ETF filter
        etf_gap = etf_gaps.get(sector, 0.0)
        if USE_ETF_FILTER and etf_gap < 0:
            skip["etf"] += 1
            if DEBUG:
                log.debug("SKIP %-8s  sector ETF gap %.2f%%", sym, etf_gap)
            continue

        # RVOL (optional)
        rvol = None
        if USE_RVOL:
            if sym not in rvol_cache:
                rvol_cache[sym] = avg_daily_volume_20d(sym, scan_date)
            avg_vol = rvol_cache[sym]
            if avg_vol and avg_vol > 0:
                rvol = pm_vol / (avg_vol * 0.20)  # vs expected PM volume
            if rvol is not None and rvol < RVOL_MIN:
                skip["rvol"] += 1
                if DEBUG:
                    log.debug("SKIP %-8s  rvol %.2fx", sym, rvol)
                continue

        # ── SCORE ────────────────────────────────────────────────
        score = confluence_score(gap_pct * 100, rvol, above_vwap, pm_range_pct, etf_gap)
        quality = gap_quality_label(score, above_vwap, pm_range_pct)

        results.append(
            {
                "symbol": sym,
                "sector": sector,
                "prev_close": round(pc, 2),
                "pm_open": round(pm_open, 2) if pm_open else None,
                "pm_close": round(pm_close, 2),
                "pm_vwap": round(pm_vwap_val, 2) if pm_vwap_val else None,
                "gap_pct": round(gap_pct * 100, 2),
                "pm_range_pct": round(pm_range_pct, 2),
                "pm_volume": int(pm_vol),
                "rvol": round(rvol, 2) if rvol else "N/A",
                "etf_gap_pct": round(etf_gap, 2),
                "confluence": score,
                "gap_quality": quality,
                "rth_open": round(rth_open, 2),
                "rth_high": round(rth_high, 2),
                "rth_close": round(rth_close, 2),
                "open_gain_pct": round(open_gain * 100, 2),
            }
        )

    log.info(
        "  Results → pass=%d | no_snap=%d no_rth=%d gap=%d pmvol=%d vwap=%d rvol=%d etf=%d",
        len(results),
        skip["no_snap"],
        skip["no_rth"],
        skip["gap"],
        skip["pmvol"],
        skip["vwap"],
        skip["rvol"],
        skip["etf"],
    )

    if not results:
        log.warning(
            "\n  No stocks passed. Run with --debug to see rejections.\n"
            "  Also try: --no-rvol  --no-etf"
        )
        return {}

    df = pd.DataFrame(results)

    # Sector ranking: median confluence score of qualifying stocks
    # (better than open_gain alone — factors in quality of the move)
    sector_perf = (
        df.groupby("sector")
        .agg(
            median_gap_pct=("gap_pct", "median"),
            median_confluence=("confluence", "median"),
            qualifying_stocks=("symbol", "count"),
            etf_gap=("etf_gap_pct", "first"),
        )
        .sort_values("median_confluence", ascending=False)
    )

    top_sector_names = sector_perf.head(TOP_SECTORS).index.tolist()
    log.info("Top %d sectors: %s", TOP_SECTORS, top_sector_names)

    top: dict[str, pd.DataFrame] = {}
    for sector in top_sector_names:
        top[sector] = (
            df[df["sector"] == sector]
            .sort_values("confluence", ascending=False)  # rank by quality score
            .head(TOP_STOCKS_PER_SECT)
            .reset_index(drop=True)
        )

    # ── List B: ETF sector momentum stocks ────────────────────────
    log.info("Building List B — top %d ETF sector momentum stocks…", TOP_ETF_SECTORS)
    etf_sector_stocks = build_etf_sector_list(snaps, etf_gaps, pm_bars, rth_bars)
    for sector, sdf in etf_sector_stocks.items():
        log.info("  ETF sector %-28s  %d stocks", sector, len(sdf))

    return {
        "date": scan_date.isoformat(),
        "sector_ranks": sector_perf,
        "etf_gaps": etf_gaps,
        "top_sectors": top,  # List A — stock catalyst plays
        "etf_sector_stocks": etf_sector_stocks,  # List B — ETF momentum plays
        "all_passing": df,
    }


# ─── OUTPUT ──────────────────────────────────────────────────────────────────

STOCK_COLS = [
    "symbol",
    "prev_close",
    "pm_open",
    "pm_close",
    "pm_vwap",
    "gap_pct",
    "pm_range_pct",
    "pm_volume",
    "rvol",
    "confluence",
    "gap_quality",
    "rth_open",
    "rth_high",
    "rth_close",
    "open_gain_pct",
]
STOCK_HDRS = [
    "Symbol",
    "Prev $",
    "PM Open",
    "PM Last",
    "PM VWAP",
    "Gap%",
    "PM Rng%",
    "PM Vol",
    "RVol",
    "Score",
    "Quality",
    "RTH Open",
    "RTH High",
    "RTH Close",
    "Gain%",
]


def print_results(res: dict):
    if not res:
        return

    print(f"\n{'═' * 80}")
    print(f"  DATE   : {res['date']}")
    flt = (
        f"  FILTERS: Gap≥{PREMARKET_GAP_MIN * 100:.0f}%  PM_Vol≥{PREMARKET_VOL_MIN:,}"
        f"  Above_VWAP"
    )
    flt += f"  RVOL≥{RVOL_MIN}x" if USE_RVOL else "  RVOL:off"
    flt += f"  ETF≥0%" if USE_ETF_FILTER else "  ETF:off"
    print(flt)
    print(
        f"  NOTE   : Score=confluence 0–100. STRONG≥65, MODERATE≥40, WEAK<40 or below VWAP"
    )
    print(f"{'═' * 80}\n")

    # ETF snapshot
    print("── SECTOR ETF PREMARKET GAPS ──")
    etf_rows = sorted(res["etf_gaps"].items(), key=lambda x: -x[1])
    print(
        tabulate(
            [
                [s, SECTOR_ETF[s], f"{g:+.2f}%", "🟢" if g > 0 else "🔴"]
                for s, g in etf_rows
            ],
            headers=["Sector", "ETF", "PM Gap", ""],
            tablefmt="simple",
        )
    )

    print("\n── SECTOR LEADERBOARD (ranked by median confluence score) ──")
    print(res["sector_ranks"].to_string())

    for rank, (sector, df) in enumerate(res["top_sectors"].items(), 1):
        row = res["sector_ranks"].loc[sector]
        etf = SECTOR_ETF.get(sector, "?")
        etfg = res["etf_gaps"].get(sector, 0)
        print(f"\n{'─' * 80}")
        print(f"  #{rank}  {sector.upper()}  [{etf}: {etfg:+.2f}%]")
        print(
            f"       Median gap: {row['median_gap_pct']:+.2f}%  "
            f"Median score: {row['median_confluence']:.0f}  "
            f"Qualifying: {int(row['qualifying_stocks'])}"
        )
        print(f"{'─' * 80}")

        # Annotate gap quality reasoning inline
        for _, r in df.iterrows():
            print()
            _bullets = []
            if r["gap_pct"] >= 5:
                _bullets.append(f"large gap {r['gap_pct']:.1f}%")
            if isinstance(r["rvol"], float) and r["rvol"] >= 3:
                _bullets.append(f"high RVOL {r['rvol']:.1f}x")
            if r["pm_range_pct"] > 8:
                _bullets.append(
                    f"extended PM range {r['pm_range_pct']:.1f}% → fade risk"
                )
            if r["pm_vwap"] and r["pm_close"] < r["pm_vwap"]:
                _bullets.append("below PM VWAP → weak")
            if etfg < 0:
                _bullets.append("sector ETF red → headwind")
            # only print bullet if there's something notable
            if _bullets:
                print(f"  {r['symbol']}: {' | '.join(_bullets)}")

        print()
        rows = df[STOCK_COLS].values.tolist()
        print(
            tabulate(
                rows, headers=STOCK_HDRS, floatfmt=".2f", tablefmt="rounded_outline"
            )
        )

    # Summary gap quality counts
    all_df = res["all_passing"]
    strong = (all_df["gap_quality"].str.startswith("STRONG")).sum()
    moderate = (all_df["gap_quality"].str.startswith("MODERATE")).sum()
    weak = (
        ~all_df["gap_quality"].str.startswith("STRONG")
        & ~all_df["gap_quality"].str.startswith("MODERATE")
    ).sum()
    print(f"\n── GAP QUALITY SUMMARY (all {len(all_df)} passing stocks) ──")
    print(f"  🟢 STRONG: {strong}   🟡 MODERATE: {moderate}   ⚠ WEAK/EXTENDED: {weak}")

    # ── LIST B: ETF Sector Momentum Plays ─────────────────────────────────────
    etf_stocks = res.get("etf_sector_stocks", {})
    if etf_stocks:
        print(f"\n\n{'═' * 80}")
        print(
            f"  LIST B — ETF SECTOR MOMENTUM PLAYS  (top {TOP_ETF_SECTORS} ETF sectors)"
        )
        print(f"  Logic : stocks LEADING their sector ETF in PM (ranked by RS vs ETF)")
        print(
            f"  Use   : sector rotation / broad-market options  (relaxed filters: gap≥0.5%, vol≥50K)"
        )
        print(
            f"  RS vs ETF = stock gap% − sector ETF gap%  →  positive = outperforming the sector"
        )
        print(f"{'═' * 80}")

        ETF_COLS = [
            "symbol",
            "prev_close",
            "pm_open",
            "pm_close",
            "pm_vwap",
            "gap_pct",
            "etf_gap_pct",
            "rs_vs_etf",
            "pm_range_pct",
            "pm_volume",
            "rth_open",
            "rth_close",
            "open_gain_pct",
        ]
        ETF_HDRS = [
            "Symbol",
            "Prev $",
            "PM Open",
            "PM Last",
            "PM VWAP",
            "Stock Gap%",
            "ETF Gap%",
            "RS vs ETF",
            "PM Rng%",
            "PM Vol",
            "RTH Open",
            "RTH Close",
            "Gain%",
        ]

        for rank, (sector, df) in enumerate(etf_stocks.items(), 1):
            etf_sym = SECTOR_ETF.get(sector, "?")
            etf_gap = res["etf_gaps"].get(sector, 0)
            n_lead = (df["rs_vs_etf"] > 0).sum()
            print(f"\n{'─' * 80}")
            print(f"  #{rank}  {sector.upper()}  [{etf_sym}: {etf_gap:+.2f}%]")
            print(
                f"       {n_lead}/{len(df)} stocks outperforming the ETF (positive RS)"
            )
            print(f"{'─' * 80}")
            rows = df[ETF_COLS].values.tolist()
            print(
                tabulate(
                    rows, headers=ETF_HDRS, floatfmt=".2f", tablefmt="rounded_outline"
                )
            )

            # Quick read: highlight the strongest RS names
            leaders = df[df["rs_vs_etf"] > 1.0]["symbol"].tolist()
            laggards = df[df["rs_vs_etf"] < 0]["symbol"].tolist()
            if leaders:
                print(f"  🚀 Strong RS leaders (>+1% vs ETF): {', '.join(leaders)}")
            if laggards:
                print(f"  ⚠  Lagging ETF (RS<0, potential drag): {', '.join(laggards)}")
    else:
        print(f"\n  (No ETF sector momentum data — all top ETF sectors may be red)")


def save_csv(res: dict, out: str = "."):
    d = res["date"]
    for sector, df in res["top_sectors"].items():
        p = os.path.join(out, f"{d}_{sector.replace(' ', '_')}.csv")
        df.to_csv(p, index=False)
        log.info("Saved → %s", p)
    fp = os.path.join(out, f"{d}_all_passing.csv")
    res["all_passing"].to_csv(fp, index=False)
    log.info("Saved → %s", fp)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="S&P 500 Premarket Sector Scanner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", metavar="YYYY-MM-DD")
    g.add_argument("--start", metavar="YYYY-MM-DD")
    p.add_argument("--end", metavar="YYYY-MM-DD")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--no-rvol", dest="use_rvol", action="store_false")
    p.add_argument("--no-etf", dest="use_etf", action="store_false")
    p.add_argument("--feed", default=None, help="iex or sip")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(use_rvol=True, use_etf=True)
    return p.parse_args()


def main():
    global client
    global USE_RVOL, USE_ETF_FILTER, DATA_FEED, DEBUG
    args = parse_args()
    from swing_trading_scanner.config import require_credentials

    require_credentials()
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    USE_RVOL = args.use_rvol
    USE_ETF_FILTER = args.use_etf
    DEBUG = args.debug
    if args.feed:
        DATA_FEED = args.feed
    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.date:
        dates = [date.fromisoformat(args.date)]
    else:
        if not args.end:
            sys.exit("--end required with --start")
        s, e = date.fromisoformat(args.start), date.fromisoformat(args.end)
        dates = [
            s + timedelta(days=i)
            for i in range((e - s).days + 1)
            if (s + timedelta(days=i)).weekday() < 5
        ]

    all_results = []
    for d in dates:
        res = scan_day(d)
        if res:
            print_results(res)
            all_results.append(res)
            if args.csv:
                save_csv(res)

    if not all_results:
        print("\nNo results. Try --no-rvol --no-etf --debug")
    else:
        print(f"\n✓  Scanned {len(all_results)} trading day(s).")


if __name__ == "__main__":
    main()
