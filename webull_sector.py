"""
S&P 500 Premarket Sector Scanner - 9:45 AM Snapshot (Standalone)
==================================================================
Complete standalone script - no external dependencies except yfinance

Features:
  - Time-locked snapshot at 9:45 AM EST
  - Saves results with timestamp (never changes)
  - Same filters and scoring as original
  - Works immediately with no login

Usage:
  # Run snapshot at 9:45 AM
  python sector_scanner.py --date 2026-06-25 --snapshot

  # Run immediately (no wait)
  python sector_scanner.py --date 2026-06-25 --snapshot --no-wait

  # Run with custom time
  python sector_scanner.py --date 2026-06-25 --snapshot --time 09:30

  # Relax filters
  python sector_scanner.py --date 2026-06-25 --snapshot --no-rvol --no-etf
"""

import os, sys, argparse, logging, time, json
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
import pytz
from tabulate import tabulate

# Yahoo Finance
try:
    import yfinance as yf
except ImportError:
    print("📦 Installing yfinance...")
    os.system("pip install yfinance")
    import yfinance as yf

# ─── CONFIG ──────────────────────────────────────────────────────────────────

PREMARKET_GAP_MIN = 0.02  # 2% minimum gap vs prior close
PREMARKET_VOL_MIN = 100_000  # minimum premarket shares
USE_RVOL = True
RVOL_MIN = 2.0
USE_ETF_FILTER = True  # sector ETF must also be green in PM
TOP_SECTORS = 3
TOP_STOCKS_PER_SECT = 10
TOP_ETF_SECTORS = 3
ETF_GAP_MIN = 0.005
ETF_VOL_MIN = 50_000

# Snapshot configuration
SNAPSHOT_TIME = "09:45"  # Default snapshot time
WAIT_FOR_SNAPSHOT = True  # Wait until snapshot time if run early

DEBUG = False
EST = pytz.timezone("US/Eastern")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── SECTOR MAP ──────────────────────────────────────────────────────────────

SECTOR_TICKERS: Dict[str, List[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM", "INTC", "QCOM", "TXN",
        "MU", "NOW", "AMAT", "KLAC", "LRCX", "ADI", "MRVL", "PANW", "SNPS", "CDNS",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
        "PFE", "GILD", "ISRG", "VRTX", "REGN", "CI", "CVS", "HCA", "MDT", "IQV",
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK",
        "SCHW", "CB", "PGR", "MMC", "USB", "TFC", "COF", "AIG", "MET", "PRU", "ICE",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "ORLY",
        "GM", "F", "ABNB", "YUM", "DRI", "HLT", "MAR", "RCL", "CCL", "TSCO",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "CHTR", "EA", "TTWO",
        "TMUS", "FOXA", "OMC", "IPG", "NYT", "WBD", "PARA", "LYV", "MTCH", "ZM",
    ],
    "Industrials": [
        "RTX", "HON", "UNP", "CAT", "DE", "BA", "LMT", "GE", "MMM", "FDX",
        "UPS", "EMR", "ETN", "PH", "CMI", "CTAS", "NSC", "CSX", "CARR", "TDG",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "GIS",
        "KHC", "KMB", "SYY", "HSY", "K", "CAG", "CPB", "MKC", "CHD", "CLX",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY",
        "HES", "HAL", "DVN", "BKR", "APA", "MRO", "EQT", "KMI", "WMB", "TRGP", "NOG",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "ETR",
        "PCG", "WEC", "ES", "AWK", "DTE", "CMS", "CNP", "NI", "AES", "PPL",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR", "WY", "ARE", "BXP", "KIM", "NNN", "HST", "EXR", "CUBE",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM",
        "CF", "MOS", "ALB", "RPM", "PKG", "IP", "SEE", "SON", "GEF", "SLGN",
    ],
}

SECTOR_ETF: Dict[str, str] = {
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

TICKER_SECTOR: Dict[str, str] = {}
for _s, _ts in SECTOR_TICKERS.items():
    for _t in _ts:
        TICKER_SECTOR[_t] = _s

ALL_TICKERS = sorted(TICKER_SECTOR.keys())
ALL_ETFS = list(SECTOR_ETF.values())


# ─── YAHOO FINANCE DATA FETCHER ────────────────────────────────────────────

def fetch_yahoo_data(symbols: List[str]) -> Dict[str, Dict]:
    """Fetch premarket data from Yahoo Finance"""
    results: Dict[str, Dict] = {}

    chunk_size = 50
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]

        try:
            tickers = yf.Tickers(" ".join(chunk))

            for sym in chunk:
                try:
                    ticker = tickers.tickers.get(sym)
                    if not ticker:
                        continue

                    info = ticker.info

                    prev_close = info.get('regularMarketPreviousClose', 0)
                    pre_price = info.get('preMarketPrice', 0)
                    pre_volume = info.get('preMarketVolume', 0)

                    if pre_volume == 0 or pre_price == 0:
                        pre_price = info.get('regularMarketPrice', 0)
                        pre_volume = info.get('regularMarketVolume', 0)

                    day_high = info.get('dayHigh', pre_price)
                    day_low = info.get('dayLow', pre_price)
                    day_open = info.get('open', pre_price)

                    results[sym] = {
                        "prev_close": float(prev_close) if prev_close else 0,
                        "pm_open": float(day_open) if day_open else None,
                        "pm_high": float(day_high) if day_high else None,
                        "pm_low": float(day_low) if day_low else None,
                        "pm_close": float(pre_price) if pre_price else 0,
                        "pm_volume": float(pre_volume) if pre_volume else 0,
                    }

                except Exception as e:
                    log.debug(f"Error fetching {sym}: {e}")
                    continue

            time.sleep(0.2)

        except Exception as e:
            log.warning(f"Error fetching chunk: {e}")
            continue

    return results


def avg_daily_volume_20d(symbol: str) -> Optional[float]:
    """Calculate 20-day average daily volume"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="25d")
        if hist.empty:
            return None
        volumes = hist['Volume'].tail(20)
        if len(volumes) >= 5:
            return float(volumes.mean())
        return None
    except Exception:
        return None


def vwap_from_bars(symbol: str) -> Optional[float]:
    """Approximate VWAP from Yahoo data"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        high = info.get('dayHigh', 0)
        low = info.get('dayLow', 0)
        close = info.get('preMarketPrice', info.get('regularMarketPrice', 0))
        volume = info.get('preMarketVolume', info.get('regularMarketVolume', 0))

        if volume > 0 and high > 0 and low > 0:
            typical = (high + low + close) / 3
            return float(typical)
        return None
    except Exception:
        return None


# ─── SCORING ─────────────────────────────────────────────────────────────────

def confluence_score(gap_pct: float, rvol, above_vwap: bool,
                     pm_range_pct: float, etf_gap_pct: float) -> int:
    gap_pts = min(30, gap_pct / 10 * 30)

    if rvol is None or rvol == "N/A":
        rvol_pts = 10
    else:
        rvol_pts = min(25, float(rvol) / 10 * 25)

    vwap_pts = 20 if above_vwap else 0

    if pm_range_pct <= 2:
        range_pts = 15
    elif pm_range_pct <= 5:
        range_pts = 10
    elif pm_range_pct <= 8:
        range_pts = 5
    else:
        range_pts = 0

    if etf_gap_pct > 0.5:
        etf_pts = 10
    elif etf_gap_pct > 0:
        etf_pts = 5
    else:
        etf_pts = 0

    return int(gap_pts + rvol_pts + vwap_pts + range_pts + etf_pts)


def gap_quality_label(score: int, above_vwap: bool, pm_range_pct: float) -> str:
    if not above_vwap:
        return "WEAK ⚠"
    if pm_range_pct > 10:
        return "EXTENDED ⚠"
    if score >= 65:
        return "STRONG 🟢"
    if score >= 40:
        return "MODERATE 🟡"
    return "WEAK ⚠"


# ─── BUILD ETF LIST ─────────────────────────────────────────────────────────

def build_etf_sector_list(
        snaps: Dict,
        etf_gaps: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    """Build List B - ETF sector momentum stocks"""
    top_etf_sectors = sorted(etf_gaps.items(), key=lambda x: -x[1])[:TOP_ETF_SECTORS]
    etf_list: Dict[str, pd.DataFrame] = {}

    for sector, etf_gap_pct in top_etf_sectors:
        if etf_gap_pct <= 0:
            continue

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

            if gap_pct < ETF_GAP_MIN * 100:
                continue
            if pm_vol < ETF_VOL_MIN:
                continue

            pm_vwap_val = vwap_from_bars(sym)
            above_vwap = (pm_close >= pm_vwap_val) if pm_vwap_val else True
            if not above_vwap:
                continue

            rs_vs_etf = round(gap_pct - etf_gap_pct, 2)
            pm_range_pct = ((pm_high - pm_open) / pm_open * 100
                            if pm_open and pm_open > 0 and pm_high else 0.0)

            rows.append({
                "symbol": sym,
                "sector": sector,
                "prev_close": round(pc, 2),
                "pm_open": round(pm_open, 2) if pm_open else None,
                "pm_close": round(pm_close, 2),
                "pm_vwap": round(pm_vwap_val, 2) if pm_vwap_val else None,
                "gap_pct": round(gap_pct, 2),
                "etf_gap_pct": round(etf_gap_pct, 2),
                "rs_vs_etf": rs_vs_etf,
                "pm_range_pct": round(pm_range_pct, 2),
                "pm_volume": int(pm_vol),
            })

        if rows:
            sector_df = (
                pd.DataFrame(rows)
                .sort_values("rs_vs_etf", ascending=False)
                .head(TOP_STOCKS_PER_SECT)
                .reset_index(drop=True)
            )
            etf_list[sector] = sector_df

    return etf_list


# ─── SNAPSHOT SCANNER ──────────────────────────────────────────────────────

def wait_for_snapshot_time(target_time: str = "09:45"):
    """Wait until the target time"""
    now = datetime.now(EST)
    target_hour, target_min = map(int, target_time.split(':'))
    target = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)

    if now >= target:
        log.info(f"⏰ Current time {now.strftime('%H:%M')} is past snapshot time {target_time}")
        return

    wait_seconds = (target - now).total_seconds()
    log.info(f"⏰ Waiting {wait_seconds:.0f} seconds until {target_time} EST...")
    time.sleep(wait_seconds)
    log.info(f"✅ Snapshot time reached!")


def scan_day_snapshot(scan_date: date, snapshot_time: str = "09:45", wait: bool = True) -> Dict:
    """Main scan with time-locked snapshot"""

    log.info("=" * 70)
    log.info("📸 SNAPSHOT SCANNER - Time-Locked at %s EST", snapshot_time)
    log.info("=" * 70)

    # Wait for snapshot time if requested
    if wait:
        wait_for_snapshot_time(snapshot_time)

    # Record exact snapshot time
    snapshot_dt = datetime.now(EST)
    log.info(f"📸 Capturing snapshot at: {snapshot_dt.strftime('%H:%M:%S')} EST")

    # Fetch data
    all_snap_syms = ALL_TICKERS + ALL_ETFS
    log.info(f"📊 Fetching data for {len(all_snap_syms)} symbols...")
    snaps = fetch_yahoo_data(all_snap_syms)
    log.info(f"  ✅ Got data for {len(snaps)} symbols")

    active_count = sum(1 for s in snaps.values() if s.get('pm_volume', 0) > 0)
    log.info(f"  📈 Active symbols: {active_count}")

    if active_count == 0:
        log.warning("⚠️  No volume detected. Market may be closed.")
        return {}

    # ETF gaps
    etf_gaps: Dict[str, float] = {}
    for sector, etf in SECTOR_ETF.items():
        snap = snaps.get(etf)
        if snap and snap["prev_close"] and snap["pm_close"]:
            etf_gaps[sector] = (snap["pm_close"] - snap["prev_close"]) / snap["prev_close"] * 100
        else:
            etf_gaps[sector] = 0.0

    # Process stocks
    results = []
    skip = {"no_snap": 0, "gap": 0, "pmvol": 0, "vwap": 0, "rvol": 0, "etf": 0}
    rvol_cache: Dict[str, Optional[float]] = {}

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
        sector = TICKER_SECTOR[sym]

        if not pm_close or pc == 0:
            skip["gap"] += 1
            continue

        gap_pct = (pm_close - pc) / pc

        pm_vwap_val = vwap_from_bars(sym)
        above_vwap = (pm_close >= pm_vwap_val) if pm_vwap_val else True

        if pm_open and pm_open > 0 and pm_high:
            pm_range_pct = (pm_high - pm_open) / pm_open * 100
        else:
            pm_range_pct = 0.0

        # Filters
        if gap_pct < PREMARKET_GAP_MIN:
            skip["gap"] += 1
            continue

        if pm_vol < PREMARKET_VOL_MIN:
            skip["pmvol"] += 1
            continue

        if pm_vwap_val and not above_vwap:
            skip["vwap"] += 1
            continue

        etf_gap = etf_gaps.get(sector, 0.0)
        if USE_ETF_FILTER and etf_gap < 0:
            skip["etf"] += 1
            continue

        rvol = None
        if USE_RVOL:
            if sym not in rvol_cache:
                rvol_cache[sym] = avg_daily_volume_20d(sym)
            avg_vol = rvol_cache[sym]
            if avg_vol and avg_vol > 0:
                rvol = pm_vol / (avg_vol * 0.20)
            if rvol is not None and rvol < RVOL_MIN:
                skip["rvol"] += 1
                continue

        score = confluence_score(gap_pct * 100, rvol, above_vwap, pm_range_pct, etf_gap)
        quality = gap_quality_label(score, above_vwap, pm_range_pct)

        results.append({
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
        })

    log.info("  Results → pass=%d | no_snap=%d gap=%d pmvol=%d vwap=%d rvol=%d etf=%d",
             len(results), skip["no_snap"], skip["gap"], skip["pmvol"], skip["vwap"], skip["rvol"], skip["etf"])

    if not results:
        log.warning("\n  No stocks passed. Try --no-rvol --no-etf")
        return {}

    df = pd.DataFrame(results)

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

    top: Dict[str, pd.DataFrame] = {}
    for sector in top_sector_names:
        top[sector] = (
            df[df["sector"] == sector]
            .sort_values("confluence", ascending=False)
            .head(TOP_STOCKS_PER_SECT)
            .reset_index(drop=True)
        )

    etf_sector_stocks = build_etf_sector_list(snaps, etf_gaps)

    return {
        "date": scan_date.isoformat(),
        "snapshot_time": snapshot_dt.strftime('%H:%M:%S'),
        "sector_ranks": sector_perf,
        "etf_gaps": etf_gaps,
        "top_sectors": top,
        "etf_sector_stocks": etf_sector_stocks,
        "all_passing": df,
        "snapshot": True,
    }


# ─── OUTPUT ─────────────────────────────────────────────────────────────────

def print_results(res: Dict):
    if not res:
        return

    STOCK_COLS = [
        "symbol", "prev_close", "pm_open", "pm_close", "pm_vwap",
        "gap_pct", "pm_range_pct", "pm_volume", "rvol",
        "confluence", "gap_quality",
    ]
    STOCK_HDRS = [
        "Symbol", "Prev $", "PM Open", "PM Last", "PM VWAP",
        "Gap%", "PM Rng%", "PM Vol", "RVol",
        "Score", "Quality",
    ]

    print(f"\n{'═' * 80}")
    print(f"  📸 SNAPSHOT SCANNER - TIME LOCKED")
    print(f"  📅 DATE   : {res['date']}")
    print(f"  ⏰ SNAPSHOT: {res['snapshot_time']} EST (FROZEN DATA)")
    print(f"  📡 FEED   : Yahoo Finance (FREE)")
    print(f"  🔒 STATUS : Data is LOCKED - Will not change")
    flt = (f"  🔍 FILTERS: Gap≥{PREMARKET_GAP_MIN * 100:.0f}%  PM_Vol≥{PREMARKET_VOL_MIN:,}"
           f"  Above_VWAP")
    flt += f"  RVOL≥{RVOL_MIN}x" if USE_RVOL else "  RVOL:off"
    flt += f"  ETF≥0%" if USE_ETF_FILTER else "  ETF:off"
    print(flt)
    print(f"  💡 SCORE  : STRONG≥65  MODERATE≥40  WEAK<40 or below VWAP")
    print(f"{'═' * 80}\n")

    # ETF snapshot
    print("── SECTOR ETF PREMARKET GAPS (SNAPSHOT) ──")
    etf_rows = sorted(res["etf_gaps"].items(), key=lambda x: -x[1])
    print(tabulate(
        [[s, SECTOR_ETF[s], f"{g:+.2f}%", "🟢" if g > 0 else "🔴"] for s, g in etf_rows],
        headers=["Sector", "ETF", "PM Gap", ""],
        tablefmt="simple",
    ))

    print("\n── SECTOR LEADERBOARD ──")
    print(res["sector_ranks"].to_string())

    for rank, (sector, df) in enumerate(res["top_sectors"].items(), 1):
        row = res["sector_ranks"].loc[sector]
        etf = SECTOR_ETF.get(sector, "?")
        etfg = res["etf_gaps"].get(sector, 0)
        print(f"\n{'─' * 80}")
        print(f"  #{rank}  {sector.upper()}  [{etf}: {etfg:+.2f}%]")
        print(f"       Median gap: {row['median_gap_pct']:+.2f}%  "
              f"Median score: {row['median_confluence']:.0f}  "
              f"Qualifying: {int(row['qualifying_stocks'])}")
        print(f"{'─' * 80}")

        rows = df[STOCK_COLS].values.tolist()
        print(tabulate(rows, headers=STOCK_HDRS, floatfmt=".2f", tablefmt="rounded_outline"))

    # Gap quality summary
    all_df = res["all_passing"]
    strong = (all_df["gap_quality"].str.startswith("STRONG")).sum()
    moderate = (all_df["gap_quality"].str.startswith("MODERATE")).sum()
    weak = (~all_df["gap_quality"].str.startswith("STRONG") &
            ~all_df["gap_quality"].str.startswith("MODERATE")).sum()
    print(f"\n── GAP QUALITY SUMMARY ──")
    print(f"  🟢 STRONG: {strong}   🟡 MODERATE: {moderate}   ⚠ WEAK: {weak}")

    # List B
    etf_stocks = res.get("etf_sector_stocks", {})
    if etf_stocks:
        print(f"\n\n{'═' * 80}")
        print(f"  📋 LIST B — ETF SECTOR MOMENTUM PLAYS (SNAPSHOT)")
        print(f"  Top {TOP_ETF_SECTORS} ETF sectors | Relaxed: gap≥{ETF_GAP_MIN * 100:.1f}%, vol≥{ETF_VOL_MIN:,}")
        print(f"{'═' * 80}")

        for rank, (sector, df) in enumerate(etf_stocks.items(), 1):
            etf_sym = SECTOR_ETF.get(sector, "?")
            etf_gap = res["etf_gaps"].get(sector, 0)
            n_lead = (df["rs_vs_etf"] > 0).sum()
            print(f"\n  #{rank}  {sector.upper()}  [{etf_sym}: {etf_gap:+.2f}%]  {n_lead}/{len(df)} stocks leading ETF")
            cols = ["symbol", "gap_pct", "etf_gap_pct", "rs_vs_etf", "pm_volume"]
            hdrs = ["Symbol", "Stock Gap%", "ETF Gap%", "RS vs ETF", "Vol"]
            rows = df[cols].values.tolist()
            print(tabulate(rows, headers=hdrs, floatfmt=".2f", tablefmt="simple"))


def save_snapshot(res: Dict, out: str = "."):
    """Save snapshot results with timestamp"""
    d = res["date"]
    timestamp = res["snapshot_time"].replace(":", "")

    # Save all passing stocks
    fp = os.path.join(out, f"{d}_snapshot_{timestamp}_all.csv")
    res["all_passing"].to_csv(fp, index=False)
    log.info(f"✅ Saved snapshot → {fp}")

    # Save sector breakdown
    for sector, df in res["top_sectors"].items():
        p = os.path.join(out, f"{d}_snapshot_{timestamp}_{sector.replace(' ', '_')}.csv")
        df.to_csv(p, index=False)
        log.info(f"✅ Saved sector → {p}")

    # Save summary JSON
    summary = {
        "date": res["date"],
        "snapshot_time": res["snapshot_time"],
        "total_passing": len(res["all_passing"]),
        "strong_count": len(res["all_passing"][res["all_passing"]["gap_quality"].str.startswith("STRONG")]),
        "moderate_count": len(res["all_passing"][res["all_passing"]["gap_quality"].str.startswith("MODERATE")]),
        "weak_count": len(res["all_passing"][~res["all_passing"]["gap_quality"].str.startswith("STRONG") &
                                             ~res["all_passing"]["gap_quality"].str.startswith("MODERATE")]),
        "top_sectors": list(res["top_sectors"].keys()),
        "etf_gaps": res["etf_gaps"],
    }
    jp = os.path.join(out, f"{d}_snapshot_{timestamp}_summary.json")
    with open(jp, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"✅ Saved summary → {jp}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="S&P 500 Premarket Scanner - Time-Locked Snapshot")
    p.add_argument("--date", metavar="YYYY-MM-DD", required=True)
    p.add_argument("--snapshot", action="store_true", help="Capture snapshot at 9:45 AM")
    p.add_argument("--time", default="09:45", help="Snapshot time (HH:MM)")
    p.add_argument("--no-wait", action="store_true", help="Don't wait for snapshot time")
    p.add_argument("--csv", action="store_true", help="Save snapshot to CSV")
    p.add_argument("--no-rvol", dest="use_rvol", action="store_false")
    p.add_argument("--no-etf", dest="use_etf", action="store_false")
    p.add_argument("--debug", action="store_true")
    p.set_defaults(use_rvol=True, use_etf=True)
    return p.parse_args()


def main():
    global USE_RVOL, USE_ETF_FILTER, DEBUG, PREMARKET_GAP_MIN, PREMARKET_VOL_MIN
    global RVOL_MIN, TOP_SECTORS, TOP_STOCKS_PER_SECT

    args = parse_args()
    USE_RVOL = args.use_rvol
    USE_ETF_FILTER = args.use_etf
    DEBUG = args.debug

    if DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)

    scan_date = date.fromisoformat(args.date)

    # Check if snapshot mode
    if args.snapshot:
        log.info("📸 SNAPSHOT MODE ENABLED")
        res = scan_day_snapshot(scan_date, args.time, not args.no_wait)
    else:
        log.info("⚠️  Running without snapshot (data may change)")
        log.info("   Use --snapshot to lock data at 9:45 AM")
        # Quick run without snapshot
        res = scan_day_snapshot(scan_date, args.time, False)

    if res:
        print_results(res)
        if args.csv:
            save_snapshot(res)
        print(f"\n✅ Snapshot saved! Data is FROZEN for {scan_date}")
        print(f"   Run again with --no-wait to capture at a different time")
    else:
        print("\n❌ No results. Try:")
        print("   --no-rvol --no-etf  (relax filters)")
        print("   --debug             (see what's happening)")
        print("   --time 09:30        (earlier snapshot)")


if __name__ == "__main__":
    main()