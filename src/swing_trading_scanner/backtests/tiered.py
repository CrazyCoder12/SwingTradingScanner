"""
EMA Backtest Engine — Alpaca API  (R-Based Position Sizing + Tiered Exits)
===========================================================================
Extends swing_trading.py with two intraday backtest strategies.

SIGNAL SOURCES (from swing_trading.py):
  Scanner 1 — EMA 5/12 daily crossover     (short-term momentum)
  Scanner 2 — EMA 12 crosses EMA 34/50     (medium-term trend shift)

ENTRY LOGIC (next trading day after signal fires):
  Pre-market check (4:00 AM – 9:29 AM bars):
    → If EMA-5 > EMA-12 at the last pre-market bar → primed for bullish entry at open
    → If EMA-5 < EMA-12 at the last pre-market bar → primed for bearish entry at open

  Market open (9:30 AM onward):
    BULLISH entry:
      Option A: EMA-5 > EMA-12 already at 9:30 open bar  → enter immediately
      Option B: EMA-5 crosses ABOVE EMA-12 AND 10m candle closes ABOVE EMA-5
    BEARISH entry:
      Option A: EMA-5 < EMA-12 already at 9:30 open bar  → enter immediately
      Option B: EMA-5 crosses BELOW EMA-12 AND 10m candle closes BELOW EMA-5

POSITION SIZING (R-based, $100 risk per trade):
  1R distance = |entry_price - EMA-12 at entry bar|
  Shares      = $100 / 1R_distance   (fractional shares)
  Capital     = shares × entry_price

TIERED EXIT PLAN:
  ┌──────┬──────────────────────────────────────┬────────────────────────────┐
  │ Level│ Price target (BULLISH)               │ Action                     │
  ├──────┼──────────────────────────────────────┼────────────────────────────┤
  │  SL  │ entry - 1R  (EMA-12 cross/close)     │ Exit 100% remaining        │
  │  2R  │ entry + 2R                           │ Sell 50%, stop → breakeven │
  │  3R  │ entry + 3R                           │ Sell 50% of remaining 50%  │
  │      │                                      │ (=25% of original)         │
  │      │                                      │ Trailing stop set at 2R    │
  │  4R  │ entry + 4R                           │ Sell all remaining (25%)   │
  └──────┴──────────────────────────────────────┴────────────────────────────┘
  (BEARISH is the mirror — short side prices and P&L inverted accordingly)

  After 3R hit: trailing stop moves with high watermark (never retreats).
  If trailing stop is hit before 4R → exit all remaining at trail price.

STRATEGIES:
  Strategy 1 — Hard stop on 10m EMA cross/close, targets on 10m bars
  Strategy 2 — Hard stop on 1h EMA cross/close, targets still checked on 10m bars

Data source: Alpaca Markets v2 (APCA-API-KEY-ID / APCA-API-SECRET-KEY from .env)

Usage:
  python3 backtest_alpaca.py --start-date 2026-03-01
  python3 backtest_alpaca.py --start-date 2026-03-01 --scanner 1
  python3 backtest_alpaca.py --start-date 2026-03-01 --symbol NVDA
  python3 backtest_alpaca.py --signals-csv my_signals.csv
  python3 backtest_alpaca.py --start-date 2026-01-01 --strategy 2 --no-bearish
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime, timedelta, timezone, date as datelib
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from swing_trading_scanner.config import load_environment

warnings.filterwarnings("ignore")
load_environment()

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

RISK_PER_TRADE = 100.0  # $ risk per trade (1R in dollar terms)
STARTING_CAPITAL = 100_000.0
MAX_HOLD_DAYS = 20  # hard bail-out after N calendar days
WARMUP_DAYS = 120  # daily bars before start_date for EMA warmup
ET = ZoneInfo("America/New_York")

MARKET_OPEN = "09:30"
PREMARKET_START = "04:00"

# Profit-booking fractions at each R level
# After 2R: sell 50% of full position  → 50% remains
# After 3R: sell 50% of remaining      → 25% of original remains
# After 4R: sell all remaining         → 0% remains
TARGET_2R_SELL_FRAC = 0.50  # fraction of TOTAL shares to sell at 2R
TARGET_3R_SELL_FRAC = 0.50  # fraction of REMAINING shares to sell at 3R
# 4R → sell everything left

# Trailing stop is set at the 2R price after 3R is hit, then trails high
TRAIL_LOCK_R = 2.0  # trailing stop floor = entry + 2R (long) once 3R hit
TRAIL_DISTANCE_R = 1.0  # trail 1R below the high watermark


# ─────────────────────────────────────────────────────────────────────────────
#  Alpaca REST helper
# ─────────────────────────────────────────────────────────────────────────────

_API_KEY = os.getenv("ALPACA_API_KEY", "")
_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
_BASE = "https://data.alpaca.markets"
_HEADERS = {
    "APCA-API-KEY-ID": _API_KEY,
    "APCA-API-SECRET-KEY": _SECRET_KEY,
}


def _get(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if r.status_code == 429:
                time.sleep(2**attempt)
                continue
            return {}
        except Exception:
            time.sleep(1)
    return {}


def fetch_bars(
    symbol: str, timeframe: str, start: datetime, end: datetime, feed: str = "sip"
) -> pd.DataFrame:
    """
    Generic Alpaca v2 bar fetcher.
    timeframe: '1Day', '10Min', '1Hour'
    Returns DataFrame indexed by UTC-aware datetime.
    """
    url = f"{_BASE}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feed": feed,
        "adjustment": "split",
        "limit": 10000,
    }
    all_bars = []
    next_token = None

    while True:
        if next_token:
            params["page_token"] = next_token
        data = _get(url, params)
        bars = data.get("bars", [])
        all_bars.extend(bars)
        next_token = data.get("next_page_token")
        if not next_token:
            break

    if not all_bars:
        if feed == "sip":
            return fetch_bars(symbol, timeframe, start, end, feed="iex")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = (
        df.rename(
            columns={
                "t": "ts",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        .set_index("ts")
        .sort_index()
    )
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep="first")]
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  EMA helpers
# ─────────────────────────────────────────────────────────────────────────────


def add_emas_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for span in (5, 12, 34, 50):
        df[f"ema{span}"] = df["close"].ewm(span=span, adjust=False).mean()
    return df


def add_emas_intraday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Daily scanner (mirrors swing_trading.py)
# ─────────────────────────────────────────────────────────────────────────────


def get_signals_s1(df: pd.DataFrame, start_dt: datetime) -> list:
    signals = []
    sim_mask = df.index >= pd.Timestamp(start_dt).tz_localize(None)
    if not sim_mask.any():
        return signals
    start_pos = df.index.get_loc(df[sim_mask].index[0])
    for i in range(max(1, start_pos), len(df)):
        prev, cur = df.iloc[i - 1], df.iloc[i]
        close = float(cur["close"])
        e5, e12 = float(cur["ema5"]), float(cur["ema12"])
        bull = prev["ema5"] <= prev["ema12"] and cur["ema5"] > cur["ema12"]
        bear = prev["ema5"] >= prev["ema12"] and cur["ema5"] < cur["ema12"]
        if bull and close > e5:
            signals.append(
                {
                    "signal_date": df.index[i].strftime("%Y-%m-%d"),
                    "direction": "BULLISH",
                    "scanner": "S1",
                }
            )
        elif bear and close < e5:
            signals.append(
                {
                    "signal_date": df.index[i].strftime("%Y-%m-%d"),
                    "direction": "BEARISH",
                    "scanner": "S1",
                }
            )
    return signals


def get_signals_s2(df: pd.DataFrame, start_dt: datetime) -> list:
    signals = []
    sim_mask = df.index >= pd.Timestamp(start_dt).tz_localize(None)
    if not sim_mask.any():
        return signals
    start_pos = df.index.get_loc(df[sim_mask].index[0])
    for i in range(max(1, start_pos), len(df)):
        prev, cur = df.iloc[i - 1], df.iloc[i]
        e5, e12 = float(cur["ema5"]), float(cur["ema12"])
        e34, e50 = float(cur["ema34"]), float(cur["ema50"])
        bull = prev["ema12"] <= prev["ema34"] and cur["ema12"] > cur["ema34"]
        bear = prev["ema12"] >= prev["ema34"] and cur["ema12"] < cur["ema34"]
        if bull and e5 > e50:
            signals.append(
                {
                    "signal_date": df.index[i].strftime("%Y-%m-%d"),
                    "direction": "BULLISH",
                    "scanner": "S2",
                }
            )
        elif bear and e5 < e50:
            signals.append(
                {
                    "signal_date": df.index[i].strftime("%Y-%m-%d"),
                    "direction": "BEARISH",
                    "scanner": "S2",
                }
            )
    return signals


def run_daily_scanner(
    symbols: list, start_date_str: str, run_s1: bool = True, run_s2: bool = True
) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    warmup_dt = start_dt - timedelta(days=WARMUP_DAYS)
    now_utc = datetime.now(timezone.utc)

    print(f"\n{'=' * 70}")
    print(f"  STEP 1 — DAILY SIGNAL SCANNER  [{start_date_str} → today]")
    print(f"{'=' * 70}")

    all_signals = []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] {sym:<8}", end="  ", flush=True)
        df = fetch_bars(sym, "1Day", warmup_dt.replace(tzinfo=timezone.utc), now_utc)
        if df.empty or len(df) < 60:
            print("no data")
            continue

        df.index = df.index.tz_localize(None)
        df = add_emas_daily(df)

        parts = []
        if run_s1:
            sigs = get_signals_s1(df, start_dt)
            for s in sigs:
                s["symbol"] = sym
                all_signals.append(s)
            b = sum(1 for s in sigs if s["direction"] == "BULLISH")
            r = sum(1 for s in sigs if s["direction"] == "BEARISH")
            if sigs:
                parts.append(f"S1: 🟢{b} 🔴{r}")

        if run_s2:
            sigs = get_signals_s2(df, start_dt)
            for s in sigs:
                s["symbol"] = sym
                all_signals.append(s)
            b = sum(1 for s in sigs if s["direction"] == "BULLISH")
            r = sum(1 for s in sigs if s["direction"] == "BEARISH")
            if sigs:
                parts.append(f"S2: 🟢{b} 🔴{r}")

        print("  |  ".join(parts) if parts else "no signals")

    if not all_signals:
        return pd.DataFrame()
    df_sig = (
        pd.DataFrame(all_signals)
        .sort_values(["signal_date", "symbol"])
        .reset_index(drop=True)
    )
    print(
        f"\n  {len(df_sig)} total signal(s) across {df_sig['symbol'].nunique()} symbols.\n"
    )
    return df_sig


# ─────────────────────────────────────────────────────────────────────────────
#  Intraday data cache
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict = {}


def get_intraday(
    symbol: str, timeframe: str, fetch_start: datetime, fetch_end: datetime
) -> pd.DataFrame:
    key = (symbol, timeframe, fetch_start.date(), fetch_end.date())
    if key not in _cache:
        df = fetch_bars(symbol, timeframe, fetch_start, fetch_end)
        if not df.empty:
            df.index = df.index.tz_convert(ET)
        _cache[key] = df
    return _cache[key]


# ─────────────────────────────────────────────────────────────────────────────
#  Calendar helpers
# ─────────────────────────────────────────────────────────────────────────────


def next_trading_day(date_str: str) -> datelib:
    d = datetime.strptime(date_str, "%Y-%m-%d").date() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def bars_for_date(
    df: pd.DataFrame,
    target_date: datelib,
    after_time: str = None,
    before_time: str = None,
) -> pd.DataFrame:
    import datetime as _dt

    mask = df.index.date == target_date
    sub = df[mask]
    if sub.empty:
        return sub
    if after_time:
        h, m = map(int, after_time.split(":"))
        sub = sub[sub.index.time >= _dt.time(h, m)]
    if before_time:
        h, m = map(int, before_time.split(":"))
        sub = sub[sub.index.time < _dt.time(h, m)]
    return sub


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-market EMA check
# ─────────────────────────────────────────────────────────────────────────────


def premarket_ema_state(
    bars_10m: pd.DataFrame, entry_date: datelib, direction: str
) -> str:
    pm_bars = bars_for_date(
        bars_10m, entry_date, after_time=PREMARKET_START, before_time=MARKET_OPEN
    )
    if pm_bars.empty:
        return "UNKNOWN"
    last = pm_bars.iloc[-1]
    e5, e12 = float(last["ema5"]), float(last["ema12"])
    if direction == "BULLISH":
        return "ALIGNED" if e5 > e12 else "AGAINST"
    return "ALIGNED" if e5 < e12 else "AGAINST"


# ─────────────────────────────────────────────────────────────────────────────
#  10m entry finder — returns (entry_ts, entry_price, ema12_at_entry)
# ─────────────────────────────────────────────────────────────────────────────


def find_entry(
    bars_10m: pd.DataFrame, entry_date: datelib, direction: str, pm_state: str
):
    """
    Returns (entry_ts, entry_price, ema12_at_entry) or (None, None, None).
    ema12_at_entry is used to calculate the 1R distance for position sizing.
    """
    mkt_bars = bars_for_date(bars_10m, entry_date, after_time=MARKET_OPEN)
    if mkt_bars.empty:
        return None, None, None

    prev_e5 = prev_e12 = None

    for i, (ts, row) in enumerate(mkt_bars.iterrows()):
        close = float(row["close"])
        e5 = float(row["ema5"])
        e12 = float(row["ema12"])

        if direction == "BULLISH":
            if i == 0 and pm_state == "ALIGNED" and e5 > e12:
                return ts, close, e12
            if prev_e5 is not None:
                if prev_e5 <= prev_e12 and e5 > e12 and close > e5:
                    return ts, close, e12
            if i == 0 and e5 > e12 and pm_state != "AGAINST":
                return ts, close, e12
        else:
            if i == 0 and pm_state == "ALIGNED" and e5 < e12:
                return ts, close, e12
            if prev_e5 is not None:
                if prev_e5 >= prev_e12 and e5 < e12 and close < e5:
                    return ts, close, e12
            if i == 0 and e5 < e12 and pm_state != "AGAINST":
                return ts, close, e12

        prev_e5, prev_e12 = e5, e12

    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  R-level calculator
# ─────────────────────────────────────────────────────────────────────────────


class RLevels:
    """
    Precomputes all R-level prices and position details for a trade.

    BULLISH:
      stop  = entry - 1R   (EMA-12 direction, also the hard $ stop)
      t2R   = entry + 2R
      t3R   = entry + 3R
      t4R   = entry + 4R
      trail_floor = entry + 2R  (trailing stop floor after 3R hit)

    BEARISH: all signs flipped.
    """

    def __init__(self, entry_price: float, ema12_at_entry: float, direction: str):
        dist = abs(entry_price - ema12_at_entry)
        if dist < 0.001:
            dist = entry_price * 0.005  # safety floor: 0.5% of price

        self.dist = dist
        self.shares = RISK_PER_TRADE / dist  # fractional shares
        self.capital = self.shares * entry_price  # $ deployed
        self.direction = direction
        self.entry = entry_price

        sign = 1 if direction == "BULLISH" else -1
        self.stop_price = entry_price - sign * dist  # 1R loss
        self.target_2r = entry_price + sign * 2 * dist
        self.target_3r = entry_price + sign * 3 * dist
        self.target_4r = entry_price + sign * 4 * dist
        self.trail_floor = entry_price + sign * TRAIL_LOCK_R * dist

    def pnl_dollar(self, exit_price: float, shares: float) -> float:
        sign = 1 if self.direction == "BULLISH" else -1
        return sign * (exit_price - self.entry) * shares

    def __repr__(self):
        sign = "+" if self.direction == "BULLISH" else "-"
        return (
            f"RLevels({self.direction}  entry={self.entry:.3f}  "
            f"dist={self.dist:.3f}  shares={self.shares:.4f}  "
            f"stop={self.stop_price:.3f}  "
            f"2R={self.target_2r:.3f}  3R={self.target_3r:.3f}  4R={self.target_4r:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  R-based tiered exit simulation on a bar series
# ─────────────────────────────────────────────────────────────────────────────


def _price_hit(bar_high: float, bar_low: float, target: float, direction: str) -> bool:
    """Did the bar reach `target`? For longs: high >= target. For shorts: low <= target."""
    return bar_high >= target if direction == "BULLISH" else bar_low <= target


def simulate_tiered_exits(
    bars: pd.DataFrame,
    entry_ts,
    rl: RLevels,
    use_ema_stop: bool,
    stop_timeframe_bars: pd.DataFrame | None = None,
) -> dict:
    """
    Walk bars after entry_ts and simulate the full tiered exit plan.

    Parameters
    ----------
    bars                : 10m bars (used for target checks in both strategies)
    entry_ts            : timestamp of entry bar (inclusive in check)
    rl                  : RLevels instance with all price levels
    use_ema_stop        : True=10m EMA stop, False=1h EMA stop
    stop_timeframe_bars : If use_ema_stop is False, supply 1h bars here

    Returns dict with full trade result including per-tier fills.
    """
    future = bars[bars.index > entry_ts]
    if future.empty:
        return None

    # ── State machine ──────────────────────────────────────────────────────────
    shares_remaining = rl.shares
    total_pnl = 0.0
    fills = []  # list of {ts, price, shares, reason}

    hit_2r = False
    hit_3r = False
    stop_price = rl.stop_price  # starts at 1R loss, moves to breakeven after 2R
    trail_active = False
    trail_stop = None
    high_watermark = rl.entry  # tracks best price for trailing

    # EMA stop state
    prev_e5_10m = float(bars.loc[entry_ts, "ema5"])
    prev_e12_10m = float(bars.loc[entry_ts, "ema12"])

    # 1h EMA stop state (Strategy 2)
    prev_e5_1h = None
    prev_e12_1h = None
    if not use_ema_stop and stop_timeframe_bars is not None:
        pre_1h = stop_timeframe_bars[stop_timeframe_bars.index <= entry_ts]
        if not pre_1h.empty:
            prev_e5_1h = float(pre_1h.iloc[-1]["ema5"])
            prev_e12_1h = float(pre_1h.iloc[-1]["ema12"])

    direction = rl.direction

    for ts, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        e5_10m = float(row["ema5"])
        e12_10m = float(row["ema12"])

        if shares_remaining <= 0:
            break

        # ── Update high watermark ──────────────────────────────────────────────
        if direction == "BULLISH":
            high_watermark = max(high_watermark, high)
        else:
            high_watermark = min(high_watermark, low)

        # ── Check profit targets (in order: 2R → 3R → 4R) ────────────────────

        # 2R target: sell 50% of original shares, stop → breakeven
        if not hit_2r and _price_hit(high, low, rl.target_2r, direction):
            sell_shares = rl.shares * TARGET_2R_SELL_FRAC
            sell_shares = min(sell_shares, shares_remaining)
            pnl = rl.pnl_dollar(rl.target_2r, sell_shares)
            total_pnl += pnl
            shares_remaining -= sell_shares
            fills.append(
                {
                    "ts": ts,
                    "price": rl.target_2r,
                    "shares": round(sell_shares, 6),
                    "reason": "TARGET_2R",
                    "pnl": round(pnl, 2),
                }
            )
            hit_2r = True
            stop_price = rl.entry  # move stop to breakeven

        # 3R target: sell 50% of remaining, activate trailing stop at 2R price
        if (
            hit_2r
            and not hit_3r
            and shares_remaining > 0
            and _price_hit(high, low, rl.target_3r, direction)
        ):
            sell_shares = shares_remaining * TARGET_3R_SELL_FRAC
            pnl = rl.pnl_dollar(rl.target_3r, sell_shares)
            total_pnl += pnl
            shares_remaining -= sell_shares
            fills.append(
                {
                    "ts": ts,
                    "price": rl.target_3r,
                    "shares": round(sell_shares, 6),
                    "reason": "TARGET_3R",
                    "pnl": round(pnl, 2),
                }
            )
            hit_3r = True
            trail_active = True
            trail_stop = rl.trail_floor  # initially locked at 2R price

        # Update trailing stop above 2R floor
        if trail_active and shares_remaining > 0:
            if direction == "BULLISH":
                new_trail = high_watermark - TRAIL_DISTANCE_R * rl.dist
                trail_stop = max(trail_stop, new_trail)
            else:
                new_trail = high_watermark + TRAIL_DISTANCE_R * rl.dist
                trail_stop = min(trail_stop, new_trail)

        # 4R target: sell everything remaining
        if (
            hit_3r
            and shares_remaining > 0
            and _price_hit(high, low, rl.target_4r, direction)
        ):
            pnl = rl.pnl_dollar(rl.target_4r, shares_remaining)
            total_pnl += pnl
            fills.append(
                {
                    "ts": ts,
                    "price": rl.target_4r,
                    "shares": round(shares_remaining, 6),
                    "reason": "TARGET_4R",
                    "pnl": round(pnl, 2),
                }
            )
            shares_remaining = 0
            break

        # ── Trailing stop hit (after 3R) ───────────────────────────────────────
        if trail_active and shares_remaining > 0 and trail_stop is not None:
            trail_hit = (direction == "BULLISH" and low <= trail_stop) or (
                direction == "BEARISH" and high >= trail_stop
            )
            if trail_hit:
                exit_p = trail_stop
                pnl = rl.pnl_dollar(exit_p, shares_remaining)
                total_pnl += pnl
                fills.append(
                    {
                        "ts": ts,
                        "price": round(exit_p, 4),
                        "shares": round(shares_remaining, 6),
                        "reason": "TRAIL_STOP",
                        "pnl": round(pnl, 2),
                    }
                )
                shares_remaining = 0
                break

        # ── EMA-based hard stop ────────────────────────────────────────────────
        stop_hit = False
        stop_exit_price = close

        if use_ema_stop:
            # Strategy 1: 10m EMA stop
            if direction == "BULLISH":
                cross = prev_e5_10m >= prev_e12_10m and e5_10m < e12_10m
                stop_hit = close < e12_10m or (cross and close < e5_10m)
            else:
                cross = prev_e5_10m <= prev_e12_10m and e5_10m > e12_10m
                stop_hit = close > e12_10m or (cross and close > e5_10m)
        else:
            # Strategy 2: 1h EMA stop (check 1h bar aligned to this 10m bar)
            if stop_timeframe_bars is not None and prev_e5_1h is not None:
                cur_1h_bars = stop_timeframe_bars[stop_timeframe_bars.index <= ts]
                if not cur_1h_bars.empty:
                    cur_1h = cur_1h_bars.iloc[-1]
                    e5_1h = float(cur_1h["ema5"])
                    e12_1h = float(cur_1h["ema12"])
                    if direction == "BULLISH":
                        cross_1h = prev_e5_1h >= prev_e12_1h and e5_1h < e12_1h
                        stop_hit = close < e12_1h or (cross_1h and close < e5_1h)
                    else:
                        cross_1h = prev_e5_1h <= prev_e12_1h and e5_1h > e12_1h
                        stop_hit = close > e12_1h or (cross_1h and close > e5_1h)
                    prev_e5_1h, prev_e12_1h = e5_1h, e12_1h

        # Also check manual stop_price level (breakeven after 2R, initial 1R before)
        manual_stop_hit = (direction == "BULLISH" and low <= stop_price) or (
            direction == "BEARISH" and high >= stop_price
        )

        if stop_hit or manual_stop_hit:
            exit_p = stop_price if manual_stop_hit else stop_exit_price
            reason = (
                "STOP_BE"
                if (hit_2r and manual_stop_hit)
                else ("STOP_10M" if use_ema_stop else "STOP_1H")
            )
            pnl = rl.pnl_dollar(exit_p, shares_remaining)
            total_pnl += pnl
            fills.append(
                {
                    "ts": ts,
                    "price": round(exit_p, 4),
                    "shares": round(shares_remaining, 6),
                    "reason": reason,
                    "pnl": round(pnl, 2),
                }
            )
            shares_remaining = 0
            break

        prev_e5_10m, prev_e12_10m = e5_10m, e12_10m

    # ── Max hold bail-out ──────────────────────────────────────────────────────
    if shares_remaining > 0 and not future.empty:
        last_bar = future.iloc[-1]
        exit_p = float(last_bar["close"])
        pnl = rl.pnl_dollar(exit_p, shares_remaining)
        total_pnl += pnl
        fills.append(
            {
                "ts": future.index[-1],
                "price": round(exit_p, 4),
                "shares": round(shares_remaining, 6),
                "reason": "MAX_HOLD",
                "pnl": round(pnl, 2),
            }
        )
        shares_remaining = 0

    if not fills:
        return None

    # ── Summarise ──────────────────────────────────────────────────────────────
    reasons = [f["reason"] for f in fills]
    last_fill = fills[-1]
    final_exit_ts = last_fill["ts"]
    final_exit_price = last_fill["price"]
    exit_reason = "+".join(dict.fromkeys(reasons))  # unique, ordered

    # Overall % return = total_pnl / capital_deployed
    pnl_pct = total_pnl / rl.capital * 100 if rl.capital > 0 else 0.0

    return {
        "total_pnl_dollar": round(total_pnl, 2),
        "pnl_pct": round(pnl_pct, 3),
        "shares": round(rl.shares, 6),
        "capital": round(rl.capital, 2),
        "risk_1r": round(rl.dist, 4),
        "stop_price": round(rl.stop_price, 4),
        "target_2r": round(rl.target_2r, 4),
        "target_3r": round(rl.target_3r, 4),
        "target_4r": round(rl.target_4r, 4),
        "final_exit_ts": final_exit_ts,
        "final_exit_price": final_exit_price,
        "exit_reason": exit_reason,
        "hit_2r": hit_2r,
        "hit_3r": hit_3r,
        "fills": fills,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1 — 10m stop, tiered exits
# ─────────────────────────────────────────────────────────────────────────────


def run_trade_s1(symbol: str, signal_date: str, direction: str) -> dict | None:
    entry_date = next_trading_day(signal_date)
    today_utc = datetime.now(timezone.utc)

    fetch_start = datetime(
        entry_date.year, entry_date.month, entry_date.day, 3, 0, tzinfo=ET
    ).astimezone(timezone.utc)
    fetch_end = min(
        datetime(entry_date.year, entry_date.month, entry_date.day, tzinfo=ET)
        + timedelta(days=MAX_HOLD_DAYS + 3),
        today_utc,
    )

    bars_10m = get_intraday(symbol, "10Min", fetch_start, fetch_end)
    if bars_10m.empty:
        return None
    bars_10m = add_emas_intraday(bars_10m)

    pm_state = premarket_ema_state(bars_10m, entry_date, direction)
    entry_ts, entry_price, ema12_entry = find_entry(
        bars_10m, entry_date, direction, pm_state
    )
    if entry_ts is None:
        return None

    rl = RLevels(entry_price, ema12_entry, direction)
    result = simulate_tiered_exits(bars_10m, entry_ts, rl, use_ema_stop=True)
    if result is None:
        return None

    return {
        "symbol": symbol,
        "scanner": None,
        "direction": direction,
        "signal_date": signal_date,
        "pm_state": pm_state,
        "entry_date": entry_ts.strftime("%Y-%m-%d %H:%M ET"),
        "entry_price": round(entry_price, 4),
        "shares": result["shares"],
        "capital": result["capital"],
        "risk_1r": result["risk_1r"],
        "stop_price": result["stop_price"],
        "target_2r": result["target_2r"],
        "target_3r": result["target_3r"],
        "target_4r": result["target_4r"],
        "hit_2r": result["hit_2r"],
        "hit_3r": result["hit_3r"],
        "exit_date": result["final_exit_ts"].strftime("%Y-%m-%d %H:%M ET"),
        "exit_price": result["final_exit_price"],
        "exit_reason": result["exit_reason"],
        "pnl_pct": result["pnl_pct"],
        "pnl_dollar": result["total_pnl_dollar"],
        "strategy": "S1_10m_stop",
        "_fills": result["fills"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2 — 1h stop, targets on 10m, tiered exits
# ─────────────────────────────────────────────────────────────────────────────


def run_trade_s2(symbol: str, signal_date: str, direction: str) -> dict | None:
    entry_date = next_trading_day(signal_date)
    today_utc = datetime.now(timezone.utc)

    fetch_start = datetime(
        entry_date.year, entry_date.month, entry_date.day, 3, 0, tzinfo=ET
    ).astimezone(timezone.utc)
    fetch_end = min(
        datetime(entry_date.year, entry_date.month, entry_date.day, tzinfo=ET)
        + timedelta(days=MAX_HOLD_DAYS + 3),
        today_utc,
    )

    bars_10m = get_intraday(symbol, "10Min", fetch_start, fetch_end)
    bars_1h = get_intraday(symbol, "1Hour", fetch_start, fetch_end)
    if bars_10m.empty or bars_1h.empty:
        return None

    bars_10m = add_emas_intraday(bars_10m)
    bars_1h = add_emas_intraday(bars_1h)

    pm_state = premarket_ema_state(bars_10m, entry_date, direction)
    entry_ts, entry_price, ema12_entry = find_entry(
        bars_10m, entry_date, direction, pm_state
    )
    if entry_ts is None:
        return None

    rl = RLevels(entry_price, ema12_entry, direction)
    result = simulate_tiered_exits(
        bars_10m, entry_ts, rl, use_ema_stop=False, stop_timeframe_bars=bars_1h
    )
    if result is None:
        return None

    return {
        "symbol": symbol,
        "scanner": None,
        "direction": direction,
        "signal_date": signal_date,
        "pm_state": pm_state,
        "entry_date": entry_ts.strftime("%Y-%m-%d %H:%M ET"),
        "entry_price": round(entry_price, 4),
        "shares": result["shares"],
        "capital": result["capital"],
        "risk_1r": result["risk_1r"],
        "stop_price": result["stop_price"],
        "target_2r": result["target_2r"],
        "target_3r": result["target_3r"],
        "target_4r": result["target_4r"],
        "hit_2r": result["hit_2r"],
        "hit_3r": result["hit_3r"],
        "exit_date": result["final_exit_ts"].strftime("%Y-%m-%d %H:%M ET"),
        "exit_price": result["final_exit_price"],
        "exit_reason": result["exit_reason"],
        "pnl_pct": result["pnl_pct"],
        "pnl_dollar": result["total_pnl_dollar"],
        "strategy": "S2_1h_stop",
        "_fills": result["fills"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio summary + P&L display
# ─────────────────────────────────────────────────────────────────────────────


def make_summary(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    # Drop internal _fills before building DataFrame
    clean = [{k: v for k, v in t.items() if k != "_fills"} for t in trades]
    df = pd.DataFrame(clean).sort_values("entry_date").reset_index(drop=True)
    df["trade_#"] = df.index + 1
    df["cum_pnl"] = df["pnl_dollar"].cumsum()
    df["portfolio"] = STARTING_CAPITAL + df["cum_pnl"]
    return df


def print_fills(trade: dict):
    """Print the detailed fill breakdown for one trade."""
    fills = trade.get("_fills", [])
    if not fills:
        return
    print(f"      Fills:")
    for f in fills:
        ts_str = (
            f["ts"].strftime("%Y-%m-%d %H:%M")
            if hasattr(f["ts"], "strftime")
            else str(f["ts"])
        )
        pnl_sign = "+" if f["pnl"] >= 0 else ""
        print(
            f"        {ts_str}  {f['reason']:<12}  "
            f"{f['shares']:.4f} sh @ ${f['price']:.3f}  "
            f"P&L: {pnl_sign}${f['pnl']:.2f}"
        )


def print_report(
    trades: list, df: pd.DataFrame, title: str, verbose_fills: bool = False
):
    if df.empty:
        print(f"\n  {title}: No completed trades.\n")
        return

    W = 82
    total_pnl = df["pnl_dollar"].sum()
    wins = df[df["pnl_dollar"] > 0]
    losses = df[df["pnl_dollar"] <= 0]
    win_rate = len(wins) / len(df) * 100
    avg_win = wins["pnl_pct"].mean() if not wins.empty else 0
    avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0
    best = df.loc[df["pnl_dollar"].idxmax()]
    worst = df.loc[df["pnl_dollar"].idxmin()]
    port_ret = total_pnl / STARTING_CAPITAL * 100
    final_port = STARTING_CAPITAL + total_pnl
    avg_risk = df["risk_1r"].mean() if "risk_1r" in df.columns else 0
    avg_shares = df["shares"].mean() if "shares" in df.columns else 0
    avg_cap = df["capital"].mean() if "capital" in df.columns else 0
    hit_2r_pct = df["hit_2r"].mean() * 100 if "hit_2r" in df.columns else 0
    hit_3r_pct = df["hit_3r"].mean() * 100 if "hit_3r" in df.columns else 0

    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")
    print(
        f"  Trades        : {len(df)}  (🟢 {len(wins)} wins  🔴 {len(losses)} losses)"
    )
    print(f"  Win rate      : {win_rate:.1f}%")
    print(f"  Avg win       : +{avg_win:.2f}%       Avg loss: {avg_loss:.2f}%")
    print(
        f"  Best trade    : {best['symbol']} {best['pnl_pct']:+.2f}%  (${best['pnl_dollar']:+.2f})"
    )
    print(
        f"  Worst trade   : {worst['symbol']} {worst['pnl_pct']:+.2f}%  (${worst['pnl_dollar']:+.2f})"
    )
    print(f"  Total P&L     : ${total_pnl:+,.2f}")
    print(
        f"  Portfolio     : ${final_port:,.2f}  ({port_ret:+.2f}% return on ${STARTING_CAPITAL:,.0f})"
    )
    print(f"\n  Position sizing (avg per trade):")
    print(
        f"    Risk (1R $)   : ${RISK_PER_TRADE:.0f}  |  Avg 1R distance: ${avg_risk:.3f}"
    )
    print(
        f"    Avg shares    : {avg_shares:.2f}  |  Avg capital deployed: ${avg_cap:.2f}"
    )
    print(
        f"    Trades hitting 2R: {hit_2r_pct:.1f}%    Trades hitting 3R: {hit_3r_pct:.1f}%"
    )

    # Exit reason breakdown
    print(f"\n  Exit reason breakdown:")
    for reason, grp in df.groupby("exit_reason"):
        avg_r = grp["pnl_pct"].mean()
        print(f"    {reason:<28}  {len(grp):>3} trade(s)  avg {avg_r:+.2f}%")

    # Pre-market state breakdown
    if "pm_state" in df.columns:
        print(f"\n  Pre-market EMA alignment vs outcome:")
        for state, grp in df.groupby("pm_state"):
            wr = (grp["pnl_dollar"] > 0).mean() * 100
            avg = grp["pnl_pct"].mean()
            print(
                f"    {state:<10}  {len(grp):>3} trade(s)  win rate {wr:.0f}%  avg {avg:+.2f}%"
            )

    # Trade ledger header
    print(
        f"\n  {'#':<4} {'Sym':<7} {'Dir':<8} {'PM':<8} "
        f"{'Entry Date/Time':<22} {'Entry$':>8} "
        f"{'1R Dist':>8} {'Shares':>8} "
        f"{'SL':>8} {'2R':>8} {'3R':>8} {'4R':>8} "
        f"{'Hit2R':>5} {'Hit3R':>5} "
        f"{'Exit Date/Time':<22} {'Exit$':>8} "
        f"{'%P&L':>7} {'$P&L':>8} {'Cum$':>10}  Reason"
    )
    print(f"  {'─' * 200}")

    for idx, (_, r) in enumerate(df.iterrows()):
        arrow = "▲" if r["pnl_dollar"] > 0 else "▼"
        scan = r.get("scanner", "–") or "–"
        pm = r.get("pm_state", "–") or "–"
        h2r = "✓" if r.get("hit_2r") else "✗"
        h3r = "✓" if r.get("hit_3r") else "✗"
        print(
            f"  {int(r['trade_#']):<4} {r['symbol']:<7} {r['direction']:<8} {pm:<8} "
            f"{str(r['entry_date']):<22} ${r['entry_price']:>7.3f} "
            f"${r.get('risk_1r', 0):>7.3f} {r.get('shares', 0):>8.3f} "
            f"${r.get('stop_price', 0):>7.3f} ${r.get('target_2r', 0):>7.3f} "
            f"${r.get('target_3r', 0):>7.3f} ${r.get('target_4r', 0):>7.3f} "
            f"  {h2r:<5} {h3r:<5} "
            f"{str(r['exit_date']):<22} ${r['exit_price']:>7.3f} "
            f"{r['pnl_pct']:>+6.2f}% "
            f"{arrow}${abs(r['pnl_dollar']):>7.2f} "
            f"${r['cum_pnl']:>+9.2f}  {r['exit_reason']}"
        )

        if verbose_fills and idx < len(trades):
            # find matching trade by symbol + entry_date
            match = next(
                (
                    t
                    for t in trades
                    if t.get("symbol") == r["symbol"]
                    and t.get("entry_date") == r["entry_date"]
                ),
                None,
            )
            if match:
                print_fills(match)

    print(f"{'=' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Watchlist
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = [
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


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EMA Backtest — Alpaca API  (R-based sizing, tiered exits)"
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date YYYY-MM-DD (required unless --signals-csv)",
    )
    parser.add_argument(
        "--signals-csv",
        default=None,
        help="Pre-saved signals CSV (symbol, signal_date, direction, scanner)",
    )
    parser.add_argument("--symbol", default=None, help="Single symbol mode")
    parser.add_argument(
        "--scanner",
        type=int,
        choices=[1, 2],
        default=None,
        help="Daily scanner to use: 1=5/12 cross, 2=5/12×34/50 (default: both)",
    )
    parser.add_argument(
        "--strategy",
        choices=["1", "2", "both"],
        default="both",
        help="Backtest strategy: 1=10m stop, 2=1h stop (default: both)",
    )
    parser.add_argument(
        "--no-bearish", action="store_true", help="Skip bearish signals (long-only)"
    )
    parser.add_argument(
        "--fills", action="store_true", help="Print detailed fill breakdown per trade"
    )
    parser.add_argument(
        "--out", default=None, help="Output CSV prefix (default: bt_alpaca_YYYYMMDD)"
    )
    args = parser.parse_args()

    if not _API_KEY or not _SECRET_KEY:
        print("\n[ERROR] ALPACA_API_KEY / ALPACA_SECRET_KEY not found in .env\n")
        sys.exit(1)

    if not args.start_date and not args.signals_csv:
        parser.error("Provide --start-date or --signals-csv")

    # ── Step 1: signals ───────────────────────────────────────────────────────
    if args.signals_csv:
        print(f"\n  Loading signals from {args.signals_csv} …")
        sig_df = pd.read_csv(args.signals_csv)
    else:
        run_s1 = args.scanner in (None, 1)
        run_s2 = args.scanner in (None, 2)
        symbols = [args.symbol.upper()] if args.symbol else WATCHLIST
        sig_df = run_daily_scanner(symbols, args.start_date, run_s1, run_s2)

    if sig_df.empty:
        print("\n  No signals to backtest. Exiting.\n")
        sys.exit(0)

    if args.no_bearish:
        sig_df = sig_df[sig_df["direction"] == "BULLISH"]
        print(f"  [--no-bearish] {len(sig_df)} bullish signal(s) remaining.")

    # ── Step 2: backtest ──────────────────────────────────────────────────────
    s1_trades, s2_trades = [], []

    print(f"\n{'=' * 70}")
    print(f"  STEP 2 — BACKTEST  ({len(sig_df)} signal(s))  risk=$100/trade")
    print(f"{'=' * 70}")

    for _, row in sig_df.iterrows():
        sym = str(row["symbol"])
        sdate = str(row["signal_date"])
        dirn = str(row["direction"])
        scnr = str(row.get("scanner", "?"))

        print(f"  {sym:<8} {sdate}  {dirn:<8} [{scnr}] …", end="  ", flush=True)
        parts = []

        if args.strategy in ("1", "both"):
            t = run_trade_s1(sym, sdate, dirn)
            if t:
                t["scanner"] = scnr
                s1_trades.append(t)
                h2 = "2R✓" if t["hit_2r"] else "2R✗"
                h3 = "3R✓" if t["hit_3r"] else "3R✗"
                parts.append(
                    f"S1: {t['pnl_pct']:+.2f}%  ${t['pnl_dollar']:+.2f}  "
                    f"{h2} {h3}  [{t['exit_reason']}]"
                )
            else:
                parts.append("S1: no entry")

        if args.strategy in ("2", "both"):
            t = run_trade_s2(sym, sdate, dirn)
            if t:
                t["scanner"] = scnr
                s2_trades.append(t)
                h2 = "2R✓" if t["hit_2r"] else "2R✗"
                h3 = "3R✓" if t["hit_3r"] else "3R✗"
                parts.append(
                    f"S2: {t['pnl_pct']:+.2f}%  ${t['pnl_dollar']:+.2f}  "
                    f"{h2} {h3}  [{t['exit_reason']}]"
                )
            else:
                parts.append("S2: no entry")

        print("  |  ".join(parts))

    # ── Step 3: reports ───────────────────────────────────────────────────────
    today_str = datetime.now().strftime("%Y%m%d")
    out_prefix = args.out or f"bt_alpaca_{today_str}"

    if s1_trades:
        df1 = make_summary(s1_trades)
        print_report(
            s1_trades,
            df1,
            "STRATEGY 1 — 10m EMA STOP  |  R-Based Sizing  |  Tiered Exits",
            verbose_fills=args.fills,
        )
        f1 = f"{out_prefix}_s1.csv"
        df1.to_csv(f1, index=False)
        print(f"  📄 Strategy 1 → {f1}")

    if s2_trades:
        df2 = make_summary(s2_trades)
        print_report(
            s2_trades,
            df2,
            "STRATEGY 2 — 1h EMA STOP  |  R-Based Sizing  |  Tiered Exits",
            verbose_fills=args.fills,
        )
        f2 = f"{out_prefix}_s2.csv"
        df2.to_csv(f2, index=False)
        print(f"  📄 Strategy 2 → {f2}\n")

    if not s1_trades and not s2_trades:
        print("\n  No trades completed.\n")


if __name__ == "__main__":
    main()
