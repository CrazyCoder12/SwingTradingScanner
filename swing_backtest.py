"""
S&P Swing Trading Strategy — "Beat the Market" Backtester
==========================================================
Multi-layer daily swing strategy on a 30-stock watchlist, shared capital pool.

Layers:
  1. Regime Filter   : Close > 200 EMA → bull mode (long only). Bear = sit in cash.
  2. Volatility      : HV20 (VIX proxy) — skip entries in panic, half-size in caution.
  3. Signal          : 21 EMA Pullback — 2-period RSI < 15 + low touched 21 EMA.
                       (EMA crossover removed — lower win rate, hurts R/R)
  4. Position Sizing : ATR-based stop (1.0× ATR), risk 2% of portfolio per trade.
                       Max 5 concurrent positions, max 20% capital per position.
  5. Exit System     : Partial exit at 2:1 R/R → breakeven stop → trail 10 EMA.

FOMC Filter: No new entries the trading day before FOMC rate decisions.

Usage:
    python swing_backtest.py --year 2025
    python swing_backtest.py --start 2025-01-01 --end 2025-12-31
"""

import argparse
import sys
from datetime import datetime, timedelta, date
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tabulate import tabulate
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD",  "META",
    "GOOGL","AMZN", "TSLA", "NFLX", "CRM",
    "ORCL", "INTC", "QCOM", "MU",   "AVGO",
    "JPM",  "BAC",  "GS",   "MS",   "V",
    "SPY",  "QQQ",  "IWM",  "XLK",  "XLF",
    "COIN", "HOOD", "SQ",   "PYPL", "SHOP"
]

INITIAL_CAPITAL  = 100_000    # Starting portfolio ($)
RISK_PER_TRADE   = 0.02       # Risk 2% of portfolio per trade
MAX_POS_PCT      = 0.20       # Max 20% of capital in a single position
MAX_CONCURRENT   = 5          # Max open positions at once
ATR_STOP_MULT    = 1.0        # Stop = 1.0× ATR(14) below entry  [tightened from 1.5]
PARTIAL_RR       = 2.0        # Take half off at 2:1 R/R target  [raised from 1:1]
RISK_FREE_RATE   = 0.045      # 2025 risk-free rate ~4.5%
WARMUP_DAYS      = 320        # Calendar days before start for indicator warmup

# Indicator periods
EMA_REGIME  = 200
EMA_TRAIL   = 10
EMA_ANCHOR  = 21
ATR_PERIOD  = 14
RSI_PERIOD  = 2

# Signal thresholds
RSI_OVERSOLD    = 15          # 2-period RSI below this = oversold
EMA21_TOUCH_PCT = 0.008       # Low within 0.8% of 21 EMA counts as touching it

# Volatility filter (HV20 = 20-day annualized vol × 100, close proxy for VIX)
HV_PANIC   = 35               # HV20 > 35 → skip new entries entirely
HV_CAUTION = 20               # HV20 > 20 → half position size

# FOMC 2025: no new entries the day BEFORE the Fed decision
FOMC_NO_ENTRY = {
    "2025-01-28", "2025-03-18", "2025-05-06", "2025-06-17",
    "2025-07-29", "2025-09-16", "2025-10-28", "2025-12-09",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER — Alpaca Daily Bars
# ─────────────────────────────────────────────────────────────────────────────

def fetch_daily_data(symbol: str, start: date, end: date,
                     api_key: str, secret_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV from Alpaca v2.
    Fetches WARMUP_DAYS extra before start so all indicators are fully warmed up.
    """
    try:
        import requests

        fetch_start = (datetime.combine(start, datetime.min.time())
                       - timedelta(days=WARMUP_DAYS)).date()

        url     = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        params = {
            "symbols":    symbol,
            "timeframe":  "1Day",
            "start":      fetch_start.isoformat(),
            "end":        end.isoformat(),
            "feed":       "iex",
            "adjustment": "split",
            "limit":      10000,
        }

        all_bars   = []
        next_token = None

        while True:
            if next_token:
                params["page_token"] = next_token
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data       = resp.json()
            bars       = data.get("bars", {}).get(symbol, [])
            all_bars.extend(bars)
            next_token = data.get("next_page_token")
            if not next_token:
                break

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"]).dt.date
        df = df.set_index("t")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                 "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    except Exception as e:
        print(f"    [!] {symbol}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df["high"], df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_200"] = _ema(df["close"], EMA_REGIME)
    df["ema_21"]  = _ema(df["close"], EMA_ANCHOR)
    df["ema_10"]  = _ema(df["close"], EMA_TRAIL)
    df["atr14"]   = _atr(df, ATR_PERIOD)
    df["rsi2"]    = _rsi(df["close"], RSI_PERIOD)
    daily_ret     = df["close"].pct_change()
    df["hv20"]    = daily_ret.rolling(20).std() * np.sqrt(252) * 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SYMBOL BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(symbol_dfs: dict, start: date, end: date) -> tuple:
    """
    Shared-capital backtest across all symbols.
    Iterates day-by-day, exits first, then enters new positions by priority
    (lowest 2-period RSI = most oversold = highest priority).

    Returns (trades_list, equity_series, filter_stats_dict)
    """
    # Add indicators to all DFs
    symbol_data = {}
    for sym, df in symbol_dfs.items():
        symbol_data[sym] = add_indicators(df)

    # All unique trading days in backtest window
    all_days = sorted({
        d for sym_df in symbol_data.values()
        for d in sym_df.index
        if start <= d <= end
    })

    capital   = float(INITIAL_CAPITAL)
    positions = {}       # {symbol: position_dict}
    trades    = []

    equity_index  = []
    equity_values = []

    filter_stats = {
        "n_regime_blocked": 0,
        "n_hv_blocked":     0,
        "n_fomc_blocked":   0,
        "n_full_blocked":   0,
        "total_deployed":   0.0,   # gross capital put to work (sum of all entry costs)
        "days_in_market":   0,     # days with ≥1 open position
        "total_days":       0,
        "sum_utilization":  0.0,   # sum of daily deployed % (for averaging)
    }

    for today in all_days:
        today_str    = str(today)
        exited_today = set()   # BUG FIX 1: prevent same-day exit + re-entry

        # ── Phase 1: Process exits for all open positions ─────────────────────
        for sym in list(positions.keys()):
            sym_df = symbol_data[sym]
            if today not in sym_df.index:
                continue

            row      = sym_df.loc[today]
            position = positions[sym]

            entry     = position["entry_price"]
            stop      = position["stop_price"]
            stop_dist = position["stop_distance"]
            target_2r = entry + PARTIAL_RR * stop_dist   # 2:1 R/R
            close     = row["close"]
            ema10     = row["ema_10"]
            exit_reason = None
            exit_price  = close

            if not position["half_exited"]:
                # Full position: watch for stop or 2:1 partial
                if close <= stop:
                    exit_reason = "Hard Stop Loss"
                    exit_price  = stop
                elif close >= target_2r:
                    # Hit 2:1 → sell half, move stop to breakeven
                    half  = position["shares_total"] // 2
                    remain = position["shares_total"] - half
                    capital += half * close
                    position["shares_remaining"] = remain
                    position["half_exited"]      = True
                    position["stop_price"]       = entry   # breakeven
                    position["partial_px"]       = close
                    position["partial_date"]     = today_str
                    position["partial_shares"]   = half
            else:
                # Half position: trail with 10 EMA, floor at breakeven
                if close <= position["stop_price"]:
                    exit_reason = "Breakeven Stop"
                    exit_price  = position["stop_price"]
                elif close < ema10:
                    exit_reason = "10 EMA Trail Exit"
                    exit_price  = close

            if exit_reason:
                shares_out = position["shares_remaining"]
                capital   += shares_out * exit_price

                total_pnl = (exit_price - entry) * shares_out
                if position["half_exited"]:
                    total_pnl += (position["partial_px"] - entry) * position["partial_shares"]

                invested   = entry * position["shares_total"]
                return_pct = (total_pnl / invested * 100) if invested > 0 else 0
                hold_days  = (today - date.fromisoformat(position["entry_date"])).days

                trades.append({
                    "symbol":       sym,
                    "entry_date":   position["entry_date"],
                    "exit_date":    today_str,
                    "entry_price":  round(entry, 2),
                    "exit_price":   round(exit_price, 2),
                    "shares":       position["shares_total"],
                    "initial_stop": round(position["initial_stop"], 2),
                    "stop_dist":    round(stop_dist, 2),
                    "entry_cost":   round(entry * position["shares_total"], 2),
                    "partial":      position["half_exited"],
                    "partial_px":   round(position.get("partial_px") or 0, 2) if position["half_exited"] else None,
                    "exit_reason":  exit_reason,
                    "pnl":          round(total_pnl, 2),
                    "return_pct":   round(return_pct, 2),
                    "hold_days":    hold_days,
                })
                del positions[sym]
                exited_today.add(sym)   # BUG FIX 1: mark as exited this bar

        # ── Phase 2: Record equity + compute portfolio value for sizing ───────
        pos_value = 0.0
        for sym, pos in positions.items():
            if today in symbol_data[sym].index:
                pos_value += pos["shares_remaining"] * symbol_data[sym].loc[today]["close"]
        portfolio_value_today = capital + pos_value   # BUG FIX 2: true portfolio value

        equity_index.append(today)
        equity_values.append(portfolio_value_today)

        # Utilization tracking
        filter_stats["total_days"] += 1
        deployed_pct = (pos_value / portfolio_value_today * 100) if portfolio_value_today > 0 else 0
        filter_stats["sum_utilization"] += deployed_pct
        if positions:
            filter_stats["days_in_market"] += 1

        # ── Phase 3: Look for new entries ─────────────────────────────────────
        if today_str in FOMC_NO_ENTRY:
            filter_stats["n_fomc_blocked"] += 1
            continue

        if len(positions) >= MAX_CONCURRENT:
            filter_stats["n_full_blocked"] += 1
            continue

        candidates = []

        for sym in symbol_data:
            if sym in positions or sym in exited_today:   # BUG FIX 1
                continue    # already holding or just exited today

            sym_df = symbol_data[sym]
            if today not in sym_df.index:
                continue

            idx = sym_df.index.get_loc(today)
            if idx == 0:
                continue

            row  = sym_df.loc[today]
            close  = row["close"]
            ema200 = row["ema_200"]
            ema21  = row["ema_21"]
            hv20   = row["hv20"]
            atr14  = row["atr14"]
            rsi2   = row["rsi2"]

            if any(pd.isna(x) for x in [ema200, ema21, hv20, atr14, rsi2]):
                continue

            # Layer 1: Regime filter
            if close < ema200:
                filter_stats["n_regime_blocked"] += 1
                continue

            # Layer 2: Volatility filter
            if hv20 > HV_PANIC:
                filter_stats["n_hv_blocked"] += 1
                continue

            size_mult = 0.5 if hv20 > HV_CAUTION else 1.0

            # Layer 3: 21 EMA Pullback signal
            low_touched = row["low"] <= ema21 * (1 + EMA21_TOUCH_PCT)
            rsi_os      = rsi2 < RSI_OVERSOLD
            bullish     = close > (row["low"] + (row["high"] - row["low"]) * 0.4)

            if low_touched and rsi_os and bullish:
                candidates.append({
                    "symbol":    sym,
                    "close":     close,
                    "atr14":     atr14,
                    "rsi2":      rsi2,
                    "size_mult": size_mult,
                })

        # Sort candidates: lowest RSI = most oversold = top priority
        candidates.sort(key=lambda x: x["rsi2"])

        for cand in candidates:
            if len(positions) >= MAX_CONCURRENT:
                break

            sym       = cand["symbol"]
            close     = cand["close"]
            atr14     = cand["atr14"]
            size_mult = cand["size_mult"]

            stop_dist    = ATR_STOP_MULT * atr14
            stop_price   = close - stop_dist
            # BUG FIX 2 & 3: use total portfolio value, not just cash balance
            risk_dollars = portfolio_value_today * RISK_PER_TRADE * size_mult
            max_alloc    = portfolio_value_today * MAX_POS_PCT

            if stop_dist <= 0:
                continue

            shares     = int(risk_dollars / stop_dist)
            max_shares = int(max_alloc / close)
            shares     = min(shares, max_shares)

            if shares <= 0:
                continue

            cost = shares * close
            if cost > capital * 0.95:   # leave 5% cash buffer
                shares = int(capital * 0.95 / close)
                cost   = shares * close

            if shares <= 0:
                continue

            capital -= cost
            filter_stats["total_deployed"] += cost   # track gross capital deployed

            positions[sym] = {
                "entry_date":      today_str,
                "entry_price":     close,
                "stop_price":      stop_price,
                "initial_stop":    stop_price,
                "stop_distance":   stop_dist,
                "shares_total":    shares,
                "shares_remaining": shares,
                "half_exited":     False,
                "partial_px":      None,
                "partial_date":    None,
                "partial_shares":  None,
            }

    # ── Force-close any remaining positions at end of period ─────────────────
    for sym, position in list(positions.items()):
        sym_df   = symbol_data[sym]
        bt_days  = [d for d in sym_df.index if start <= d <= end]
        if not bt_days:
            continue
        last_day   = max(bt_days)
        last_close = sym_df.loc[last_day]["close"]
        last_str   = str(last_day)

        entry      = position["entry_price"]
        shares_out = position["shares_remaining"]
        capital   += shares_out * last_close

        total_pnl = (last_close - entry) * shares_out
        if position["half_exited"]:
            total_pnl += (position["partial_px"] - entry) * position["partial_shares"]

        invested   = entry * position["shares_total"]
        return_pct = (total_pnl / invested * 100) if invested > 0 else 0
        hold_days  = (date.fromisoformat(last_str) - date.fromisoformat(position["entry_date"])).days

        trades.append({
            "symbol":       sym,
            "entry_date":   position["entry_date"],
            "exit_date":    last_str,
            "entry_price":  round(entry, 2),
            "exit_price":   round(last_close, 2),
            "shares":       position["shares_total"],
            "initial_stop": round(position["initial_stop"], 2),
            "stop_dist":    round(position["stop_distance"], 2),
            "entry_cost":   round(entry * position["shares_total"], 2),
            "partial":      position["half_exited"],
            "partial_px":   round(position.get("partial_px") or 0, 2) if position["half_exited"] else None,
            "exit_reason":  "End of Period",
            "pnl":          round(total_pnl, 2),
            "return_pct":   round(return_pct, 2),
            "hold_days":    hold_days,
        })

    equity_series = pd.Series(equity_values, index=equity_index, name="portfolio")
    return trades, equity_series, filter_stats


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series, initial: float,
                    trading_days: int) -> dict:
    daily_ret  = equity.pct_change().dropna()
    total_ret  = (equity.iloc[-1] / initial) - 1
    ann_ret    = (1 + total_ret) ** (252 / max(trading_days, 1)) - 1
    ann_std    = daily_ret.std() * np.sqrt(252)
    sharpe     = (ann_ret - RISK_FREE_RATE) / ann_std if ann_std > 0 else 0
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    return {
        "total_return": total_ret,
        "ann_return":   ann_ret,
        "ann_std":      ann_std,
        "sharpe":       sharpe,
        "max_drawdown": drawdown.min(),
    }


def compute_buyhold_metrics(spy_df: pd.DataFrame, start: date,
                             end: date, initial: float) -> dict:
    """Buy SPY at first-bar open, sell at last-bar close."""
    bt = spy_df[(spy_df.index >= start) & (spy_df.index <= end)]
    if bt.empty:
        return {}
    buy_price  = bt.iloc[0]["open"]
    shares     = int(initial / buy_price)
    cash_left  = initial - shares * buy_price
    equity     = bt["close"] * shares + cash_left
    equity.index = pd.to_datetime(equity.index)
    metrics    = compute_metrics(equity, initial, len(bt))
    metrics["buy_price"]  = buy_price
    metrics["sell_price"] = bt.iloc[-1]["close"]
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_results(trades: list, equity: pd.Series, spy_df: pd.DataFrame,
                    start: date, end: date, filter_stats: dict):

    SEP = "=" * 68

    print(f"\n{SEP}")
    print(f"  SWING STRATEGY BACKTEST — 30-STOCK WATCHLIST")
    print(f"  Period  : {start} → {end}")
    print(f"  Capital : ${INITIAL_CAPITAL:,.0f}  |  Risk/trade: {RISK_PER_TRADE*100:.0f}%  |  "
          f"Stop: {ATR_STOP_MULT}×ATR  |  Partial: {PARTIAL_RR:.0f}:1")
    print(f"  Max pos.: {MAX_CONCURRENT} concurrent  |  Max/position: {MAX_POS_PCT*100:.0f}%")
    print(SEP)

    # ── Filter stats ──────────────────────────────────────────────────────────
    print(f"\n  FILTER ACTIVITY (symbol-day events blocked)")
    print(f"  {'Regime  (<200 EMA — bear mode)':<38} {filter_stats['n_regime_blocked']:>6}")
    print(f"  {'Volatility (HV20 > '+str(HV_PANIC)+'% panic)':<38} {filter_stats['n_hv_blocked']:>6}")
    print(f"  {'FOMC blackout days':<38} {filter_stats['n_fomc_blocked']:>6}")
    print(f"  {'Max positions full':<38} {filter_stats['n_full_blocked']:>6}")

    if not trades:
        print("\n  No trades generated. Check API keys / date range.\n")
        return

    trades_df = pd.DataFrame(trades)
    n_total   = len(trades_df)
    n_wins    = (trades_df["pnl"] > 0).sum()
    n_loss    = (trades_df["pnl"] <= 0).sum()
    win_rate  = n_wins / n_total * 100
    total_pnl = trades_df["pnl"].sum()
    avg_win   = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if n_wins > 0 else 0
    avg_loss  = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if n_loss > 0 else 0
    rr_ratio  = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    avg_hold  = trades_df["hold_days"].mean()

    # Pre-compute capital utilisation stats
    total_deployed   = filter_stats["total_deployed"]
    days_in_market   = filter_stats["days_in_market"]
    total_days       = filter_stats["total_days"]
    avg_utilization  = filter_stats["sum_utilization"] / total_days if total_days else 0
    capital_turnover = total_deployed / INITIAL_CAPITAL

    # ── INVESTMENT AUDIT ──────────────────────────────────────────────────────
    print(f"\n\n  INVESTMENT AUDIT — GENUINE RETURN VERIFICATION\n")
    sanity_check     = INITIAL_CAPITAL + total_pnl
    final_equity_val = equity.iloc[-1]
    match_ok         = abs(sanity_check - final_equity_val) < 1.0  # within $1 rounding

    audit_rows = [
        ["Starting Capital",                 f"${INITIAL_CAPITAL:>12,.2f}"],
        ["Sum of all trade P&Ls",            f"${total_pnl:>+12,.2f}"],
        ["Expected final equity",            f"${sanity_check:>12,.2f}"],
        ["Actual equity curve (last value)", f"${final_equity_val:>12,.2f}"],
        ["Accounting match (< $1 error)",    "YES ✓" if match_ok else "NO — BUG"],
        ["", ""],
        ["Gross capital deployed (all trades)", f"${total_deployed:>12,.0f}"],
        ["Capital turnover (deployed/initial)", f"{capital_turnover:>11.1f}x"],
        ["Avg position size (per trade)",       f"${total_deployed/n_total:>12,.0f}"],
        ["Avg capital utilization (daily)",     f"{avg_utilization:>11.1f}%"],
        ["Days with ≥1 open position",          f"{days_in_market:>11} / {total_days} ({days_in_market/total_days*100:.0f}%)"],
    ]
    print(tabulate(audit_rows, headers=["Check", "Value"],
                   tablefmt="rounded_outline", showindex=False))

    # ── All Trades ────────────────────────────────────────────────────────────
    print(f"\n\n  ALL TRADES\n")
    log_rows = []
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        pnl_str  = f"+${t['pnl']:,.0f}" if t["pnl"] >= 0 else f"-${abs(t['pnl']):,.0f}"
        half_str = f"${t['partial_px']}" if t["partial"] else "-"
        log_rows.append([
            t["symbol"],
            t["entry_date"],
            t["exit_date"],
            t["entry_price"],
            t["exit_price"],
            t["shares"],
            t["initial_stop"],
            "Y" if t["partial"] else "-",
            half_str,
            f"{t['return_pct']:+.1f}%",
            pnl_str,
            t["hold_days"],
            t["exit_reason"][:20],
        ])
    print(tabulate(
        log_rows,
        headers=["Sym", "Entry", "Exit", "Entry$", "Exit$",
                 "Shs", "Stop$", "2:1?", "Half@",
                 "Ret%", "P&L", "Days", "Exit Reason"],
        tablefmt="rounded_outline",
        showindex=False,
    ))

    # ── Per-Symbol Breakdown ─────────────────────────────────────────────────
    print(f"\n\n  PER-SYMBOL PERFORMANCE\n")
    sym_rows = []
    sym_grp  = trades_df.groupby("symbol")
    sym_summary = (sym_grp.agg(
        Trades      = ("pnl", "count"),
        Wins        = ("pnl", lambda x: (x > 0).sum()),
        Total_PnL   = ("pnl", "sum"),
        Avg_Ret     = ("return_pct", "mean"),
        Best        = ("return_pct", "max"),
        Worst       = ("return_pct", "min"),
        Avg_Hold    = ("hold_days", "mean"),
    ).reset_index().sort_values("Total_PnL", ascending=False))

    for _, r in sym_summary.iterrows():
        wr = r["Wins"] / r["Trades"] * 100
        sym_rows.append([
            r["symbol"],
            int(r["Trades"]),
            int(r["Wins"]),
            f"{wr:.0f}%",
            f"${r['Total_PnL']:+,.0f}",
            f"{r['Avg_Ret']:+.1f}%",
            f"{r['Best']:+.1f}%",
            f"{r['Worst']:+.1f}%",
            f"{r['Avg_Hold']:.0f}d",
        ])
    print(tabulate(
        sym_rows,
        headers=["Symbol", "Trades", "Wins", "Win%",
                 "Total P&L", "Avg Ret", "Best", "Worst", "Avg Hold"],
        tablefmt="rounded_outline",
        showindex=False,
    ))

    # ── Exit Reason Breakdown ─────────────────────────────────────────────────
    print(f"\n\n  EXIT REASON BREAKDOWN\n")
    exit_grp = (trades_df.groupby("exit_reason")
                .agg(Count     = ("pnl", "count"),
                     Total_PnL = ("pnl", "sum"),
                     Avg_PnL   = ("pnl", "mean"),
                     Win_Rate  = ("pnl", lambda x: f"{(x>0).mean()*100:.1f}%"))
                .reset_index()
                .sort_values("Total_PnL", ascending=False))
    print(tabulate(
        exit_grp,
        headers=["Exit Reason", "Count", "Total P&L ($)", "Avg P&L ($)", "Win Rate"],
        tablefmt="rounded_outline",
        showindex=False,
        floatfmt=".2f",
    ))

    # ── Monthly P&L ───────────────────────────────────────────────────────────
    print(f"\n\n  MONTHLY P&L (all symbols combined)\n")
    trades_df["month"] = pd.to_datetime(trades_df["exit_date"]).dt.to_period("M")
    monthly = (trades_df.groupby("month")
               .agg(Trades = ("pnl", "count"),
                    Wins   = ("pnl", lambda x: (x > 0).sum()),
                    PnL    = ("pnl", "sum"))
               .reset_index())
    monthly["Win%"]  = (monthly["Wins"] / monthly["Trades"] * 100).round(1).astype(str) + "%"
    monthly["PnL$"]  = monthly["PnL"].apply(lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}")
    monthly["month"] = monthly["month"].astype(str)
    print(tabulate(
        monthly[["month", "Trades", "Wins", "Win%", "PnL$"]],
        headers=["Month", "Trades", "Wins", "Win%", "P&L"],
        tablefmt="rounded_outline",
        showindex=False,
    ))

    # ── Monthly Equity Snapshots ──────────────────────────────────────────────
    print(f"\n\n  MONTHLY EQUITY CURVE\n")
    equity_dt        = equity.copy()
    equity_dt.index  = pd.to_datetime(equity_dt.index)
    eq_monthly       = equity_dt.resample("ME").last()
    eq_rows = []
    prev_val = INITIAL_CAPITAL
    for month_end, val in eq_monthly.items():
        month_ret = (val / prev_val - 1) * 100
        total_ret = (val / INITIAL_CAPITAL - 1) * 100
        flag = " ▲" if month_ret > 0 else (" ▼" if month_ret < 0 else "")
        eq_rows.append([
            str(month_end)[:7],
            f"${val:>11,.0f}",
            f"{month_ret:+.2f}%{flag}",
            f"{total_ret:+.2f}%",
        ])
        prev_val = val
    print(tabulate(
        eq_rows,
        headers=["Month", "Portfolio $", "Month Return", "Cumulative"],
        tablefmt="rounded_outline",
        showindex=False,
    ))

    # ── Strategy vs Buy-and-Hold SPY ─────────────────────────────────────────
    trading_days  = len(equity)
    equity_dt2    = equity.copy()
    equity_dt2.index = pd.to_datetime(equity_dt2.index)
    strat_m       = compute_metrics(equity_dt2, INITIAL_CAPITAL, trading_days)
    bh_m          = compute_buyhold_metrics(spy_df, start, end, INITIAL_CAPITAL)

    final_val = equity.iloc[-1]
    bh_final  = INITIAL_CAPITAL * (1 + bh_m.get("total_return", 0))

    print(f"\n\n  STRATEGY vs BUY-AND-HOLD SPY ({start.year})\n")
    comparison = [
        ["Starting Capital",
         f"${INITIAL_CAPITAL:>10,.0f}", f"${INITIAL_CAPITAL:>10,.0f}"],
        ["Final Value",
         f"${final_val:>10,.0f}", f"${bh_final:>10,.0f}"],
        ["Total Return",
         f"{strat_m['total_return']*100:>+.2f}%",
         f"{bh_m.get('total_return',0)*100:>+.2f}%"],
        ["Annualized Return",
         f"{strat_m['ann_return']*100:>+.2f}%",
         f"{bh_m.get('ann_return',0)*100:>+.2f}%"],
        ["Max Drawdown",
         f"{strat_m['max_drawdown']*100:>.2f}%",
         f"{bh_m.get('max_drawdown',0)*100:>.2f}%"],
        ["Sharpe Ratio",
         f"{strat_m['sharpe']:>.2f}",
         f"{bh_m.get('sharpe',0):>.2f}"],
        ["Win Rate",            f"{win_rate:.1f}%",    "N/A"],
        ["Total Trades",        f"{n_total}",          "1 (hold all year)"],
        ["Avg R/R Ratio",       f"{rr_ratio:.2f}",     "N/A"],
        ["Avg Holding Period",  f"{avg_hold:.1f}d",    f"{trading_days}d"],
        ["Total P&L",
         f"${total_pnl:>+,.0f}",
         f"${INITIAL_CAPITAL*bh_m.get('total_return',0):>+,.0f}"],
    ]
    beat_bh = strat_m["total_return"] > bh_m.get("total_return", 0)
    print(tabulate(
        comparison,
        headers=["Metric", "Our Strategy", "Buy & Hold SPY"],
        tablefmt="rounded_outline",
        showindex=False,
    ))
    verdict = "BEAT" if beat_bh else "UNDERPERFORMED"
    margin  = (strat_m["total_return"] - bh_m.get("total_return", 0)) * 100
    print(f"\n  Verdict: Strategy {verdict} Buy-and-Hold by {margin:+.2f}%")

    # ── Overall Summary ───────────────────────────────────────────────────────
    print(f"\n\n  OVERALL STRATEGY SUMMARY\n")
    summary = [
        ["Total Trades",            n_total],
        ["Wins / Losses",           f"{n_wins} / {n_loss}"],
        ["Win Rate",                f"{win_rate:.1f}%"],
        ["Avg Win ($)",             f"${avg_win:,.0f}"],
        ["Avg Loss ($)",            f"${avg_loss:,.0f}"],
        ["Reward/Risk Ratio",       f"{rr_ratio:.2f}"],
        ["Total P&L",               f"${total_pnl:+,.0f}"],
        ["Total Return",            f"{strat_m['total_return']*100:+.2f}%"],
        ["Max Drawdown",            f"{strat_m['max_drawdown']*100:.2f}%"],
        ["Sharpe Ratio",            f"{strat_m['sharpe']:.2f}"],
        ["Avg Holding Period",      f"{avg_hold:.1f} days"],
        ["Partial Exits (2:1)",     (trades_df["partial"] == True).sum()],
        ["Gross Capital Deployed",  f"${total_deployed:,.0f}  ({capital_turnover:.1f}x turnover)"],
        ["Avg Daily Utilization",   f"{avg_utilization:.1f}% of portfolio in market"],
        ["Days in Market",          f"{days_in_market} / {total_days} trading days ({days_in_market/total_days*100:.0f}%)"],
        ["Watchlist Size",          f"{len(WATCHLIST)} symbols"],
    ]
    print(tabulate(summary, headers=["Metric", "Value"],
                   tablefmt="rounded_outline"))

    print(f"\n  To compare other years, run:")
    print(f"    python swing_backtest.py --year 2023")
    print(f"    python swing_backtest.py --year 2024")
    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Swing Strategy Backtester — 30-Stock Watchlist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python swing_backtest.py --year 2025
  python swing_backtest.py --start 2025-01-01 --end 2025-06-30
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year",  type=int, help="Full year (e.g. 2025)")
    group.add_argument("--start", type=str, help="Start date YYYY-MM-DD (use with --end)")
    parser.add_argument("--end",        type=str, default="")
    parser.add_argument("--api-key",    type=str, default="")
    parser.add_argument("--secret-key", type=str, default="")
    args = parser.parse_args()
    if args.start and not args.end:
        parser.error("--start requires --end")
    return args


def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        print(f"  Invalid date: '{s}' — use YYYY-MM-DD")
        sys.exit(1)


def main():
    args       = parse_args()
    api_key    = args.api_key    or os.getenv("ALPACA_API_KEY", "")
    secret_key = args.secret_key or os.getenv("ALPACA_SECRET_KEY", "")

    if args.year:
        start = date(args.year, 1, 1)
        end   = date(args.year, 12, 31)
    else:
        start = parse_date(args.start)
        end   = parse_date(args.end)

    if start > end:
        print("  Start must be before end.")
        sys.exit(1)

    print(f"\n  Fetching {len(WATCHLIST)} symbols ({start} → {end}) ...")
    print(f"  {'Symbol':<8} {'Bars':>6}")
    print(f"  {'──────':<8} {'────':>6}")

    symbol_dfs = {}
    spy_df     = None

    for sym in WATCHLIST:
        df = fetch_daily_data(sym, start, end, api_key, secret_key)
        if df is not None and not df.empty:
            symbol_dfs[sym] = df
            bt_bars = len(df[df.index >= start])
            print(f"  {sym:<8} {bt_bars:>6} bars")
            if sym == "SPY":
                spy_df = df
        else:
            print(f"  {sym:<8}    n/a  (skipped)")

    if not symbol_dfs:
        print("\n  No data returned. Check API keys in .env file.")
        sys.exit(1)

    # If SPY wasn't in the successful fetches (unlikely), fetch separately
    if spy_df is None:
        print("  Fetching SPY separately for benchmark ...")
        spy_df = fetch_daily_data("SPY", start, end, api_key, secret_key)

    print(f"\n  Loaded {len(symbol_dfs)}/{len(WATCHLIST)} symbols. Running backtest ...\n")

    trades, equity, filter_stats = run_backtest(symbol_dfs, start, end)

    display_results(trades, equity, spy_df, start, end, filter_stats)


if __name__ == "__main__":
    main()
