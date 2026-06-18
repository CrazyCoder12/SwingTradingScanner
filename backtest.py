"""
EMA 5/12/34/50 Day Trading Backtester
======================================
Entry 1 (EMA Cross): 5/12 EMA above 34/50 EMA + 5 EMA crosses 12 EMA, confirmed on next 10-min candle
Entry 2 (PM Break):  5/12 EMA above 34/50 EMA + price breaks pre-market high after 9:40 AM + price near 5 EMA
Stop Loss: 2% hard stop OR price closes below 12 EMA
Data Source: Alpaca Markets API (free tier - historical data, includes pre-market)

Usage:
    Single day:   python backtest.py --date 2024-01-15
    Date range:   python backtest.py --start 2024-01-01 --end 2024-01-31
    With API keys: python backtest.py --date 2024-01-15 --api-key YOUR_KEY --secret-key YOUR_SECRET
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

load_dotenv()  # reads your .env file automatically

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD",  "META",
    "GOOGL","AMZN", "TSLA", "NFLX", "CRM",
    "ORCL", "INTC", "QCOM", "MU",   "AVGO",
    "JPM",  "BAC",  "GS",   "MS",   "V",
    "SPY",  "QQQ",  "IWM",  "XLK",  "XLF",
    "COIN", "HOOD", "SQ",   "PYPL", "SHOP"
]

CAPITAL_PER_TRADE   = 1000.00   # USD per trade
EMA_FAST            = 5
EMA_SLOW            = 12
EMA_MED             = 34
EMA_LONG            = 50
HARD_STOP_PCT       = 0.02      # 2% hard stop loss
BREAKEVEN_TRIGGER   = 0.02      # 2% profit triggers breakeven protection
NEAR_EMA_PCT        = 0.015     # Entry 2: price within 1.5% above 5 EMA
MARKET_OPEN         = "09:30"
FORCE_CLOSE_TIME    = "15:55"
MAX_TRADES_PER_STOCK = 3        # Entry 1 + Entry 2 (x2: once before 3 PM, once after)
INTERVAL            = "10Min"   # 10-minute candles
ENTRY_EARLIEST      = "09:50"   # Skip the first open candle (9:40 whipsaw)
NO_ENTRY_START      = "11:30"   # Lunch chop zone — no new entries
NO_ENTRY_END        = "14:00"   # Resume entries after this time

# ─────────────────────────────────────────────
# DATA FETCHER (Alpaca)
# ─────────────────────────────────────────────

def fetch_alpaca_data(symbol: str, start: date, end: date,
                      api_key: str, secret_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch 10-minute OHLCV bars from Alpaca for a given symbol and date range.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC, then converted to ET)
    """
    try:
        import requests

        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

        # Alpaca needs RFC3339 format
        start_str = f"{start.isoformat()}T06:00:00-04:00"  # include pre-market (6 AM ET)
        end_str   = f"{end.isoformat()}T16:00:00-04:00"

        params = {
            "symbols":    symbol,
            "timeframe":  "10Min",
            "start":      start_str,
            "end":        end_str,
            "feed":       "iex",        # free tier feed
            "limit":      10000,
        }

        all_bars = []
        next_token = None

        while True:
            if next_token:
                params["page_token"] = next_token

            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            bars = data.get("bars", {}).get(symbol, [])
            all_bars.extend(bars)

            next_token = data.get("next_page_token")
            if not next_token:
                break

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df = df.set_index("t")
        df.index = df.index.tz_convert("America/New_York")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                 "c": "close", "v": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return df

    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return None


# ─────────────────────────────────────────────
# EMA CALCULATION
# ─────────────────────────────────────────────

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────
# SINGLE STOCK BACKTEST FOR ONE DAY
# ─────────────────────────────────────────────

def backtest_stock_day(symbol: str, day_df: pd.DataFrame) -> list:
    """
    Run backtest for a single stock on a single trading day.

    Entry 1 — EMA crossover (10-min charge):
      • 5/12 EMA stack is above 34/50 EMA stack (trend filter)
      • 5 EMA crosses above 12 EMA
      • Wait one 10-min candle to confirm the cross holds, then enter

    Entry 2 — Pre-market high breakout:
      • 5/12 EMA stack is above 34/50 EMA stack (trend filter)
      • After 9:40 AM, price closes above pre-market high
      • Price is within NEAR_EMA_PCT of the 5 EMA (not overextended)

    Stop: 2% hard stop OR price closes below 12 EMA
    Returns list of trade dicts.
    """
    trades = []

    # ── Pre-market high/low (6:00–9:29 AM) ───────────────────
    premarket_df = day_df.between_time("06:00", "09:29")
    if not premarket_df.empty:
        premarket_high = premarket_df["high"].max()
        premarket_low  = premarket_df["low"].min()
    else:
        premarket_high = None
        premarket_low  = None

    # ── Filter to regular market hours for trading ────────────
    day_df = day_df.between_time(MARKET_OPEN, "15:50").copy()

    if len(day_df) < 3:
        return trades

    # ── State ─────────────────────────────────────────────────
    position       = None
    trades_today   = 0
    pending_long   = False   # Entry 1: cross detected, waiting 10-min confirmation
    pm_break_count = 0       # Entry 2: 0=not taken, 1=taken once, 2=taken twice (max)

    for i in range(1, len(day_df)):
        row         = day_df.iloc[i]
        prev_row    = day_df.iloc[i - 1]
        candle_time = day_df.index[i]
        time_str    = candle_time.strftime("%H:%M")

        close      = row["close"]
        ema5       = row["ema_fast"]
        ema12      = row["ema_slow"]
        ema34      = row["ema34"]
        ema50      = row["ema50"]
        prev_ema5  = prev_row["ema_fast"]
        prev_ema12 = prev_row["ema_slow"]

        # Trend filter: 5 and 12 EMA both above 34 and 50 EMA
        trend_bullish = (ema5 > ema34) and (ema12 > ema34) and (ema5 > ema50) and (ema12 > ema50)

        # ── Force close at 15:50 ─────────────────────────────
        if position is not None and time_str >= "15:50":
            entry      = position["entry_price"]
            pnl_pct    = (close - entry) / entry
            pnl_dollar = pnl_pct * CAPITAL_PER_TRADE
            trades.append({
                "symbol":        symbol,
                "direction":     "LONG",
                "entry_type":    position["entry_type"],
                "entry_time":    position["entry_time"].strftime("%H:%M"),
                "exit_time":     time_str,
                "entry_price":   round(entry, 2),
                "exit_price":    round(close, 2),
                "exit_reason":   "End of Day (3:55 PM)",
                "return_pct":    round(pnl_pct * 100, 2),
                "return_dollar": round(pnl_dollar, 2),
                "trade_num":     position["trade_num"],
            })
            position = None
            break

        # ── Manage open position ──────────────────────────────
        if position is not None:
            entry   = position["entry_price"]
            pnl_pct = (close - entry) / entry

            if pnl_pct >= BREAKEVEN_TRIGGER:
                position["breakeven_active"] = True

            exit_reason = None
            if pnl_pct <= -HARD_STOP_PCT:
                exit_reason = "Hard Stop (-2%)"
            elif close < ema12:
                if position["breakeven_active"] and pnl_pct < 0:
                    exit_reason = "Breakeven Protection (Below 12 EMA)"
                else:
                    exit_reason = "Price Closed Below 12 EMA"

            if exit_reason:
                pnl_dollar = pnl_pct * CAPITAL_PER_TRADE
                trades.append({
                    "symbol":        symbol,
                    "direction":     "LONG",
                    "entry_type":    position["entry_type"],
                    "entry_time":    position["entry_time"].strftime("%H:%M"),
                    "exit_time":     time_str,
                    "entry_price":   round(entry, 2),
                    "exit_price":    round(close, 2),
                    "exit_reason":   exit_reason,
                    "return_pct":    round(pnl_pct * 100, 2),
                    "return_dollar": round(pnl_dollar, 2),
                    "trade_num":     position["trade_num"],
                })
                position = None
            continue

        # ── Track Entry 1 cross signal (detectable before ENTRY_EARLIEST) ──
        cross_up = (prev_ema5 <= prev_ema12) and (ema5 > ema12)
        if cross_up and trend_bullish:
            pending_long = True

        # Cancel pending if cross reversed or trend lost
        if pending_long and (ema5 < ema12 or not trend_bullish):
            pending_long = False

        # ── Skip if outside tradeable window ─────────────────
        if trades_today >= MAX_TRADES_PER_STOCK:
            continue
        if time_str >= FORCE_CLOSE_TIME:
            continue
        if NO_ENTRY_START <= time_str < NO_ENTRY_END:
            continue

        # ── Entry 1: 5/12 cross above 34/50, 10-min charge confirmed ──
        if (time_str >= ENTRY_EARLIEST and pending_long
                and trend_bullish and ema5 > ema12 and close > ema12):
            pending_long = False
            trades_today += 1
            position = {
                "direction":        "LONG",
                "entry_type":       "EMA_CROSS",
                "entry_price":      close,
                "entry_time":       candle_time,
                "breakeven_active": False,
                "trade_num":        trades_today,
            }
            continue

        # ── Entry 2: Pre-market high breakout ────────────────
        # 1st trigger: anytime after 9:40 AM
        # 2nd trigger: only after 3:00 PM, and only if 1st already fired
        pm_conditions = (
            premarket_high is not None
            and pm_break_count < 2
            and trend_bullish
            and close > premarket_high
            and close <= ema5 * (1 + NEAR_EMA_PCT)
        )
        first_trigger  = pm_break_count == 0 and time_str >= "09:40"
        second_trigger = pm_break_count == 1 and time_str >= "15:00"

        if pm_conditions and (first_trigger or second_trigger):
            pm_break_count += 1
            trades_today += 1
            position = {
                "direction":        "LONG",
                "entry_type":       f"PM_BREAK_{pm_break_count}",
                "entry_price":      close,
                "entry_time":       candle_time,
                "breakeven_active": False,
                "trade_num":        trades_today,
            }

    # Safety net: close any position still open at end of data
    if position is not None:
        last_row   = day_df.iloc[-1]
        close      = last_row["close"]
        entry      = position["entry_price"]
        pnl_pct    = (close - entry) / entry
        pnl_dollar = pnl_pct * CAPITAL_PER_TRADE
        trades.append({
            "symbol":        symbol,
            "direction":     "LONG",
            "entry_type":    position["entry_type"],
            "entry_time":    position["entry_time"].strftime("%H:%M"),
            "exit_time":     day_df.index[-1].strftime("%H:%M"),
            "entry_price":   round(entry, 2),
            "exit_price":    round(close, 2),
            "exit_reason":   "End of Data",
            "return_pct":    round(pnl_pct * 100, 2),
            "return_dollar": round(pnl_dollar, 2),
            "trade_num":     position["trade_num"],
        })

    return trades


# ─────────────────────────────────────────────
# MAIN BACKTEST RUNNER
# ─────────────────────────────────────────────

def run_backtest(start_date: date, end_date: date,
                 api_key: str, secret_key: str) -> pd.DataFrame:
    """
    Run full backtest over date range for all stocks in watchlist.
    Returns DataFrame of all trades.
    """
    all_trades = []
    trading_days = get_trading_days(start_date, end_date)

    print(f"\n{'='*60}")
    print(f"  EMA 5/12 Crossover Backtest")
    print(f"  Period : {start_date} → {end_date}")
    print(f"  Days   : {len(trading_days)} trading days")
    print(f"  Stocks : {len(WATCHLIST)}")
    print(f"{'='*60}\n")

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"[{i:02d}/{len(WATCHLIST)}] Fetching {symbol}...", end=" ", flush=True)

        df = fetch_alpaca_data(symbol, start_date, end_date, api_key, secret_key)

        if df is None or df.empty:
            print("No data — skipped")
            continue

        # Calculate all EMAs on full dataset so they warm up across days
        df["ema_fast"] = calculate_ema(df["close"], EMA_FAST)
        df["ema_slow"] = calculate_ema(df["close"], EMA_SLOW)
        df["ema34"]    = calculate_ema(df["close"], EMA_MED)
        df["ema50"]    = calculate_ema(df["close"], EMA_LONG)

        # Run backtest per trading day
        stock_trades = 0
        for day in trading_days:
            day_df = df[df.index.date == day]
            if day_df.empty:
                continue
            day_trades = backtest_stock_day(symbol, day_df)
            for t in day_trades:
                t["date"] = str(day)
            all_trades.extend(day_trades)
            stock_trades += len(day_trades)

        print(f"{stock_trades} trades")

    return pd.DataFrame(all_trades)


def get_trading_days(start: date, end: date) -> list:
    """Return list of weekdays between start and end (approximate trading days)."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            days.append(current)
        current += timedelta(days=1)
    return days


# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────

def display_results(trades_df: pd.DataFrame, start_date: date, end_date: date):
    """Print formatted results tables."""

    if trades_df.empty:
        print("\n⚠️  No trades were generated for this period.")
        print("   This could mean: no EMA crossover signals occurred,")
        print("   or the API returned no data. Check your API keys and dates.\n")
        return

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS — {start_date} to {end_date}")
    print(f"{'='*60}\n")

    # ── Per-Stock Summary ─────────────────────────────────
    stock_summary = trades_df.groupby("symbol").agg(
        Trades      = ("return_dollar", "count"),
        Wins        = ("return_dollar", lambda x: (x > 0).sum()),
        Losses      = ("return_dollar", lambda x: (x <= 0).sum()),
        Total_PnL   = ("return_dollar", "sum"),
        Avg_Return  = ("return_pct", "mean"),
        Best_Trade  = ("return_pct", "max"),
        Worst_Trade = ("return_pct", "min"),
    ).reset_index()

    stock_summary["Win Rate"]  = (stock_summary["Wins"] / stock_summary["Trades"] * 100).round(1).astype(str) + "%"
    stock_summary["Total_PnL"] = stock_summary["Total_PnL"].round(2)
    stock_summary["Avg_Return"]= stock_summary["Avg_Return"].round(2).astype(str) + "%"
    stock_summary["Best_Trade"]= stock_summary["Best_Trade"].round(2).astype(str) + "%"
    stock_summary["Worst_Trade"]= stock_summary["Worst_Trade"].round(2).astype(str) + "%"
    stock_summary = stock_summary.sort_values("Total_PnL", ascending=False)

    print("📊 PER-STOCK PERFORMANCE\n")
    print(tabulate(
        stock_summary[["symbol","Trades","Wins","Losses","Win Rate",
                        "Total_PnL","Avg_Return","Best_Trade","Worst_Trade"]],
        headers=["Symbol","Trades","Wins","Losses","Win Rate",
                 "Total P&L ($)","Avg Return","Best","Worst"],
        tablefmt="rounded_outline",
        showindex=False,
        floatfmt=".2f"
    ))

    # ── Entry Type breakdown ──────────────────────────────
    if "entry_type" in trades_df.columns:
        print(f"\n\n📊 ENTRY TYPE BREAKDOWN\n")
        for etype, label in [("EMA_CROSS", "EMA Cross (10-min charge)"), ("PM_BREAK_1", "Pre-Market Breakout (1st)"), ("PM_BREAK_2", "Pre-Market Breakout (2nd, after 3 PM)")]:
            sub = trades_df[trades_df["entry_type"] == etype]
            if sub.empty:
                continue
            wins = (sub["return_dollar"] > 0).sum()
            pnl  = sub["return_dollar"].sum()
            print(f"  {label:35s}  Trades: {len(sub):4d}  Wins: {wins:4d} ({wins/len(sub)*100:.1f}%)  Total P&L: {'+'if pnl>=0 else ''}${pnl:.2f}")

    # ── All Trades Detail ─────────────────────────────────
    print(f"\n\n📋 ALL TRADES DETAIL\n")
    cols = ["date","symbol"]
    if "entry_type" in trades_df.columns:
        cols.append("entry_type")
    cols += ["entry_time","exit_time","entry_price","exit_price","return_pct","return_dollar","exit_reason"]
    detail = trades_df[cols].copy()
    detail["return_pct"]    = detail["return_pct"].astype(str) + "%"
    detail["return_dollar"] = detail["return_dollar"].apply(
        lambda x: f"+${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
    )

    headers = ["Date","Symbol"]
    if "entry_type" in trades_df.columns:
        headers.append("Entry Type")
    headers += ["Entry","Exit","Entry $","Exit $","Return %","P&L","Exit Reason"]

    print(tabulate(
        detail,
        headers=headers,
        tablefmt="rounded_outline",
        showindex=False
    ))

    # ── Exit Reason Breakdown ─────────────────────────────
    print(f"\n\n🚪 EXIT REASON BREAKDOWN\n")
    exit_summary = trades_df.groupby("exit_reason").agg(
        Count     = ("return_dollar", "count"),
        Total_PnL = ("return_dollar", "sum"),
        Avg_PnL   = ("return_dollar", "mean"),
        Win_Rate  = ("return_dollar", lambda x: f"{(x>0).mean()*100:.1f}%")
    ).reset_index().sort_values("Count", ascending=False)

    print(tabulate(
        exit_summary,
        headers=["Exit Reason","Count","Total P&L ($)","Avg P&L ($)","Win Rate"],
        tablefmt="rounded_outline",
        showindex=False,
        floatfmt=".2f"
    ))

    # ── Overall Summary ───────────────────────────────────
    print(f"\n\n📈 OVERALL SUMMARY\n")
    total_trades   = len(trades_df)
    winning_trades = (trades_df["return_dollar"] > 0).sum()
    losing_trades  = (trades_df["return_dollar"] <= 0).sum()
    win_rate       = winning_trades / total_trades * 100
    total_pnl      = trades_df["return_dollar"].sum()
    avg_win        = trades_df[trades_df["return_dollar"] > 0]["return_dollar"].mean() if winning_trades > 0 else 0
    avg_loss       = trades_df[trades_df["return_dollar"] <= 0]["return_dollar"].mean() if losing_trades > 0 else 0
    rr_ratio       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Max drawdown (cumulative PnL drawdown)
    cum_pnl        = trades_df["return_dollar"].cumsum()
    rolling_max    = cum_pnl.cummax()
    drawdown       = cum_pnl - rolling_max
    max_drawdown   = drawdown.min()

    summary_data = [
        ["Total Trades",          total_trades],
        ["Winning Trades",        f"{winning_trades} ({win_rate:.1f}%)"],
        ["Losing Trades",         f"{losing_trades} ({100-win_rate:.1f}%)"],
        ["Total P&L",             f"${total_pnl:+.2f}"],
        ["Average Win",           f"${avg_win:.2f}"],
        ["Average Loss",          f"${avg_loss:.2f}"],
        ["Risk/Reward Ratio",     f"{rr_ratio:.2f}"],
        ["Max Drawdown",          f"${max_drawdown:.2f}"],
        ["Capital Per Trade",     f"${CAPITAL_PER_TRADE:.0f}"],
        ["Max Theoretical Exposure", f"${CAPITAL_PER_TRADE * len(WATCHLIST):,.0f}"],
    ]

    print(tabulate(summary_data, headers=["Metric","Value"],
                   tablefmt="rounded_outline"))

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="EMA 5/12 Day Trading Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py --date 2024-03-15 --api-key ABC --secret-key XYZ
  python backtest.py --start 2024-01-01 --end 2024-01-31 --api-key ABC --secret-key XYZ
        """
    )

    # Date arguments
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date", type=str,
        help="Single trading date (YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--start", type=str,
        help="Start date for range (YYYY-MM-DD) — use with --end"
    )
    parser.add_argument(
        "--end", type=str,
        help="End date for range (YYYY-MM-DD) — use with --start"
    )

    # API keys
    parser.add_argument("--api-key",    type=str, default="",
                        help="Alpaca API Key ID (get free at alpaca.markets)")
    parser.add_argument("--secret-key", type=str, default="",
                        help="Alpaca Secret Key")

    args = parser.parse_args()

    # Validate date range
    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")

    return args


def parse_date(date_str: str) -> date:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"❌ Invalid date format: '{date_str}' — use YYYY-MM-DD")
        sys.exit(1)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    if args.date:
        start_date = end_date = parse_date(args.date)
    else:
        start_date = parse_date(args.start)
        end_date   = parse_date(args.end)

    if start_date > end_date:
        print("❌ Start date must be before end date")
        sys.exit(1)

    # Warn if no API keys
    api_key = args.api_key or os.getenv("ALPACA_API_KEY", "")
    secret_key = args.secret_key or os.getenv("ALPACA_SECRET_KEY", "")

    # if not args.api_key or not args.secret_key:
    #     print("\n⚠️  WARNING: No API keys provided.")
    #     print("   Get your free keys at: https://app.alpaca.markets/signup")
    #     print("   Run with: --api-key YOUR_KEY --secret-key YOUR_SECRET\n")
    #     sys.exit(1)

    trades_df = run_backtest(start_date, end_date, api_key, secret_key)
    display_results(trades_df, start_date, end_date)


if __name__ == "__main__":
    main()
