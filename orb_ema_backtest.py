"""
ORB + EMA Combined Day Trading Backtester
==========================================
Strategy:
  Opening Range : 9:30 candle only (first 10-min bar)
  EMA stack     : 5 & 12 EMA both ABOVE 34 & 50 EMA → bullish (long bias)
                  5 & 12 EMA both BELOW 34 & 50 EMA → bearish (short bias)
  Long  entry   : candle closes ABOVE ORB high + price > VWAP + RVOL >= 1.5x + bullish EMA stack
  Short entry   : candle closes BELOW ORB low  + price < VWAP + RVOL >= 1.5x + bearish EMA stack
  Stop (ORB)    : price closes below 12 EMA OR below ORB low (long), whichever first
                  price closes above 12 EMA OR above ORB high (short), whichever first
  Entry window  : 9:40 – 11:00
  PM Cross entry: 15:00 – 15:49 — 5 EMA crosses 12 EMA + candle confirms direction
  Stop (PM)     : price closes below 12 EMA (long) / above 12 EMA (short)
  Max trades    : 1 per stock per day
  Force close   : 15:50 candle

Usage:
    python orb_ema_backtest.py --start 2026-01-01 --end 2026-03-26
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

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD",  "META",
    "GOOGL","AMZN", "TSLA", "NFLX",
    "ORCL", "MU",
    "COIN"
]

CAPITAL_PER_TRADE  = 2500.00
ORB_END_TIME       = "09:40"   # First candle eligible for entry (after 10-min ORB)
ENTRY_CUTOFF       = "11:00"   # ORB signal expires — no entries after this
PM_CROSS_START     = "15:00"   # After 3 PM: 5/12 EMA cross entries allowed
FORCE_CLOSE_TIME   = "15:50"   # Force close all positions
RVOL_MIN           = 2.0       # Breakout candle must have >= 2.0x avg volume
RVOL_LOOKBACK      = 20        # Days to average for RVOL baseline (per time slot)
MIN_ORB_RANGE_PCT  = 0.003     # Minimum ORB range (0.3%) — skip flat/holiday opens
BASELINE_DAYS      = 45        # Calendar days of extra data fetched for EMA/RVOL warmup

EMA_FAST   = 5
EMA_MID    = 12
EMA_SLOW1  = 34
EMA_SLOW2  = 50


# ─────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────

def fetch_alpaca_data(symbol: str, start: date, end: date,
                      api_key: str, secret_key: str) -> Optional[pd.DataFrame]:
    """Fetch 10-min OHLCV bars from Alpaca. Returns ET-indexed DataFrame."""
    try:
        import requests
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        headers  = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        start_str = f"{start.isoformat()}T09:30:00-04:00"
        end_str   = f"{end.isoformat()}T16:00:00-04:00"
        params = {
            "symbols":   symbol,
            "timeframe": "10Min",
            "start":     start_str,
            "end":       end_str,
            "feed":      "iex",
            "limit":     10000,
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
# INDICATORS
# ─────────────────────────────────────────────

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday VWAP, reset each trading day."""
    df = df.copy()
    typical    = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"] = typical * df["volume"]
    df["_day"] = df.index.date

    vwap_list = []
    for _, group in df.groupby("_day"):
        vwap_day = group["_tpv"].cumsum() / group["volume"].cumsum()
        vwap_list.append(vwap_day)

    df["vwap"] = pd.concat(vwap_list)
    df.drop(columns=["_tpv", "_day"], inplace=True)
    return df


def add_rvol(df: pd.DataFrame) -> pd.DataFrame:
    """RVOL per time-of-day slot vs rolling 20-day average. No lookahead."""
    df = df.copy()
    df["_ts"] = df.index.strftime("%H:%M")
    df["rvol"] = df.groupby("_ts")["volume"].transform(
        lambda x: x / x.shift(1).rolling(RVOL_LOOKBACK, min_periods=5).mean()
    )
    df.drop(columns=["_ts"], inplace=True)
    return df


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA 5, 12, 34, 50 across the full dataset (continuous, not reset daily)."""
    df = df.copy()
    for period in [EMA_FAST, EMA_MID, EMA_SLOW1, EMA_SLOW2]:
        df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


# ─────────────────────────────────────────────
# SINGLE STOCK BACKTEST FOR ONE DAY
# ─────────────────────────────────────────────

def backtest_day(symbol: str, day_df: pd.DataFrame) -> list:
    """
    Run ORB + EMA combined backtest for a single stock on a single trading day.
    Returns list of trade dicts.
    """
    trades = []
    day_df = day_df.between_time("09:30", "15:50").copy()

    if len(day_df) < 4:
        return trades

    # ── Build Opening Range (9:30 candle only) ─────────────
    orb_candles = day_df.between_time("09:30", "09:39")
    if len(orb_candles) < 1:
        return trades

    orb_high  = orb_candles["high"].max()
    orb_low   = orb_candles["low"].min()
    orb_range = orb_high - orb_low
    orb_mid   = (orb_high + orb_low) / 2

    if orb_range / orb_candles["close"].iloc[-1] < MIN_ORB_RANGE_PCT:
        return trades

    # ── Post-ORB candles ───────────────────────────────────
    trade_df = day_df[day_df.index.strftime("%H:%M") >= ORB_END_TIME]
    if trade_df.empty:
        return trades

    position     = None
    traded_today = False   # ORB window trade taken
    pm_traded    = False   # PM cross trade taken
    prev_ema5    = np.nan
    prev_ema12   = np.nan

    for candle_time, row in trade_df.iterrows():
        time_str  = candle_time.strftime("%H:%M")
        cur_ema5  = row.get("ema5",  np.nan)
        cur_ema12 = row.get("ema12", np.nan)

        # ── Force close ────────────────────────────────────
        if time_str >= FORCE_CLOSE_TIME:
            if position is not None:
                close     = row["close"]
                entry     = position["entry_price"]
                direction = position["direction"]
                pnl_pct   = ((close - entry) / entry if direction == "LONG"
                             else (entry - close) / entry)
                trades.append(_make_trade(
                    symbol, direction, position["entry_time"], candle_time,
                    entry, close, orb_range, orb_mid, "Force Close", pnl_pct
                ))
                position = None
            break

        # ── Manage open position ───────────────────────────
        if position is not None:
            direction  = position["direction"]
            entry      = position["entry_price"]
            trade_type = position["trade_type"]
            close      = row["close"]

            exit_price  = None
            exit_reason = None

            if direction == "LONG":
                # ORB trades also use ORB low as hard stop
                if trade_type == "ORB" and close < orb_low:
                    exit_price, exit_reason = close, "ORB Stop"
                elif not pd.isna(cur_ema12) and close < cur_ema12:
                    exit_price, exit_reason = close, "EMA Stop"
            else:  # SHORT
                if trade_type == "ORB" and close > orb_high:
                    exit_price, exit_reason = close, "ORB Stop"
                elif not pd.isna(cur_ema12) and close > cur_ema12:
                    exit_price, exit_reason = close, "EMA Stop"

            if exit_reason:
                pnl_pct = ((exit_price - entry) / entry if direction == "LONG"
                           else (entry - exit_price) / entry)
                trades.append(_make_trade(
                    symbol, direction, position["entry_time"], candle_time,
                    entry, exit_price, orb_range, orb_mid, exit_reason, pnl_pct
                ))
                position = None

            prev_ema5, prev_ema12 = cur_ema5, cur_ema12
            continue

        # ── ORB entry (9:40 – 11:00) ──────────────────────
        if not traded_today and ORB_END_TIME <= time_str <= ENTRY_CUTOFF:
            rvol = row.get("rvol", np.nan)
            if not (pd.isna(rvol) or rvol < RVOL_MIN):
                close = row["close"]
                vwap  = row.get("vwap", close)
                ema5  = cur_ema5
                ema12 = cur_ema12
                ema34 = row.get("ema34", np.nan)
                ema50 = row.get("ema50", np.nan)

                if not any(pd.isna(v) for v in [ema5, ema12, ema34, ema50]):
                    bullish_stack = ema5 > ema34 and ema5 > ema50 and ema12 > ema34 and ema12 > ema50
                    bearish_stack = ema5 < ema34 and ema5 < ema50 and ema12 < ema34 and ema12 < ema50

                    if close > orb_high and close > vwap and bullish_stack:
                        traded_today = True
                        position = {
                            "direction":   "LONG",
                            "entry_price": close,
                            "entry_time":  candle_time,
                            "trade_type":  "ORB",
                        }
                    elif close < orb_low and close < vwap and bearish_stack:
                        traded_today = True
                        position = {
                            "direction":   "SHORT",
                            "entry_price": close,
                            "entry_time":  candle_time,
                            "trade_type":  "ORB",
                        }

        # ── PM EMA cross entry (15:00 – 15:49) ────────────
        elif not pm_traded and time_str >= PM_CROSS_START:
            if not any(pd.isna(v) for v in [prev_ema5, prev_ema12, cur_ema5, cur_ema12]):
                close = row["close"]
                bullish_cross = (prev_ema5 <= prev_ema12 and cur_ema5 > cur_ema12
                                 and close > cur_ema12)
                bearish_cross = (prev_ema5 >= prev_ema12 and cur_ema5 < cur_ema12
                                 and close < cur_ema12)

                if bullish_cross:
                    pm_traded = True
                    position = {
                        "direction":   "LONG",
                        "entry_price": close,
                        "entry_time":  candle_time,
                        "trade_type":  "PM_CROSS",
                    }
                elif bearish_cross:
                    pm_traded = True
                    position = {
                        "direction":   "SHORT",
                        "entry_price": close,
                        "entry_time":  candle_time,
                        "trade_type":  "PM_CROSS",
                    }

        prev_ema5, prev_ema12 = cur_ema5, cur_ema12

    # Safety net
    if position is not None:
        last      = day_df.iloc[-1]
        close     = last["close"]
        entry     = position["entry_price"]
        direction = position["direction"]
        pnl_pct   = ((close - entry) / entry if direction == "LONG"
                     else (entry - close) / entry)
        trades.append(_make_trade(
            symbol, direction, position["entry_time"], day_df.index[-1],
            entry, close, orb_range, orb_mid, "End of Data", pnl_pct
        ))

    return trades


def _make_trade(symbol, direction, entry_time, exit_time,
                entry_price, exit_price, orb_range, orb_mid, exit_reason, pnl_pct):
    return {
        "symbol":        symbol,
        "direction":     direction,
        "entry_time":    entry_time.strftime("%H:%M"),
        "exit_time":     exit_time.strftime("%H:%M"),
        "entry_price":   round(entry_price, 2),
        "exit_price":    round(exit_price, 2),
        "orb_range_pct": round(orb_range / exit_price * 100, 3),
        "exit_reason":   exit_reason,
        "return_pct":    round(pnl_pct * 100, 2),
        "return_dollar": round(pnl_pct * CAPITAL_PER_TRADE, 2),
    }


# ─────────────────────────────────────────────
# MAIN BACKTEST RUNNER
# ─────────────────────────────────────────────

def get_trading_days(start: date, end: date) -> list:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def run_backtest(start_date: date, end_date: date,
                 api_key: str, secret_key: str) -> pd.DataFrame:
    all_trades   = []
    trading_days = get_trading_days(start_date, end_date)
    fetch_start  = start_date - timedelta(days=BASELINE_DAYS)

    print(f"\n{'='*64}")
    print(f"  ORB + EMA Stack Day Trading Backtest")
    print(f"  Period  : {start_date} → {end_date}")
    print(f"  Days    : {len(trading_days)} trading days")
    print(f"  Stocks  : {len(WATCHLIST)}")
    print(f"  Entry   : ORB breakout + VWAP + RVOL≥{RVOL_MIN}x + EMA {EMA_FAST}/{EMA_MID} vs {EMA_SLOW1}/{EMA_SLOW2}")
    print(f"  ORB window : {ORB_END_TIME}–{ENTRY_CUTOFF}  |  PM cross: {PM_CROSS_START}–{FORCE_CLOSE_TIME}")
    print(f"  Stop       : 12 EMA or ORB low/high (ORB trades) | 12 EMA (PM trades)")
    print(f"{'='*64}\n")

    for i, symbol in enumerate(WATCHLIST, 1):
        print(f"[{i:02d}/{len(WATCHLIST)}] {symbol}...", end=" ", flush=True)

        df = fetch_alpaca_data(symbol, fetch_start, end_date, api_key, secret_key)
        if df is None or df.empty:
            print("no data — skipped")
            continue

        df = add_vwap(df)
        df = add_rvol(df)
        df = add_emas(df)

        stock_trades = 0
        for day in trading_days:
            day_df = df[df.index.date == day]
            if day_df.empty:
                continue
            day_trades = backtest_day(symbol, day_df)
            for t in day_trades:
                t["date"] = str(day)
            all_trades.extend(day_trades)
            stock_trades += len(day_trades)

        print(f"{stock_trades} trades")

    return pd.DataFrame(all_trades)


# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────

def display_results(trades_df: pd.DataFrame, start_date: date, end_date: date):
    if trades_df.empty:
        print("\n⚠️  No trades generated. Check API keys / date range.\n")
        return

    print(f"\n{'='*64}")
    print(f"  ORB + EMA BACKTEST RESULTS — {start_date} to {end_date}")
    print(f"{'='*64}\n")

    # Long vs Short
    print("📊 LONG vs SHORT BREAKDOWN\n")
    for direction in ["LONG", "SHORT"]:
        sub = trades_df[trades_df["direction"] == direction]
        if sub.empty:
            continue
        w   = (sub["return_dollar"] > 0).sum()
        pnl = sub["return_dollar"].sum()
        aw  = sub[sub["return_dollar"] > 0]["return_dollar"].mean() if w > 0 else 0
        al  = sub[sub["return_dollar"] <= 0]["return_dollar"].mean() if (len(sub)-w) > 0 else 0
        print(f"  {direction:5s}  Trades:{len(sub):4d}  Wins:{w:3d}({w/len(sub)*100:.1f}%)  "
              f"P&L:{'+'if pnl>=0 else ''}${pnl:.2f}  AvgWin:+${aw:.2f}  AvgLoss:${al:.2f}")

    # Per-stock
    stock_summary = trades_df.groupby("symbol").agg(
        Trades      = ("return_dollar", "count"),
        Wins        = ("return_dollar", lambda x: (x > 0).sum()),
        Losses      = ("return_dollar", lambda x: (x <= 0).sum()),
        Total_PnL   = ("return_dollar", "sum"),
        Avg_Return  = ("return_pct",    "mean"),
        Best_Trade  = ("return_pct",    "max"),
        Worst_Trade = ("return_pct",    "min"),
    ).reset_index()
    stock_summary["Win Rate"]    = (stock_summary["Wins"] / stock_summary["Trades"] * 100).round(1).astype(str) + "%"
    stock_summary["Total_PnL"]   = stock_summary["Total_PnL"].round(2)
    stock_summary["Avg_Return"]  = stock_summary["Avg_Return"].round(2).astype(str) + "%"
    stock_summary["Best_Trade"]  = stock_summary["Best_Trade"].round(2).astype(str) + "%"
    stock_summary["Worst_Trade"] = stock_summary["Worst_Trade"].round(2).astype(str) + "%"
    stock_summary = stock_summary.sort_values("Total_PnL", ascending=False)

    print("\n\n📊 PER-STOCK PERFORMANCE\n")
    print(tabulate(
        stock_summary[["symbol","Trades","Wins","Losses","Win Rate",
                        "Total_PnL","Avg_Return","Best_Trade","Worst_Trade"]],
        headers=["Symbol","Trades","Wins","Losses","Win Rate",
                 "Total P&L ($)","Avg Return","Best","Worst"],
        tablefmt="rounded_outline", showindex=False, floatfmt=".2f"
    ))

    # Exit reasons
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
        tablefmt="rounded_outline", showindex=False, floatfmt=".2f"
    ))

    # All trades
    print(f"\n\n📋 ALL TRADES DETAIL\n")
    detail = trades_df[["date","symbol","direction","entry_time","exit_time",
                          "entry_price","exit_price","return_pct",
                          "return_dollar","exit_reason"]].copy()
    detail["return_pct"]    = detail["return_pct"].astype(str) + "%"
    detail["return_dollar"] = detail["return_dollar"].apply(
        lambda x: f"+${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
    )
    print(tabulate(
        detail,
        headers=["Date","Symbol","Dir","Entry","Exit","Entry $","Exit $",
                 "Return %","P&L","Exit Reason"],
        tablefmt="rounded_outline", showindex=False
    ))

    # Overall
    print(f"\n\n📈 OVERALL SUMMARY\n")
    n      = len(trades_df)
    wins   = (trades_df["return_dollar"] > 0).sum()
    losses = (trades_df["return_dollar"] <= 0).sum()
    pnl    = trades_df["return_dollar"].sum()
    aw     = trades_df[trades_df["return_dollar"] > 0]["return_dollar"].mean() if wins > 0 else 0
    al     = trades_df[trades_df["return_dollar"] <= 0]["return_dollar"].mean() if losses > 0 else 0
    rr     = abs(aw / al) if al != 0 else float("inf")
    dd     = (trades_df["return_dollar"].cumsum() - trades_df["return_dollar"].cumsum().cummax()).min()
    ema_stops = (trades_df["exit_reason"] == "EMA Stop").sum()
    forced    = (trades_df["exit_reason"] == "Force Close").sum()

    trading_days_count = len(get_trading_days(start_date, end_date))
    rows = [
        ["Total Trades",             n],
        ["Winning Trades",           f"{wins} ({wins/n*100:.1f}%)"],
        ["Losing Trades",            f"{losses} ({losses/n*100:.1f}%)"],
        ["EMA Stop",                 f"{ema_stops} ({ema_stops/n*100:.1f}%)"],
        ["Force Close",              f"{forced} ({forced/n*100:.1f}%)"],
        ["Total P&L",                f"${pnl:+.2f}"],
        ["P&L per Day",              f"${pnl/trading_days_count:+.2f}"],
        ["Average Win",              f"${aw:.2f}"],
        ["Average Loss",             f"${al:.2f}"],
        ["Actual R/R Ratio",         f"{rr:.2f}"],
        ["Max Drawdown",             f"${dd:.2f}"],
        ["Capital Per Trade",        f"${CAPITAL_PER_TRADE:.0f}"],
        ["Max Theoretical Exposure", f"${CAPITAL_PER_TRADE * len(WATCHLIST):,.0f}"],
    ]
    print(tabulate(rows, headers=["Metric","Value"], tablefmt="rounded_outline"))
    print(f"\n{'='*64}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ORB + EMA Stack Combined Day Trading Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python orb_ema_backtest.py --date 2026-01-15\n"
               "  python orb_ema_backtest.py --start 2026-01-01 --end 2026-03-26"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--date",       type=str, help="Single date (YYYY-MM-DD)")
    grp.add_argument("--start",      type=str, help="Start date (YYYY-MM-DD)")
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
        print(f"❌ Invalid date: '{s}'")
        sys.exit(1)


def main():
    args       = parse_args()
    start_date = parse_date(args.date) if args.date else parse_date(args.start)
    end_date   = parse_date(args.date) if args.date else parse_date(args.end)

    if start_date > end_date:
        print("❌ Start must be before end")
        sys.exit(1)

    api_key    = args.api_key    or os.getenv("ALPACA_API_KEY", "")
    secret_key = args.secret_key or os.getenv("ALPACA_SECRET_KEY", "")

    trades_df = run_backtest(start_date, end_date, api_key, secret_key)
    display_results(trades_df, start_date, end_date)


if __name__ == "__main__":
    main()
