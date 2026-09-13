"""Run the daily watchlist scanner and evaluate each symbol separately."""

import argparse
from datetime import datetime, timedelta

import requests

from ..config import require_credentials
from .backtest import backtest
from .data import fetch_bars
from .indicators import add_indicators
from .market_regime import get_regime
from .scanner import scan
from .watchlist import WATCHLIST


def run(symbols=None, days=365):
    require_credentials()
    print("Running daily signal scanner...")
    regime = get_regime()
    print(
        "Market regime:",
        "unknown" if regime is None else "bullish" if regime else "mixed/bearish",
    )
    end = datetime.now()
    start = end - timedelta(days=days)
    all_signals = []
    total_trades = 0
    weighted_wins = 0.0
    for symbol in WATCHLIST if symbols is None else symbols:
        df = fetch_bars(symbol, start, end)
        if df.empty:
            print(f"{symbol}: no data")
            continue
        df = add_indicators(df)
        signals = scan(df)
        for signal in signals:
            signal["symbol"] = symbol
        all_signals.extend(signals)
        stats = backtest(signals, df)
        total_trades += stats["trades"]
        weighted_wins += stats["win_rate"] * stats["trades"]
        print(
            f"{symbol}: {len(signals)} signals; target-hit rate {stats['win_rate']:.1%}"
        )
    stats = {
        "trades": total_trades,
        "win_rate": weighted_wins / total_trades if total_trades else 0,
    }
    print("Historical signals:", len(all_signals))
    print("Underlying-price evaluation:", stats)
    return all_signals, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", help="Override the built-in watchlist")
    parser.add_argument(
        "--days", type=int, default=365, help="Calendar days of history (default: 365)"
    )
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    try:
        run(args.symbols, args.days)
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
