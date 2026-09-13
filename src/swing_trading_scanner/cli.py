"""Discover and run the project's scanner and backtest modules."""

import argparse
import runpy
import sys

from .config import load_environment

COMMANDS = {
    "ema": ("scanners.ema", "Daily EMA crossover scanner (Alpaca)"),
    "confluence": ("scanners.confluence", "Multi-indicator signal scanner (Alpaca)"),
    "pullback": ("scanners.pullback", "9/21 EMA pullback scanner (Yahoo Finance)"),
    "sector": ("scanners.sector", "Premarket sector scanner (Alpaca)"),
    "sector-open": (
        "scanners.sector_open",
        "Sector scanner with opening-window detection (Alpaca)",
    ),
    "sector-snapshot": (
        "scanners.sector_snapshot",
        "Timed sector snapshot (Yahoo Finance)",
    ),
    "sector-history": (
        "scanners.sector_history",
        "Historical sector snapshot (Yahoo Finance)",
    ),
    "options": (
        "options.main",
        "Daily scored signals and underlying-price evaluation (Alpaca)",
    ),
    "backtest-ema": ("backtests.ema", "EMA and premarket-breakout backtest"),
    "backtest-orb": ("backtests.orb", "Opening-range breakout backtest"),
    "backtest-orb-ema": (
        "backtests.orb_ema",
        "Opening-range breakout with EMA filters",
    ),
    "backtest-swing": ("backtests.swing", "Daily swing portfolio backtest"),
    "backtest-tiered": (
        "backtests.tiered",
        "Risk-based sizing and tiered-exit backtest",
    ),
    "backtest-sector": ("backtests.sector", "Sector momentum backtest"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stock scanners and historical strategy research tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        + "\n".join(
            f"  {name:20} {description}" for name, (_, description) in COMMANDS.items()
        )
        + "\n\nRun swing-scanner COMMAND --help for command-specific options.",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    load_environment()
    previous = sys.argv
    sys.argv = [f"swing-scanner {args.command}", *args.arguments]
    try:
        runpy.run_module(
            f"swing_trading_scanner.{COMMANDS[args.command][0]}", run_name="__main__"
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        sys.argv = previous
