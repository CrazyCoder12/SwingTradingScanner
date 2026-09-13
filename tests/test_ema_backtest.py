"""Synthetic scenarios for the current long-only intraday engine."""

import pandas as pd
import pytest

from swing_trading_scanner.backtests.ema import backtest_stock_day, calculate_ema


def candles(closes, fast=None):
    count = len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "close": closes,
            "high": [p + 0.1 for p in closes],
            "low": [p - 0.1 for p in closes],
            "volume": [1000] * count,
            "ema_fast": fast if fast is not None else [99] + [101] * (count - 1),
            "ema_slow": [100] * count,
            "ema34": [98] * count,
            "ema50": [97] * count,
        },
        index=pd.date_range(
            "2025-01-02 10:00", periods=count, freq="10min", tz="America/New_York"
        ),
    )


def test_no_cross_no_trade():
    assert backtest_stock_day("AAA", candles([101, 102, 103], [101, 101, 101])) == []


def test_hard_stop_records_loss():
    trades = backtest_stock_day("AAA", candles([100, 102, 99]))
    assert len(trades) == 1
    assert trades[0]["entry_type"] == "EMA_CROSS"
    assert trades[0]["exit_reason"] == "Hard Stop (-2%)"
    assert trades[0]["return_dollar"] < 0


def test_open_trade_is_closed_at_end_of_data():
    trades = backtest_stock_day("AAA", candles([100, 102, 103]))
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "End of Data"
    assert trades[0]["exit_price"] == 103


def test_lunch_window_blocks_entries():
    df = candles([100, 102, 103])
    df.index = pd.date_range(
        "2025-01-02 12:00", periods=3, freq="10min", tz="America/New_York"
    )
    assert backtest_stock_day("AAA", df) == []


def test_ema_known_values():
    assert calculate_ema(pd.Series([10, 12, 14]), 3).tolist() == pytest.approx(
        [10, 11, 12.5]
    )
