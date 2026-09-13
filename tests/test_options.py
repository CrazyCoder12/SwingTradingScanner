from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from swing_trading_scanner.options import data, main, market_regime
from swing_trading_scanner.options.backtest import backtest
from swing_trading_scanner.options.indicators import add_indicators
from swing_trading_scanner.options.scoring import score_signal


def test_score_full_confirmation_and_no_confirmation():
    row = dict(
        ema9=110,
        ema21=105,
        ema50=100,
        low=105,
        rsi=60,
        volume=200,
        vol_avg=100,
        close=110,
    )
    assert score_signal(row, {"close": 109}) == 8
    row.update(ema9=80, ema21=90, low=110, rsi=90, volume=50, close=80)
    assert score_signal(row, {"close": 90}) == 0


def test_indicators_preserve_input_and_have_warmup():
    frame = pd.DataFrame(
        {
            "close": range(100, 160),
            "high": range(101, 161),
            "low": range(99, 159),
            "volume": [1000] * 60,
        }
    )
    result = add_indicators(frame)
    assert "ema9" not in frame
    assert result["atr"].iloc[:13].isna().all()
    assert result["atr"].iloc[-1] == 2
    assert result["ema9"].iloc[-1] > result["ema21"].iloc[-1] > result["ema50"].iloc[-1]
    assert result["rsi"].iloc[-1] == 100


@pytest.mark.parametrize(
    "high,low,expected", [(103, 99, 1), (103, 98, 0), (101, 99, 0)]
)
def test_outcomes_target_stop_priority_and_unresolved(high, low, expected):
    df = pd.DataFrame(
        {"close": [100, 100], "atr": [1, 1], "high": [100, high], "low": [100, low]}
    )
    assert backtest([{"date": 0}], df) == {"trades": 1, "win_rate": expected}


def test_each_symbol_uses_its_own_prices(monkeypatch):
    frames = {
        "AAA": pd.DataFrame(
            {"close": [100, 101], "atr": [1, 1], "high": [100, 103], "low": [100, 99]}
        ),
        "BBB": pd.DataFrame(
            {"close": [200, 199], "atr": [1, 1], "high": [200, 201], "low": [200, 198]}
        ),
        "EMPTY": pd.DataFrame(),
    }
    monkeypatch.setattr(main, "require_credentials", lambda: ("test", "test"))
    monkeypatch.setattr(main, "get_regime", lambda: None)
    monkeypatch.setattr(main, "fetch_bars", lambda symbol, *a: frames[symbol])
    monkeypatch.setattr(main, "add_indicators", lambda df: df)
    monkeypatch.setattr(main, "scan", lambda df: [{"date": 0}])
    signals, stats = main.run(["AAA", "BBB", "EMPTY"])
    assert [s["symbol"] for s in signals] == ["AAA", "BBB"]
    assert stats == {"trades": 2, "win_rate": 0.5}


def test_missing_benchmark_is_unknown(monkeypatch):
    monkeypatch.setattr(market_regime, "fetch_bars", lambda *a: pd.DataFrame())
    assert market_regime.get_regime() is None


def test_paginated_data_is_sorted_and_deduplicated(monkeypatch):
    monkeypatch.setattr(data, "require_credentials", lambda: ("test", "test"))
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    bar = {
        "t": "2025-01-02T05:00:00Z",
        "o": 100,
        "h": 102,
        "l": 99,
        "c": 101,
        "v": 1000,
    }
    responses = [Mock(), Mock()]
    responses[0].json.return_value = {"bars": [bar], "next_page_token": "page-two"}
    responses[1].json.return_value = {"bars": [bar], "next_page_token": None}
    calls = []

    def get(url, **kwargs):
        calls.append(dict(kwargs["params"]))
        assert kwargs["timeout"] == 30
        return responses[len(calls) - 1]

    monkeypatch.setattr(data.requests, "get", get)
    result = data.fetch_bars("AAA", date(2025, 1, 1), date(2025, 1, 3))
    assert len(result) == 1
    assert result.index[0] == date(2025, 1, 2)
    assert calls[1]["page_token"] == "page-two"
    assert calls[0]["feed"] == "iex"
    for response in responses:
        response.raise_for_status.assert_called_once()


def test_http_error_is_not_reported_as_no_data(monkeypatch):
    monkeypatch.setattr(data, "require_credentials", lambda: ("test", "test"))
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: response)
    with pytest.raises(requests.HTTPError):
        data.fetch_bars("AAA", date(2025, 1, 1), date(2025, 1, 3))


def test_no_bars_returns_valid_empty_frame(monkeypatch):
    monkeypatch.setattr(data, "require_credentials", lambda: ("test", "test"))
    response = Mock()
    response.json.return_value = {"bars": None, "next_page_token": None}
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: response)
    df = data.fetch_bars("AAA", date(2025, 1, 1), date(2025, 1, 3))
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_repeated_page_token_fails_instead_of_looping(monkeypatch):
    monkeypatch.setattr(data, "require_credentials", lambda: ("test", "test"))
    response = Mock()
    response.json.return_value = {"bars": [], "next_page_token": "same"}
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: response)
    with pytest.raises(RuntimeError, match="repeated"):
        data.fetch_bars("AAA", date(2025, 1, 1), date(2025, 1, 3))
