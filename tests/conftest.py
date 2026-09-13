"""Keep all tests isolated from credentials and external services."""

import pytest
import requests


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(
        "swing_trading_scanner.config.load_dotenv", lambda *a, **k: None
    )

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not make network requests")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
