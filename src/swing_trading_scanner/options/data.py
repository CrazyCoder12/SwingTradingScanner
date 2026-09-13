"""Fetch paginated daily bars from Alpaca's market-data API."""

import os

import pandas as pd
import requests

from ..config import require_credentials
from .config import ALPACA_BASE_URL


def fetch_bars(symbol, start, end):
    key, secret = require_credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "adjustment": "split",
        "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
        "limit": 10000,
    }
    bars = []
    seen_pages = set()
    while True:
        response = requests.get(
            f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        bars.extend(payload.get("bars") or [])
        page = payload.get("next_page_token")
        if not page:
            break
        if page in seen_pages:
            raise RuntimeError("Alpaca returned a repeated pagination token")
        seen_pages.add(page)
        params["page_token"] = page

    columns = ["open", "high", "low", "close", "volume"]
    if not bars:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="date"))
    df = pd.DataFrame(bars).rename(
        columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    return df.drop_duplicates("date").set_index("date").sort_index()[columns]
