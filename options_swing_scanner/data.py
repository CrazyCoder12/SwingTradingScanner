import requests
import pandas as pd
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

def fetch_bars(symbol, start, end):
    url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars"

    params = {
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "adjustment": "split",
        "limit": 10000,
    }

    bars = []
    page = None

    while True:
        if page:
            params["page_token"] = page

        r = requests.get(url, headers=HEADERS, params=params)
        print("STATUS:", r.status_code)
        print("TEXT:", r.text[:500])
        data = r.json()

        bars.extend(data.get("bars", []))
        page = data.get("next_page_token")

        if not page:
            break

    df = pd.DataFrame(bars)
    if df.empty:
        return df

    df["t"] = pd.to_datetime(df["t"]).dt.date
    df = df.rename(columns={"t":"date","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df = df.set_index("date").sort_index()
    return df