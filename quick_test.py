# quick_test.py — just to confirm keys work
from dotenv import load_dotenv
import os, requests

load_dotenv()

headers = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
}

r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
print(r.status_code, r.json().get("status"))  # should print: 200 ACTIVE