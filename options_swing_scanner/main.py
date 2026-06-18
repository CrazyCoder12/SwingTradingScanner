from dotenv import load_dotenv
load_dotenv()
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))
from datetime import datetime, timedelta
from watchlist import WATCHLIST
from data import fetch_bars
from indicators import add_indicators
from scanner import scan
from backtest import backtest
from market_regime import get_regime

def run():
    print("Running scanner...")

    regime = get_regime()
    print("Market regime:", regime)

    start = datetime.now() - timedelta(days=365)
    end = datetime.now()

    all_signals = []

    for symbol in WATCHLIST:
        df = fetch_bars(symbol, start, end)
        if df.empty:
            continue

        df = add_indicators(df)
        signals = scan(df)

        for s in signals:
            s["symbol"] = symbol

        all_signals.extend(signals)

    print("Signals:", len(all_signals))

    if all_signals:
        stats = backtest(all_signals, df)
        print("Backtest:", stats)

if __name__ == "__main__":
    run()