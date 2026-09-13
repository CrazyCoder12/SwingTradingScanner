# Swing Trading Scanner

**Python tools for screening stock setups and researching trading strategies with historical market data.**

Swing Trading Scanner brings together daily EMA scans, multi-indicator scoring, sector momentum screens, and intraday and swing backtests. It uses Alpaca and Yahoo Finance data, with a command-line interface for choosing a strategy and inspecting its results.

The project focuses on the engineering workflow behind strategy research: fetching OHLCV data, calculating indicators, detecting signals, simulating exits, and summarizing outcomes. It does not submit orders.

## Features

- **Daily trend scans:** bullish and bearish EMA crossovers and pullback setups.
- **Multi-indicator screening:** EMA, RSI, MACD, stochastic, Bollinger Bands, ATR, and volume confirmation in the confluence scanner.
- **Sector screening:** compare sector ETFs and their constituent watchlists using gap, volume, and VWAP filters.
- **Historical simulations:** EMA, opening-range breakout, sector momentum, daily swing, and tiered-exit strategy variants.
- **Modular daily pipeline:** separate data access, indicators, scoring, signal detection, and per-symbol outcome evaluation.
- **Console reports and selected CSV exports:** output support depends on the command; use its `--help` and the [strategy guide](docs/strategies.md).
- **Offline regression tests and CI:** synthetic market data and mocked HTTP requests validate core behavior without API credentials.

## Quick start

Requires **Python 3.10 or newer**. Commands below use macOS/Linux shell syntax.

```bash
git clone https://github.com/CrazyCoder12/SwingTradingScanner.git
cd SwingTradingScanner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
swing-scanner --help
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`.

For runtime dependencies only, use `python -m pip install -e .`.

### Try a scan without API keys

The Yahoo Finance pullback scanner needs internet access but no Alpaca credentials:

```bash
swing-scanner pullback --tickers AAPL MSFT NVDA
```

It prints matching setups and writes `scan_results_YYYYMMDD.csv` in the working directory when matches exist. No matches is a valid result. Yahoo Finance requests may be throttled or return no data.

### Configure Alpaca data access

```bash
cp .env.example .env
```

Set your **Alpaca market-data credentials** in `.env`:

```dotenv
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_DATA_FEED=iex
```

Run commands from the repository root so `.env` is found. Exported environment variables take precedence. `.env` is excluded from Git. A GitHub token cannot authenticate Alpaca requests.

Feed coverage and historical availability depend on your provider account. Some older backtest modules select IEX internally; see [strategy limitations](docs/strategies.md#limitations).

```bash
# Scan daily EMA signals from a chosen date
swing-scanner ema --start-date 2025-01-02 --symbol AAPL

# Evaluate multi-indicator confirmations
swing-scanner confluence --start-date 2025-01-02 --symbol AAPL --min-score 3

# Run the modular daily pipeline on a small watchlist
swing-scanner options --symbols AAPL MSFT --days 365

# Simulate an opening-range breakout over a historical period
swing-scanner backtest-orb --start 2025-01-02 --end 2025-01-31
```

Dates are examples; select a period available from your data provider. Large watchlists and long histories can make many API requests.

## Command reference

| Command | Purpose | Data source |
| --- | --- | --- |
| `ema` | Daily EMA crossover scans | Alpaca |
| `confluence` | Multi-indicator signal confirmation | Alpaca |
| `pullback` | 9/21 EMA pullback setups | Yahoo Finance |
| `sector` | Premarket sector ranking | Alpaca |
| `sector-open` | Sector scan with opening-window detection | Alpaca |
| `sector-snapshot` | Timed sector snapshot | Yahoo Finance |
| `sector-history` | Historical intraday snapshot | Yahoo Finance |
| `options` | Scored daily stock signals and underlying-price evaluation | Alpaca |
| `backtest-ema` | EMA crossover and premarket breakout | Alpaca |
| `backtest-orb` | Opening-range breakout with VWAP/RVOL | Alpaca |
| `backtest-orb-ema` | Opening-range breakout with EMA confirmation | Alpaca |
| `backtest-swing` | Daily swing portfolio simulation | Alpaca |
| `backtest-tiered` | Risk-based sizing and tiered exits | Alpaca |
| `backtest-sector` | Sector momentum simulation | Alpaca |

Run `swing-scanner COMMAND --help` for arguments. `python -m swing_trading_scanner` is an equivalent entry point.

## Project structure

```text
SwingTradingScanner/
├── src/swing_trading_scanner/
│   ├── cli.py                # Command discovery and dispatch
│   ├── config.py             # Environment loading and credential validation
│   ├── scanners/             # Daily, pullback, and sector screens
│   ├── backtests/            # Historical simulation variants
│   └── options/              # Modular daily research pipeline
│       ├── data.py           # Paginated market-data requests
│       ├── indicators.py     # EMA, RSI, ATR, volume averages
│       ├── scoring.py        # Signal scoring rules
│       ├── scanner.py        # Signal detection
│       ├── market_regime.py  # SPY/QQQ trend context
│       └── backtest.py       # Per-symbol target/stop evaluation
├── tests/                    # Offline regression tests
├── docs/
│   ├── strategies.md         # Strategy behavior and limitations
│   ├── development.md        # Development workflow and migration guide
│   └── historical-research.md # Archived, unverified research notes
├── .github/workflows/ci.yml
├── .env.example
└── pyproject.toml
```

The modular pipeline follows:

```text
Watchlist → Daily OHLCV bars → Indicators → Signal scores → Historical signals
                                                           ↓
                                             Per-symbol outcome evaluation
```

The other strategy modules remain independently runnable research implementations. Their indicator periods, watchlists, entry rules, and output formats intentionally vary.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m build
```

Tests use synthetic fixtures and mocked data access. CI runs tests, critical-error lint checks, and packaging on Python 3.10 and 3.12. See the [development guide](docs/development.md) for migration details.

## Research scope and limitations

This is a research project, not a production execution platform or an options-pricing engine. Options-related output describes stock signals and heuristic contract preferences; it does not calculate option premiums, Greeks, or options P&L.

Backtest assumptions vary by module. Fees, spreads, slippage, exchange holidays, and intrabar execution are not consistently modeled. Some signals overlap, some filters use current snapshots, and some calendars are hardcoded. The [strategy guide](docs/strategies.md) explains these limits before interpreting results.

Historical notes are retained as an archive, not verified performance evidence. No profitability claim is made. Use for educational research, not as investment advice.

## License

No license has been selected. A public repository does not automatically grant permission to reuse or redistribute its code.
