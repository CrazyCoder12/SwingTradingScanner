# Development guide

## Local workflow

Create a virtual environment and install `python -m pip install -e ".[dev]"` from the repository root. Keep credentials in an untracked `.env` file. Environment variables take precedence over that file.

Run `python -m pytest`, `python -m ruff check .`, and `python -m build` before proposing a change. The lint configuration catches syntax and critical name errors; it does not enforce comprehensive style rules on the retained research scripts.

Tests use synthetic candles and mocked HTTP calls. They cover selected strategy invariants, per-symbol evaluation, pagination, missing data, credential handling, and CLI discovery. They do not establish profitability or certify every strategy. Live data access is a separate integration check and depends on provider credentials and entitlements.

## Folder responsibilities

- `scanners/`: screening commands and standalone strategy variants.
- `backtests/`: historical simulations; keep strategy-specific parameters with their engine.
- `options/`: reusable daily pipeline with separate data access and scoring.
- `config.py`: shared environment loading and credential validation.
- `cli.py`: command names and module dispatch; strategy modules own their argument parsing.
- `tests/`: assertions that run offline, with no account or market-data calls.
- `docs/`: current usage and clearly labeled historical research.

## Migration from the original flat layout

| Original file | Command / new location |
| --- | --- |
| `swing_trading.py` | `swing-scanner ema` / `scanners/ema.py` |
| `new_option_swing_test.py` | `swing-scanner confluence` / `scanners/confluence.py` |
| `option_swing_trading.py` | `swing-scanner pullback` / `scanners/pullback.py` |
| `sector_scanner.py` | `swing-scanner sector` / `scanners/sector.py` |
| `2_sector.py` | `swing-scanner sector-open` / `scanners/sector_open.py` |
| `webull_sector.py` | `swing-scanner sector-snapshot` / `scanners/sector_snapshot.py` |
| `tets_yahoo_sector.py` | `swing-scanner sector-history` / `scanners/sector_history.py` |
| `backtest.py` | `swing-scanner backtest-ema` / `backtests/ema.py` |
| `orb_backtest.py` | `swing-scanner backtest-orb` / `backtests/orb.py` |
| `orb_ema_backtest.py` | `swing-scanner backtest-orb-ema` / `backtests/orb_ema.py` |
| `swing_backtest.py` | `swing-scanner backtest-swing` / `backtests/swing.py` |
| `ripster_backtest.py` | `swing-scanner backtest-tiered` / `backtests/tiered.py` |
| `bactest_sector.py` | `swing-scanner backtest-sector` / `backtests/sector.py` |
| `options_swing_scanner/` | `swing-scanner options` / `options/` |
| `STRATEGY_NOTES.md` | `docs/historical-research.md` |
| `test_backtest.py` | Replaced with assertion-based offline tests under `tests/` |
| `quick_test.py` | Removed account-response dump; use a small data scan to check connectivity |

Package-relative paths in this table are under `src/swing_trading_scanner/`. The old root-level script paths are no longer supported. Use the CLI or `python -m swing_trading_scanner.scanners.ema --help`, for example.

## Sharing the project

Before publishing, replace any previously committed provider credentials at the provider. Removing them from current source files does not remove them from Git history. History rewriting is a separate coordinated operation.

Choose a license before inviting external code reuse. Do not present archived research metrics as validated performance. A resume description can focus on the implemented engineering work:

> Built a Python market-data research toolkit with EMA and multi-indicator stock screening, sector ranking, historical strategy simulations, a packaged CLI, and offline regression tests.
