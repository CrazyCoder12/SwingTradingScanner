# Strategy guide

## Daily scanners

- **`ema`** uses daily Alpaca bars for 5/12 crossovers and medium-term trend shifts involving 34/50 EMAs. It reports bullish and bearish signals in separate tables.
- **`confluence`** combines EMA crossovers with momentum, volatility, and volume checks. `--min-score` controls the confirmation threshold; `--no-atr-filter` disables the minimum ATR gate.
- **`pullback`** uses Yahoo Finance data to find 9/21 EMA pullbacks after a crossover, with volume and SPY trend context. Suggested option direction, expiry, and delta are heuristics, not live option-chain selections.
- **`options`** is the modular daily pipeline. It computes an eight-point score from EMA alignment, proximity to EMA21, RSI, volume, a rising close, and price above EMA50. Scores of six or more produce historical signals after a 50-bar warmup.

The modular pipeline reports SPY/QQQ market regime as context; it does not use that regime to suppress signals. Missing benchmark data produces an unknown regime. Each symbol is evaluated only against its own price history.

Its outcome evaluator assumes a long entry at the signal close, a stop 1.5 ATR below entry, and a target 2 ATR above entry. It examines up to 14 subsequent bars. A bar that touches both levels counts as a stop; an unresolved signal counts in the denominator but not as a win. The displayed `win_rate` is therefore a target-hit fraction, not realized portfolio profitability. Recent signals have shorter evaluation windows, and signals can overlap.

## Sector scanners

The sector commands use fixed stock-to-sector watchlists. These lists are not a maintained index-constituent database.

```bash
swing-scanner sector --date 2025-01-02 --csv
swing-scanner sector-open --date 2025-01-02 --csv
swing-scanner sector-snapshot --date 2025-01-02 --snapshot --no-wait --csv
swing-scanner sector-history --date 2025-01-02 --snapshot --no-wait --csv
```

These illustrate syntax only. Use a recent available session for intraday Yahoo Finance history. Live snapshots cannot reconstruct an arbitrary past market state. Snapshot commands may wait for their configured time unless `--no-wait` is supplied. Outputs are written relative to the current working directory; each command prints the paths it creates.

## Backtests

| Command | Main behavior |
| --- | --- |
| `backtest-ema` | Long-only 5/12 trend setups above 34/50 EMAs; EMA cross and premarket-break entries; close-based stops |
| `backtest-orb` | First 10-minute opening range; breakout with VWAP and relative volume; fixed risk/reward target |
| `backtest-orb-ema` | Opening-range breakout with EMA alignment and an afternoon crossover entry variant |
| `backtest-swing` | Daily portfolio with regime/volatility filters, ATR sizing, partial exits, and trailing rules |
| `backtest-tiered` | Daily signal selection followed by intraday entries and tiered risk-multiple exits |
| `backtest-sector` | Sector ETF ranking followed by five-minute EMA momentum simulation |

```bash
swing-scanner backtest-ema --date 2025-01-02
swing-scanner backtest-orb-ema --start 2025-01-02 --end 2025-01-31
swing-scanner backtest-swing --year 2025
swing-scanner backtest-tiered --start-date 2025-01-02 --symbol AAPL --out reports/tiered
swing-scanner backtest-sector --start 2025-01-02 --end 2025-01-31
```

Create `reports/` before using a nested output prefix. The tiered strategy supports CSV exports through `--out`; the sector backtest writes trade CSV and text reports. Several other backtests report only to the terminal.

## Limitations

- Strategy implementations are research variants; shared execution assumptions have not been standardized.
- The EMA backtest's implementation can enter on the crossover candle despite its older next-candle-confirmation description. Its breakeven flag changes the exit label; it does not guarantee a fill at the entry price.
- Close-based stops can exceed their nominal loss threshold. OHLC bars do not reveal the order of intrabar events, spread, or executable liquidity.
- Commission, slippage, and borrowing costs are absent or simplified depending on the module. Results should not be compared without normalizing assumptions.
- Some date helpers skip weekends only. The daily swing strategy includes a hardcoded FOMC exclusion calendar that requires maintenance.
- Live snapshot data mixed with historical dates can introduce lookahead or date mismatches. Sector snapshot outputs are not validated historical backtests.
- Feed settings differ across older implementations; several backtests request IEX directly. Setting `ALPACA_DATA_FEED` does not override a module's explicit feed selection.
- Yahoo Finance intraday history is limited. Requests can fail or be throttled, and older scripts sometimes skip unavailable symbols rather than fail the whole run.
- Fixed watchlists can introduce selection and survivorship bias. No walk-forward evaluation or comprehensive bias audit has been performed.
- Options suggestions are static preferences based on underlying stock setups. No option-chain valuation or option execution is performed.

The [archived notes](historical-research.md) document earlier experiments. Their filenames, rules, and reported outcomes may not match the current implementation and have not been reproduced during this cleanup.
