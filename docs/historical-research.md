> Archived research notes from earlier strategy versions. Results have not been reproduced
> during this cleanup. Rules, filenames, and planned work below may differ from the current code.
> See [the strategy guide](strategies.md) for current module names and known limitations.

# EMA Trading — Strategy Research & Backtest Notes

## Project Overview
Day trading backtester using Alpaca Markets API (free IEX feed), 10-min candles, 30-stock watchlist.
Capital: $1,000/trade. Language: Python. API keys via `.env` (ALPACA_API_KEY, ALPACA_SECRET_KEY).

---

## Files
| File | Purpose |
|------|---------|
| `backtest.py` | EMA 5/12 crossover strategy (Long + Short) — most refined version |
| `orb_backtest.py` | ORB + VWAP + RVOL strategy — new higher win-rate strategy |
| `test_backtest.py` | Unit tests for backtest.py logic using synthetic data |
| `quick_test.py` | Verify Alpaca API key connectivity |

---

## Watchlist (30 stocks)
```
AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, TSLA, NFLX, CRM,
ORCL, INTC, QCOM, MU, AVGO, JPM, BAC, GS, MS, V,
SPY, QQQ, IWM, XLK, XLF, COIN, HOOD, SQ (no data), PYPL, SHOP
```
Note: SQ returns no data from Alpaca IEX feed — always skipped.

---

## Strategy 1: EMA 5/12 Crossover (`backtest.py`)

### Current Rules (fully evolved version)
- **Long entry:** 5 EMA crosses above 12 EMA + close above both EMAs
- **Short entry:** 5 EMA crosses below 12 EMA + close below both EMAs
- **Stop:** Hard stop -2% (long) / +2% (short) from entry
- **Exit:** Price crosses back through 12 EMA OR 5 EMA crosses back
- **Breakeven protection:** After 2% profit, don't exit at a loss
- **Force close:** 15:50 candle (EOD)
- **ENTRY_EARLIEST:** `09:50` — skip the 9:40 open candle (whipsaw trap)
- **NO_ENTRY_START:** `11:30` — lunch chop zone begins
- **NO_ENTRY_END:** `14:00` — resume entries after 2pm
- **MAX_TRADES_PER_STOCK:** 2 per day

### Evolution of Rules (what changed and why)

#### Original (long only)
- March 2026 raw: **898 trades, 26.9% win rate, -$388.89**
- Problem: 11:30–13:30 window had only 13% win rate, -$309 in losses

#### Step 1: Block 11:30–13:30 lunch window
- March improved: **729 trades, 30.2%, -$79** (saved $309)

#### Step 2: Skip 9:40 open candle + extend block to 14:00
- Feb 2026 long-only: **461 trades, 37.7%, +$206** (was -$61)
- March 2026 long-only: **512 trades, 31.2%, +$204** (was -$389)
- Key win: hard stops went to ZERO — gap-down opens at 9:40 caused all the stops

#### Step 3: Add Short trades
- Feb 2026 L+S: **727 trades, 37.1%, +$263**
- March 2026 L+S: **775 trades, 35.1%, +$457**
- Combined Feb+Mar: **1,502 trades, 36.1%, +$720**
- Shorts outperformed longs significantly (market was bearish Jan-Mar 2026)

### Full Backtest Results — EMA Strategy

| Period | Days | Trades | Win% | P&L | P&L/day |
|--------|------|--------|------|-----|---------|
| Q4 2025 (Oct–Dec) | 66 | 2,421 | 33.3% | +$896 | +$13.58 |
| Feb 2026 | 20 | 727 | 37.1% | +$263 | +$13.14 |
| Mar 2026 | 19 | 775 | 35.1% | +$457 | +$24.06 |
| **All combined** | **105** | **3,923** | **34.4%** | **+$1,616** | **+$15.39** |

### Q4 2025 Monthly Breakdown
| Month | Trades | Win% | P&L |
|-------|--------|------|-----|
| October | 873 | 34.4% | +$640 |
| November | 699 | 37.6% | +$581 |
| December | 849 | 28.6% | -$324 ⚠️ |

December underperformed — holiday low-volume chop. Watch out for Dec every year.

### EMA Strategy: Long vs Short Performance
Q4 2025:
- LONG: 1,168 trades, 31.4% win, **-$207**
- SHORT: 1,253 trades, 35.0% win, **+$1,103**

Feb+Mar 2026:
- LONG: 750 trades, 34.5%, +$313
- SHORT: 752 trades, 37.6%, +$407

**Shorts consistently outperform longs** — the Jan-Mar 2026 market was in a downtrend (tariffs, macro).

### EMA Strategy: Time Slot Analysis (from March 2026 unfiltered baseline)
Original all-time slots before filtering (March, long-only, raw):
| Slot | Win% | P&L | Decision |
|------|------|-----|---------|
| 09:30–09:40 | ~15% | -$192 | **BLOCKED** (now skip 9:40) |
| 10:00–11:00 | 36.3% | +$144 | ✅ Keep |
| 11:30–13:30 | 13.0% | -$309 | **BLOCKED** |
| 14:00–15:00 | 23.8% | -$77 | Marginal, kept |
| 15:00–15:55 | 45-49% | +$65-177 | ✅ Best slot |

### EOD (End of Day) Trade Analysis
EOD trades (held to 15:50) have ~75% win rate across all periods.
**Why:** Survivor bias — trades that survive all exit rules for 5+ hours were already in strong momentum. They self-select as the best trades.

Early-entry EOD trades (+$30 avg for 10am entry) >> late-entry EOD (+$2 avg for 15:30 entry).

### Exit Reason Breakdown (Feb+Mar 2026 with all rules)
| Exit | Count | Win% | P&L |
|------|-------|------|-----|
| Price < 12 EMA (long exit) | ~600 | 22-29% | -$400 to -$250 |
| Price > 12 EMA (short exit) | ~250 | 25% | -$90 |
| End of Day (3:55 PM) | ~200 | 74-79% | +$450-578 |
| Hard Stop Loss (-2%) | very rare | 0% | -$20 each |

---

## Strategy 2: ORB + VWAP + RVOL (`orb_backtest.py`)

### Rules
- **Opening Range:** First 30 min (9:30–9:59 candles = 9:30, 9:40, 9:50)
  - ORB High = max(high) of those 3 candles
  - ORB Low = min(low) of those 3 candles
- **Long entry:** 10-min candle closes ABOVE ORB high + price > VWAP + RVOL ≥ 1.5x
- **Short entry:** 10-min candle closes BELOW ORB low + price < VWAP + RVOL ≥ 1.5x
- **Stop:** ORB low (long) / ORB high (short)
- **Target:** entry ± 2 × risk (2:1 R/R fixed)
- **Entry window:** 10:00–11:30 only (ORB signal expires at 11:30)
- **Max trades:** 1 per stock per day
- **Force close:** 15:50
- **Min ORB range:** 0.3% of price (skip flat/holiday opens)
- **RVOL lookback:** 20 trading days per time slot (fetches 45 extra calendar days of data)

### VWAP Calculation
```
typical_price = (high + low + close) / 3
VWAP = cumsum(typical_price × volume) / cumsum(volume)
Resets every trading day at 9:30
```

### RVOL Calculation
Groups candles by time-of-day (e.g., all "10:00" candles), then rolling 20-period mean.
Uses shift(1) to prevent lookahead bias.
```python
df.groupby(time_slot)['volume'].transform(
    lambda x: x / x.shift(1).rolling(20, min_periods=5).mean()
)
```

### ORB Backtest Results — Jan–Mar 2026 (61 trading days)

| Month | Trades | Win% | P&L |
|-------|--------|------|-----|
| January | 505 | 49.5% | -$121 ⚠️ |
| February | 418 | 56.7% | +$707 |
| March | 253 | 50.6% | +$450 |
| **Total** | **1,176** | **52.3%** | **+$1,036** |

**P&L/day: +$16.98**

January weakness: RVOL baseline was still warming up (only 45 cal days of history, some slots had < 5 samples).

### ORB Long vs Short
| Dir | Trades | Win% | P&L |
|-----|--------|------|-----|
| LONG | 508 | 48.0% | +$228 |
| SHORT | 668 | 55.5% | +$808 |

### ORB Entry Time Distribution
| Entry Time | Win% | P&L | Notes |
|------------|------|-----|-------|
| **10:00** | **61.0%** | **+$395** | 🏆 Best — immediate breakout |
| **10:10** | **53.6%** | **+$344** | Strong |
| 10:20 | 48.1% | +$65 | OK |
| 10:30 | 50.4% | +$120 | OK |
| 10:40 | 44.6% | -$210 | ⚠️ Weak |
| **10:50** | **56.2%** | **+$211** | Strong bounce-back |
| 11:00 | 48.9% | -$21 | Marginal |
| 11:10 | 54.9% | +$125 | OK |
| 11:20 | 53.7% | +$59 | OK |
| 11:30 | 40.0% | -$52 | ⚠️ Late/weak — near cutoff |

**Key insight:** 10:00 and 10:10 are the golden candles. Earlier breakout = stronger conviction.

### ORB Exit Reason Issue (Important!)
| Exit | Count | Win% | P&L |
|------|-------|------|-----|
| Force Close (15:50) | 556 | 54.5% | +$841 |
| Stop Loss | 55 | 0% | -$862 |
| Target Hit (2:1) | 9 | 100% | +$216 |

**Critical finding:** The 2:1 fixed target almost never triggers (9 times in 61 days).
Trades are instead riding to EOD force close. The strategy's real edge is **momentum continuation**, not the fixed target.
**Recommended next step:** Replace fixed target with trailing stop or VWAP break exit.

### ORB Top/Bottom Stocks (Jan–Mar 2026)
Top performers:
- COIN: 60.0% win, +$358
- MU: 60.5% win, +$277
- MS: 56.1% win, +$126
- AMD: 50.0% win, +$126
- INTC: 64.7% win, +$98

Bottom performers:
- META: 34.7% win, -$97
- SHOP: 50.0% win, -$76
- XLK: 39.1% win, -$69

---

## Strategy Comparison

| Strategy | Period | Days | Trades | Win% | P&L | P&L/day |
|----------|--------|------|--------|------|-----|---------|
| EMA 5/12 L+S (evolved) | Feb+Mar 2026 | 39 | 1,502 | 36.1% | +$720 | +$18.46 |
| EMA 5/12 L+S (evolved) | Q4 2025 | 66 | 2,421 | 33.3% | +$896 | +$13.58 |
| **ORB+VWAP+RVOL** | **Jan–Mar 2026** | **61** | **1,176** | **52.3%** | **+$1,036** | **+$16.98** |

**ORB wins on win rate (52% vs 34-36%) but similar P&L/day.**
The EMA strategy trades more frequently (higher volume of trades).

---

## Planned Next Steps (not yet built)

### ORB Improvements
1. **Replace fixed 2:1 target** with trailing stop or VWAP cross exit — 2:1 target almost never hits
2. **Filter 10:40 entries** — consistently underperforms (-$210), consider blocking 10:40 candle
3. **Filter 11:30 entries** — near expiry, 40% win rate, consider tightening to 11:20 cutoff
4. **More history** — January RVOL baseline was thin; retest with 6+ months data

### Strategy 3: Swing Trade (not yet coded)
- Multi-timeframe: Daily 21 EMA trend filter → only long if stock above daily 21 EMA
- Entry: First pullback to 9 EMA on 5-min chart after strong first-hour move (>1.5%)
- PEAD: Post-Earnings Announcement Drift — buy first pullback after 5%+ earnings gap up
- Hold: 3–5 days
- File to create: `swing_backtest.py`

### General Improvements to Consider
- **Relative Volume filter on EMA strategy** — only take EMA crossovers when RVOL ≥ 1.5x
- **Daily trend filter on EMA strategy** — only long when stock above daily 21 EMA
- **Tighten 14:00–15:00 window** on EMA strategy — still slightly negative in some months
- **December filter** — consider reducing position size or pausing in Dec (holiday chop)

---

## Key Lessons Learned

1. **Lunch hour kills returns** — 11:30–14:00 is universally bad. Low volume, no direction, choppy.
2. **Opening candle (9:40) is a trap** — gaps, news reactions, algos fighting each other. Skip it.
3. **EOD trades are the best** — anything held to 3:55pm has ~75% win rate (survivor bias, momentum confirmation).
4. **Shorts matter** — Jan-Mar 2026 was bearish. Long-only would have missed half the edge.
5. **Early ORB breakouts >> late ones** — 10:00 candle at 61% win rate. 11:30 candle at 40%.
6. **Fixed targets don't work well intraday** — momentum continuation (hold to EOD) beats fixed R/R.
7. **RVOL filter is powerful** — removes noise trades. Forces you to only trade when institutions are active.
8. **December is consistently the worst month** — holiday volume, year-end repositioning.

---

## Running the Backtests

```bash
# EMA Strategy
uv run backtest.py --start 2026-01-01 --end 2026-03-26

# ORB Strategy
uv run orb_backtest.py --start 2026-01-01 --end 2026-03-26

# Single day test
uv run backtest.py --date 2026-03-15
uv run orb_backtest.py --date 2026-03-15

# Verify API keys
uv run quick_test.py
```

API keys loaded from `.env`:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```
