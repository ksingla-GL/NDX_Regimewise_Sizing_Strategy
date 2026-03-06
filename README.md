# NDX Regime-Wise Sizing Strategy

Automated end-of-day systematic trading strategy for **TQQQ** (3× long Nasdaq-100) and **SQQQ** (3× short Nasdaq-100), driven entirely by the **$NDX** daily close.

## Project Overview

The system classifies each trading day into one of four market regimes based on where $NDX sits relative to its 20-day and 250-day moving averages, then selects a sub-strategy rule and computes a position size accordingly. Bollinger Band momentum detection and trend health overlays adjust conviction, while a layered risk management system protects capital.

**Developer**: Kshitij Singla
**Inspiration**: Modeled after the "Whitelight" approach - EOD trend/momentum system on leveraged Nasdaq ETFs, ~1-2 trades/week.

### Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Strategy Formalization (logic document) | Complete |
| 2 | Data Acquisition & Strategy Coding | Complete |
| 3 | Backtesting & Optimization | Pending |
| 4 | IBKR TWS Automation & Paper Trading | Pending |
| 5 | Final Tuning & Go-Live | Pending |

## Strategy Summary

- **Decision time**: Market close (~3:59 PM ET). One signal per day.
- **Outputs**: Target TQQQ % (0–100) and target SQQQ % (0–80). Remainder is cash.
- **Constraint**: TQQQ and SQQQ are never held simultaneously.
- **Trade frequency**: ~1-2 trades/week, ~8-10/month.

### Regimes

| Regime | Condition | Bias |
|--------|-----------|------|
| A - Strong Uptrend | Close > MA_20 AND Close > MA_250 | TQQQ heavy |
| B - Pullback in Uptrend | Close ≤ MA_20 AND Close > MA_250 | TQQQ moderate |
| C - Strong Downtrend | Close < MA_20 AND Close ≤ MA_250 | SQQQ heavy |
| D - Bounce in Downtrend | Close > MA_20 AND Close ≤ MA_250 | SQQQ moderate |

### Sub-Strategy Rules

**Long side (Regimes A/B)** - evaluated in priority order:

| Rule | Regime | Trigger | Sizing Range |
|------|--------|---------|-------------|
| L4: Oversold Snap-Back | B | Close < BB_Lower | 10–20% TQQQ |
| L3: Deep Pullback | B | EXT_20 ≤ −3% | 10–25% TQQQ |
| L2: Shallow Pullback | B | EXT_20 > −3% | 40–50% TQQQ |
| L1: Trend Following | A | Always in Regime A | 40–80% TQQQ |

**Short side (Regimes C/D)** - evaluated in priority order:

| Rule | Regime | Trigger | Sizing Range |
|------|--------|---------|-------------|
| S2: Breakdown Short | C | Fresh MA_250 break + ROC_ACCEL < 0 | 50–70% SQQQ |
| S1: Trend Following Short | C | Always in Regime C | 25–60% SQQQ |
| S3: Overbought Mean Reversion | D | Always in Regime D | 15–50% SQQQ |

### Post-Rule Adjustments

1. **Trend Health** (MA_250 slope): 0.75× if trend structure is weakening.
2. **Momentum Phase** (Bollinger Band ride + expansion): 1.25× during confirmed momentum.

### Risk Management

- **Whipsaw Filter**: MA_250 cross → exit to cash, require 2 consecutive confirms before re-entry.
- **Drawdown Circuit Breaker**: −40% → 50% reduction; −50% → full halt.
- **Daily Loss Limits**: >15% single-day → reduce-only next day; 2 consecutive >10% days → 50% for 5 days.
- **Min Trade Threshold**: 5% portfolio delta required to rebalance (risk actions exempt).
- **Intraday Stop-Loss** (optional): If portfolio intraday P&L from prior close breaches threshold (default -10%), liquidate all positions to cash immediately. EOD system still runs at close and may re-enter. Enabled via `intraday_stop_enabled` config flag. Requires TQQQ/SQQQ daily Low prices for backtesting.

## Repository Structure

```
NDX_Regimewise_Sizing_Strategy/
├── config.py          # All tunable parameters (StrategyConfig dataclass)
├── data.py            # Yahoo Finance data pull for NDX, TQQQ, SQQQ
├── indicators.py      # Indicator computation (MAs, EXT, ROC, BB, momentum)
├── regime.py          # Regime classification (A/B/C/D)
├── rules.py           # Sub-strategy rule selection & base position sizing
├── adjustments.py     # Trend health & momentum phase overlays
├── engine.py          # Daily decision flow orchestrator (StrategyEngine)
├── test_leaves.py     # Unit tests - 84 sizing leaves + early exit tests
├── Docs/
│   ├── Strategy_Rules.pdf   # Authoritative strategy specification (Phase 1)
└── Inputs/
    ├── NDX.csv              # $NDX historical daily OHLCV
    ├── TQQQ.csv             # TQQQ historical daily OHLCV
    └── SQQQ.csv             # SQQQ historical daily OHLCV
```

## Module Guide

### `config.py`
Central configuration via `StrategyConfig` dataclass. Every threshold, period, sizing value, and risk parameter lives here. Pass modified copies to the engine for optimization sweeps.

### `data.py`
Downloads historical OHLCV data from Yahoo Finance for $NDX, TQQQ, and SQQQ. Includes date alignment validation across all three tickers.

```bash
python data.py
```

### `indicators.py`
Computes all strategy indicators from $NDX close prices: MA_20, MA_250, EXT_20, EXT_250, ROC_10, ROC_ACCEL, MA_250_Slope, Bollinger Bands, momentum phase detection, and consecutive days below BB_Lower.

### `regime.py`
Pure function: given (close, ma_20, ma_250), returns regime A/B/C/D. Implements boundary rules (Close = MA_250 → bear; Close = MA_20 → below).

### `rules.py`
Two concerns: **rule selection** (which sub-strategy is active) and **base sizing** (what percentage). Long rules (L1–L4) and short rules (S1–S3) are each priority-ranked.

### `adjustments.py`
Applies trend health modifier (0.75× or 1.0×) then momentum phase overlay (1.25× or 1.0×), with final caps at 100% TQQQ / 80% SQQQ.

### `engine.py`
The `StrategyEngine.run()` method walks day-by-day through historical data, maintaining all state: whipsaw filter, S2 break tracking, risk modes, dollar positions, and equity curve. Produces a DataFrame with daily signals, allocations, and performance.

### `test_leaves.py`
Unit tests covering:
- **4 early-exit paths**: data unavailable (skip), whipsaw waiting, MA_250 cross day, circuit breaker halt.
- **84 sizing leaves**: 9 long base rules × 4 adjustment combos + 12 short base rules × 4 adjustment combos. Each verifies rule selection, base sizing, and adjusted output.

## Usage

### Pull Data
```bash
python data.py
```

### Run Tests
```bash
python -m pytest test_leaves.py -v
```

### Parameter Sweeps
```python
cfg = DEFAULT_CONFIG.copy()
cfg.ma_short_period = 15
cfg.whipsaw_confirm_days = 3
results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)
```

## Key Design Decisions

1. **Dollar-based position tracking**: The engine tracks actual dollar amounts (not stale target percentages) so positions drift naturally with returns, giving precise P&L even when the min-trade threshold blocks rebalancing.

2. **Daily loss triggers affect the next day only**: Detected after today's signal is finalized (Step 11b in engine), so they cannot retroactively change today's allocation - matching the PDF's "next trading day" language.

3. **S2 break window includes whipsaw days**: The 10-day countdown starts on the break date and does not pause during the whipsaw filter's cash period. This is intentional per the strategy spec.

4. **Whipsaw cross day is not a confirm day**: The cross day exits to cash but does not count toward the 2-day confirmation requirement.

## Dependencies

- Python 3.9+
- pandas
- numpy
- yfinance (data acquisition only)
