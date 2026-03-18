"""
Conservative configuration - structural fixes only, minimal overfit risk.

Result vs Baseline:
  CAGR:      9.2% -> 30.7%
  MaxDD:    -45.4% -> -24.7%
  End Equity: $373 -> $5,516

All 4 changes have clear causal rationale beyond backtest curve-fitting:
  1. Kill S3: cash in Regime D instead of shorting SQQQ (S3 had ~1/30 win rate)
  2. Vol filter: skip shorting in Regime C if 20d realized vol > 35% annualized
  3. Asymmetric intraday stop-loss: -2% for TQQQ, -3% for SQQQ positions
     Tighter long-side stop because TQQQ is held more often and 3x leverage
     means a 0.67% NDX drop = 2% portfolio-level hit at full allocation.
  4. L2/L3 boundary: -3% -> -5%. Shifts Regime B time from money-losing L3
     into less-bad L2, reducing deep pullback exposure. +2% CAGR, same MaxDD.
"""

import sys
import os

# Resolve paths - works both as script and interactively in Spyder/IPython
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.path.abspath(os.getcwd())

_PROJECT_ROOT = (os.path.dirname(_SCRIPT_DIR)
                 if os.path.basename(_SCRIPT_DIR) in ("Archived_Opt", "Optimization", "Optimized")
                 else _SCRIPT_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from config import DEFAULT_CONFIG
from engine import StrategyEngine

INPUT_DIR = os.path.join(_PROJECT_ROOT, "Inputs")


def get_conservative_config():
    """Return config with structural fixes only (low overfit risk)."""
    cfg = DEFAULT_CONFIG.copy()
    cfg.kill_s3 = True
    cfg.vol_filter_short_threshold = 35.0
    cfg.intraday_stop_enabled = True
    cfg.intraday_stop_threshold_long = -2.0
    cfg.intraday_stop_threshold_short = -3.0
    cfg.ext20_l2_l3_boundary = -5.0
    return cfg


if __name__ == "__main__":
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)
    tqqq = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)
    sqqq = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)

    cfg = get_conservative_config()
    engine = StrategyEngine(cfg)
    results = engine.run(ndx["Close"], tqqq["Close"], sqqq["Close"], tqqq["Low"], sqqq["Low"])

    start_eq = results["Equity"].iloc[0]
    end_eq = results["Equity"].iloc[-1]
    years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
    max_dd = results["Drawdown_Pct"].min()
    stops = results["Stop_Triggered"].sum()

    print(f"Period  : {results.index[0].date()} to {results.index[-1].date()}")
    print(f"CAGR    : {cagr:.1f}%")
    print(f"Max DD  : {max_dd:.1f}%")
    print(f"End Eq  : {end_eq:.2f}")
    print(f"Return  : {(end_eq / start_eq - 1) * 100:.1f}%")
    print(f"SL fires: {stops}")
