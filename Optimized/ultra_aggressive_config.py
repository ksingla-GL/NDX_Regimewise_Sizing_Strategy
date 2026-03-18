"""
Optimization: Ultra aggressive parameter tuning.

Result vs Base (asym SL -2/-3):
  CAGR:   28.7% -> 38.0%
  MaxDD:  -24.7% -> -29.5%
  EndEq:  $4,389 -> $12,407

Parameter changes from default:
  l1_accel_size:          80 -> 100   (full allocation in confirmed uptrends)
  ext20_l2_l3_boundary:  -3% -> -5%  (fewer shallow pullback trades)
  ext20_l1_trim:          5% -> 10%  (let winners run longer)
  trend_health_weak_mult: 0.75 -> 1.0 (disable slope penalty, SL handles risk)
  momentum_phase_mult:   1.25 -> 1.5  (stronger momentum boost)

Plus optimized base (V2 short fix + V3 asymmetric SL):
  kill_s3 = True
  vol_filter_short_threshold = 35.0
  intraday_stop_threshold_long = -2.0
  intraday_stop_threshold_short = -3.0
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


def get_ultra_aggressive_config():
    """Return config with ultra aggressive parameter tuning."""
    cfg = DEFAULT_CONFIG.copy()
    # Optimized base (V2 short fix)
    cfg.kill_s3 = True
    cfg.vol_filter_short_threshold = 35.0
    # Asymmetric SL
    cfg.intraday_stop_enabled = True
    cfg.intraday_stop_threshold_long = -2.0
    cfg.intraday_stop_threshold_short = -3.0
    # Ultra aggressive params
    cfg.l1_accel_size = 100.0
    cfg.ext20_l2_l3_boundary = -5.0
    cfg.ext20_l1_trim = 10.0
    cfg.trend_health_weak_mult = 1.0
    cfg.momentum_phase_mult = 1.5
    return cfg


if __name__ == "__main__":
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)
    tqqq = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)
    sqqq = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)

    cfg = get_ultra_aggressive_config()
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
