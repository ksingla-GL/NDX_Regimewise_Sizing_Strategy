"""
Optimization Sweep V5 - Combo sweep of top parameter movers from V4.

Top movers identified in V4 (each tested individually):
  - L1 accel 90-100: +3 to +6% CAGR (but DD worsens)
  - L2/L3 boundary -5%: +2% CAGR, same DD
  - MA_short=30: +1% CAGR
  - TrendHealth=0.85-1.0: +0.4 to +0.9% CAGR
  - Momentum=1.4-1.5: +0.3-0.4% CAGR, better DD
  - L1_trim=7-10: +0.3% CAGR

Now testing combos. Base = V3 winner (kill_s3 + vol_filter + asymmetric SL).
Conservative combo (31.5%, -24.8%) is the best risk-adjusted option.
"""

import sys
import os

# Resolve paths - works both as script and interactively in Spyder/IPython
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.path.abspath(os.getcwd())

_PROJECT_ROOT = (os.path.dirname(_SCRIPT_DIR)
                 if os.path.basename(_SCRIPT_DIR) in ("Archived_Opt", "Optimization")
                 else _SCRIPT_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG
from engine import StrategyEngine

INPUT_DIR = os.path.join(_PROJECT_ROOT, "Inputs")


def base_config():
    cfg = DEFAULT_CONFIG.copy()
    cfg.kill_s3 = True
    cfg.vol_filter_short_threshold = 35.0
    cfg.intraday_stop_enabled = True
    cfg.intraday_stop_threshold_long = -2.0
    cfg.intraday_stop_threshold_short = -3.0
    return cfg


def run_and_measure(cfg, ndx, tqqq, sqqq, tqqq_lo, sqqq_lo):
    engine = StrategyEngine(cfg)
    results = engine.run(ndx, tqqq, sqqq, tqqq_lo, sqqq_lo)
    start_eq = results["Equity"].iloc[0]
    end_eq = results["Equity"].iloc[-1]
    years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
    max_dd = results["Drawdown_Pct"].min()
    stops = results["Stop_Triggered"].sum()
    return cagr, max_dd, end_eq, stops


if __name__ == "__main__":
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)["Close"]
    tqqq_df = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)
    sqqq_df = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)
    tqqq = tqqq_df["Close"]
    sqqq = sqqq_df["Close"]
    tqqq_lo = tqqq_df["Low"]
    sqqq_lo = sqqq_df["Low"]

    scenarios = []

    # Base
    scenarios.append(("0.  Base", base_config()))

    # Single best movers (for reference)
    c = base_config(); c.ext20_l2_l3_boundary = -5.0
    scenarios.append(("1.  L2/L3 boundary=-5", c))

    c = base_config(); c.l1_accel_size = 90
    scenarios.append(("2.  L1_accel=90", c))

    c = base_config(); c.momentum_phase_mult = 1.5
    scenarios.append(("3.  Momentum=1.5", c))

    # Conservative combos (keep DD low)
    c = base_config(); c.ext20_l2_l3_boundary = -5.0; c.momentum_phase_mult = 1.4
    scenarios.append(("4.  Bndry-5 + Mom1.4", c))

    c = base_config(); c.ext20_l2_l3_boundary = -5.0; c.trend_health_weak_mult = 0.85
    scenarios.append(("5.  Bndry-5 + TH0.85", c))

    c = base_config(); c.ext20_l2_l3_boundary = -5.0; c.momentum_phase_mult = 1.4; c.trend_health_weak_mult = 0.85
    scenarios.append(("6.  Bndry-5+Mom1.4+TH0.85", c))

    c = base_config(); c.ext20_l2_l3_boundary = -5.0; c.ext20_l1_trim = 7.0
    scenarios.append(("7.  Bndry-5 + Trim7", c))

    c = base_config(); c.ext20_l2_l3_boundary = -5.0; c.ext20_l1_trim = 7.0; c.momentum_phase_mult = 1.4
    scenarios.append(("8.  Bndry-5+Trim7+Mom1.4", c))

    # Aggressive combos (push CAGR)
    c = base_config(); c.l1_accel_size = 90; c.ext20_l2_l3_boundary = -5.0
    scenarios.append(("9.  L1_90 + Bndry-5", c))

    c = base_config(); c.l1_accel_size = 90; c.ext20_l2_l3_boundary = -5.0; c.momentum_phase_mult = 1.4
    scenarios.append(("10. L1_90+Bndry-5+Mom1.4", c))

    c = base_config(); c.l1_accel_size = 90; c.ext20_l2_l3_boundary = -5.0; c.trend_health_weak_mult = 0.85
    scenarios.append(("11. L1_90+Bndry-5+TH0.85", c))

    c = base_config(); c.l1_accel_size = 100; c.ext20_l2_l3_boundary = -5.0
    scenarios.append(("12. L1_100+Bndry-5", c))

    c = base_config(); c.l1_accel_size = 100; c.ext20_l2_l3_boundary = -5.0; c.momentum_phase_mult = 1.5
    scenarios.append(("13. L1_100+Bndry-5+Mom1.5", c))

    # MA_short=30 combos
    c = base_config(); c.ma_short_period = 30; c.ext20_l2_l3_boundary = -5.0
    scenarios.append(("14. MA30+Bndry-5", c))

    c = base_config(); c.ma_short_period = 30; c.ext20_l2_l3_boundary = -5.0; c.momentum_phase_mult = 1.4
    scenarios.append(("15. MA30+Bndry-5+Mom1.4", c))

    # Kitchen sink conservative
    c = base_config()
    c.ext20_l2_l3_boundary = -5.0; c.ext20_l1_trim = 7.0
    c.trend_health_weak_mult = 0.85; c.momentum_phase_mult = 1.4
    scenarios.append(("16. Conservative combo", c))

    # Kitchen sink aggressive
    c = base_config()
    c.l1_accel_size = 90; c.ext20_l2_l3_boundary = -5.0; c.ext20_l1_trim = 7.0
    c.trend_health_weak_mult = 0.85; c.momentum_phase_mult = 1.4
    scenarios.append(("17. Aggressive combo", c))

    # Ultra aggressive
    c = base_config()
    c.l1_accel_size = 100; c.ext20_l2_l3_boundary = -5.0; c.ext20_l1_trim = 10.0
    c.trend_health_weak_mult = 1.0; c.momentum_phase_mult = 1.5
    scenarios.append(("18. Ultra aggressive", c))

    # Print results
    print(f"{'Scenario':<35s} {'CAGR':>6s} {'MaxDD':>7s} {'EndEq':>9s} {'Stops':>5s}")
    print("=" * 68)

    rows = []
    for name, cfg in scenarios:
        cagr, max_dd, end_eq, stops = run_and_measure(cfg, ndx, tqqq, sqqq, tqqq_lo, sqqq_lo)
        print(f"{name:<35s} {cagr:>5.1f}% {max_dd:>6.1f}% {end_eq:>9.2f} {stops:>5.0f}")
        rows.append({"Scenario": name, "CAGR_%": round(cagr, 1),
                      "Max_DD_%": round(max_dd, 1), "End_Equity": round(end_eq, 2),
                      "Stops": int(stops)})

    df = pd.DataFrame(rows)
    out_path = os.path.join(_SCRIPT_DIR, "sweep_v5_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
