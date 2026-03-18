"""
Optimization Sweep V4 - Core parameter sensitivity.

Base: V3 winner (V2 + Asymmetric SL -2%L/-3%S) = 28.7% CAGR, -24.7% MaxDD

Tests each core parameter individually to find which move the needle:
  A. MA periods (short: 10/15/20/30, long: 150/200/250/300)
  B. Whipsaw confirm days (1/2/3)
  C. L1 sizing (accel: 50/60/70/80, decel: 40/50/60)
  D. L2 sizing (20/30/40)
  E. EXT thresholds (L2/L3 boundary, L1 trim)
  F. Adjustment multipliers (trend health, momentum)
  G. Global sizing scale (0.5/0.6/0.7/0.8/1.0)
  H. Drawdown thresholds
  I. Max SQQQ cap

Key finding: L1 accel and L2/L3 boundary are the biggest movers.
DD thresholds are inert (SL catches drawdowns first).
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

# Sizing fields for global scale
SIZING_FIELDS = [
    "l1_accel_size", "l1_decel_extended_size", "l1_decel_normal_size",
    "l2_extended_size", "l2_normal_size", "l3_moderate_size", "l3_extreme_size",
    "l4_fresh_size", "l4_extended_size",
    "s1_accel_size", "s1_decel_stretched_size", "s1_decel_normal_size",
    "s2_early_expand_size", "s2_early_noexpand_size", "s2_late_expand_size",
    "s2_late_noexpand_size", "s3_bb_upper_override_size",
    "s3_ext_high_decel_size", "s3_ext_high_accel_size",
    "s3_ext_low_decel_size", "s3_ext_low_accel_size",
]


def base_config():
    """Current best: V3 winner (kill_s3 + vol_filter + asymmetric SL)."""
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


def scaled_config(scale):
    """Apply global sizing scale to all position size fields."""
    cfg = base_config()
    for field in SIZING_FIELDS:
        setattr(cfg, field, getattr(cfg, field) * scale)
    return cfg


if __name__ == "__main__":
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)["Close"]
    tqqq_df = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)
    sqqq_df = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)
    tqqq = tqqq_df["Close"]
    sqqq = sqqq_df["Close"]
    tqqq_lo = tqqq_df["Low"]
    sqqq_lo = sqqq_df["Low"]

    all_rows = []

    def test(name, cfg):
        cagr, max_dd, end_eq, stops = run_and_measure(cfg, ndx, tqqq, sqqq, tqqq_lo, sqqq_lo)
        print(f"  {name:<40s} {cagr:>5.1f}% {max_dd:>6.1f}% {end_eq:>9.2f} {stops:>5.0f}")
        all_rows.append({"Scenario": name, "CAGR_%": round(cagr, 1),
                         "Max_DD_%": round(max_dd, 1), "End_Equity": round(end_eq, 2),
                         "Stops": int(stops)})

    hdr = f"  {'Scenario':<40s} {'CAGR':>6s} {'MaxDD':>7s} {'EndEq':>9s} {'Stops':>5s}"
    sep = "  " + "=" * 72

    # === BASE ===
    print("\n--- BASE ---")
    print(hdr); print(sep)
    test("0. Base (asym SL -2/-3)", base_config())

    # === A. MA Periods ===
    print("\n--- A. MA Short Period ---")
    print(hdr); print(sep)
    for p in [10, 15, 20, 30]:
        cfg = base_config(); cfg.ma_short_period = p
        tag = " *" if p == 20 else ""
        test(f"A1. MA_short={p}{tag}", cfg)

    print("\n--- A. MA Long Period ---")
    print(hdr); print(sep)
    for p in [150, 200, 250, 300]:
        cfg = base_config(); cfg.ma_long_period = p
        tag = " *" if p == 250 else ""
        test(f"A2. MA_long={p}{tag}", cfg)

    # === B. Whipsaw Confirm Days ===
    print("\n--- B. Whipsaw Confirm Days ---")
    print(hdr); print(sep)
    for d in [1, 2, 3]:
        cfg = base_config(); cfg.whipsaw_confirm_days = d
        tag = " *" if d == 2 else ""
        test(f"B. Whipsaw={d}{tag}", cfg)

    # === C. L1 Sizing ===
    print("\n--- C. L1 Accel Size ---")
    print(hdr); print(sep)
    for s in [50, 60, 70, 80, 90, 100]:
        cfg = base_config(); cfg.l1_accel_size = s
        tag = " *" if s == 80 else ""
        test(f"C1. L1_accel={s}{tag}", cfg)

    print("\n--- C. L1 Decel Normal Size ---")
    print(hdr); print(sep)
    for s in [30, 40, 50, 60]:
        cfg = base_config(); cfg.l1_decel_normal_size = s
        tag = " *" if s == 60 else ""
        test(f"C2. L1_decel={s}{tag}", cfg)

    # === D. L2 Sizing ===
    print("\n--- D. L2 Normal Size ---")
    print(hdr); print(sep)
    for s in [15, 20, 25, 30, 40]:
        cfg = base_config(); cfg.l2_normal_size = s; cfg.l2_extended_size = s + 10
        tag = " *" if s == 40 else ""
        test(f"D. L2_normal={s} ext={s+10}{tag}", cfg)

    # === E. EXT Thresholds ===
    print("\n--- E. L2/L3 Boundary (ext20) ---")
    print(hdr); print(sep)
    for t in [-2.0, -3.0, -4.0, -5.0, -6.0]:
        cfg = base_config(); cfg.ext20_l2_l3_boundary = t
        tag = " *" if t == -3.0 else ""
        test(f"E1. L2/L3_boundary={t}{tag}", cfg)

    print("\n--- E. L1 Trim Threshold ---")
    print(hdr); print(sep)
    for t in [3.0, 4.0, 5.0, 7.0, 10.0]:
        cfg = base_config(); cfg.ext20_l1_trim = t
        tag = " *" if t == 5.0 else ""
        test(f"E2. L1_trim={t}{tag}", cfg)

    # === F. Adjustment Multipliers ===
    print("\n--- F. Trend Health Multiplier ---")
    print(hdr); print(sep)
    for m in [0.5, 0.65, 0.75, 0.85, 1.0]:
        cfg = base_config(); cfg.trend_health_weak_mult = m
        tag = " *" if m == 0.75 else ""
        test(f"F1. TrendHealth={m}{tag}", cfg)

    print("\n--- F. Momentum Multiplier ---")
    print(hdr); print(sep)
    for m in [1.0, 1.15, 1.25, 1.4, 1.5]:
        cfg = base_config(); cfg.momentum_phase_mult = m
        tag = " *" if m == 1.25 else ""
        test(f"F2. Momentum={m}{tag}", cfg)

    # === G. Global Sizing Scale ===
    print("\n--- G. Global Sizing Scale ---")
    print(hdr); print(sep)
    for s in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        cfg = scaled_config(s)
        tag = " *" if s == 1.0 else ""
        test(f"G. Scale={s}{tag}", cfg)

    # === H. Drawdown Thresholds ===
    print("\n--- H. Drawdown Reduce ---")
    print(hdr); print(sep)
    for t in [-25.0, -30.0, -35.0, -40.0]:
        cfg = base_config(); cfg.drawdown_reduce = t
        tag = " *" if t == -40.0 else ""
        test(f"H1. DD_reduce={t}{tag}", cfg)

    print("\n--- H. Drawdown Halt ---")
    print(hdr); print(sep)
    for t in [-30.0, -35.0, -40.0, -45.0, -50.0]:
        cfg = base_config(); cfg.drawdown_halt = t
        tag = " *" if t == -50.0 else ""
        test(f"H2. DD_halt={t}{tag}", cfg)

    # === I. Max SQQQ Cap ===
    print("\n--- I. Max SQQQ Cap ---")
    print(hdr); print(sep)
    for c in [40.0, 50.0, 60.0, 70.0, 80.0]:
        cfg = base_config(); cfg.max_sqqq = c
        tag = " *" if c == 80.0 else ""
        test(f"I. Max_SQQQ={c}{tag}", cfg)

    # Save all
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(_SCRIPT_DIR, "sweep_v4_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nAll results saved to {out_path}")

    # === Summary: top 10 by CAGR ===
    print(f"\n{'='*60}")
    print("TOP 10 BY CAGR")
    print(f"{'='*60}")
    df_sorted = df.sort_values("CAGR_%", ascending=False).head(10)
    for _, r in df_sorted.iterrows():
        print(f"  {r['Scenario']:<40s} {r['CAGR_%']:>5.1f}% {r['Max_DD_%']:>6.1f}% {r['End_Equity']:>9.2f}")

    # === Summary: top 10 by MaxDD (least negative) ===
    print(f"\n{'='*60}")
    print("TOP 10 BY MAX DD (best risk)")
    print(f"{'='*60}")
    df_sorted = df.sort_values("Max_DD_%", ascending=False).head(10)
    for _, r in df_sorted.iterrows():
        print(f"  {r['Scenario']:<40s} {r['CAGR_%']:>5.1f}% {r['Max_DD_%']:>6.1f}% {r['End_Equity']:>9.2f}")
