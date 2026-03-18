"""
Optimization Sweep V2 - Diagnosing and fixing the short side.

Key findings:
  1: SQQQ is the dominant performance drag. Halving all short sizes was
     the single best V2 move (12.8% CAGR vs 9.2% baseline).
  2: Vol filter (skip shorts when 20d realized vol > 35%) + killing S3
     (cash in Regime D) = 15.7% CAGR, -41.7% MaxDD. Breakthrough combo.

Root cause: SQQQ loses money even in bear regimes due to 3x daily
rebalancing decay. Decay is worst in high-volatility environments --
exactly when bear markets are most violent.
"""

import sys
import os

# Resolve paths - works both as script and interactively
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


def load_data():
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)["Close"]
    tqqq = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)["Close"]
    sqqq = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)["Close"]
    return ndx, tqqq, sqqq


def compute_metrics(results):
    """Compute key metrics from engine results."""
    start_eq = results["Equity"].iloc[0]
    end_eq = results["Equity"].iloc[-1]
    years = (results.index[-1] - results.index[0]).days / 365.25
    total_ret = (end_eq / start_eq - 1) * 100
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
    max_dd = results["Drawdown_Pct"].min()
    avg_tqqq = results["Target_TQQQ_Pct"].mean()
    avg_sqqq = results["Target_SQQQ_Pct"].mean()

    # Per-rule P&L
    rule_pnl = {}
    entry_idx = 0
    cur_rule = results["Active_Rule"].iloc[0]
    trades = []
    for i in range(1, len(results)):
        r = results["Active_Rule"].iloc[i]
        if r != cur_rule:
            trades.append((entry_idx, i - 1, cur_rule))
            entry_idx = i
            cur_rule = r
    trades.append((entry_idx, len(results) - 1, cur_rule))

    for eidx, xidx, rule in trades:
        if rule in ("WHIPSAW", "CASH"):
            continue
        side = "long" if rule in ("L1", "L2", "L3", "L4") else "short"
        pnl = 0.0
        for j in range(eidx, xidx + 1):
            if j == 0:
                continue
            eq_prev = results["Equity"].iloc[j - 1]
            if eq_prev == 0:
                continue
            if side == "long":
                hp = results["Target_TQQQ_Pct"].iloc[j - 1]
                pc = results["TQQQ_Close"].iloc[j]
                pp = results["TQQQ_Close"].iloc[j - 1]
            else:
                hp = results["Target_SQQQ_Pct"].iloc[j - 1]
                pc = results["SQQQ_Close"].iloc[j]
                pp = results["SQQQ_Close"].iloc[j - 1]
            if pp > 0:
                pnl += hp / 100.0 * eq_prev * (pc / pp - 1)
        rule_pnl[rule] = rule_pnl.get(rule, 0.0) + pnl

    for k in rule_pnl:
        rule_pnl[k] = round(rule_pnl[k], 2)

    long_pnl = sum(v for k, v in rule_pnl.items() if k.startswith("L"))
    short_pnl = sum(v for k, v in rule_pnl.items() if k.startswith("S"))

    return {
        "total_ret": round(total_ret, 1),
        "cagr": round(cagr, 1),
        "max_dd": round(max_dd, 1),
        "avg_tqqq": round(avg_tqqq, 1),
        "avg_sqqq": round(avg_sqqq, 1),
        "end_equity": round(end_eq, 2),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "rule_pnl": rule_pnl,
    }


def build_scenarios():
    """Build all test scenarios. Each is (name, config)."""
    scenarios = []

    # 0. Baseline
    scenarios.append(("0. Baseline", DEFAULT_CONFIG.copy()))

    # SQQQ is the drag ===

    # 1. Kill S3 (cash in Regime D instead of shorting)
    cfg = DEFAULT_CONFIG.copy()
    cfg.kill_s3 = True
    scenarios.append(("1. Kill S3 (cash in D)", cfg))

    # 2. Reduce ALL short sizes by 50%
    cfg = DEFAULT_CONFIG.copy()
    cfg.s1_accel_size = 30.0
    cfg.s1_decel_stretched_size = 12.0
    cfg.s1_decel_normal_size = 20.0
    cfg.s2_early_expand_size = 35.0
    cfg.s2_early_noexpand_size = 25.0
    cfg.s2_late_expand_size = 30.0
    cfg.s2_late_noexpand_size = 25.0
    cfg.s3_bb_upper_override_size = 25.0
    cfg.s3_ext_high_decel_size = 20.0
    cfg.s3_ext_high_accel_size = 10.0
    cfg.s3_ext_low_decel_size = 15.0
    cfg.s3_ext_low_accel_size = 8.0
    scenarios.append(("2. All shorts 50%", cfg))

    # 3. Only short with S1 accel (most selective short)
    cfg = DEFAULT_CONFIG.copy()
    cfg.s1_decel_stretched_size = 0.0
    cfg.s1_decel_normal_size = 0.0
    cfg.s3_bb_upper_override_size = 0.0
    cfg.s3_ext_high_decel_size = 0.0
    cfg.s3_ext_high_accel_size = 0.0
    cfg.s3_ext_low_decel_size = 0.0
    cfg.s3_ext_low_accel_size = 0.0
    scenarios.append(("3. Short only S1 accel", cfg))

    # vol filter is the breakthrough ===

    # 4-6. Vol filter sweep (skip shorts when 20d realized vol is too high)
    for vol_thresh in [30.0, 35.0, 40.0]:
        cfg = DEFAULT_CONFIG.copy()
        cfg.vol_filter_short_threshold = vol_thresh
        scenarios.append((f"4. Vol filter <{vol_thresh:.0f}%", cfg))

    # 7. WINNER: Vol filter 35% + Kill S3
    cfg = DEFAULT_CONFIG.copy()
    cfg.kill_s3 = True
    cfg.vol_filter_short_threshold = 35.0
    scenarios.append(("5. Vol35 + Kill S3 ***", cfg))

    return scenarios


def main():
    print("Loading data...")
    ndx, tqqq, sqqq = load_data()

    scenarios = build_scenarios()
    results_list = []

    for name, cfg in scenarios:
        print(f"  Running: {name}...")
        engine = StrategyEngine(cfg)
        res = engine.run(ndx, tqqq, sqqq)
        metrics = compute_metrics(res)
        metrics["name"] = name
        results_list.append(metrics)

    # --- Print comparison table ---
    baseline = results_list[0]
    all_rules = sorted(set(r for m in results_list for r in m["rule_pnl"].keys()))

    print(f"\n{'='*100}")
    print("SHORT SIDE OPTIMIZATION RESULTS")
    print(f"{'='*100}")
    header = (f"{'Scenario':<28s} {'CAGR':>6s} {'MaxDD':>7s} {'TotRet':>8s} "
              f"{'EndEq':>9s} {'LongPnL':>8s} {'ShortPnL':>9s} {'AvgSQ':>6s}")
    print(header)
    print("-" * 100)

    for m in results_list:
        marker = ""
        if m["cagr"] > baseline["cagr"] and m["max_dd"] >= baseline["max_dd"]:
            marker = " ** WIN"
        elif m["cagr"] > baseline["cagr"]:
            marker = " *"

        row = (f"{m['name']:<28s} {m['cagr']:>5.1f}% {m['max_dd']:>6.1f}% "
               f"{m['total_ret']:>7.1f}% {m['end_equity']:>9.2f} "
               f"{m['long_pnl']:>8.1f} {m['short_pnl']:>9.1f} "
               f"{m['avg_sqqq']:>5.1f}%{marker}")
        print(row)

    # --- Per-rule P&L breakdown ---
    print(f"\n{'='*100}")
    print("PER-RULE P&L ($)")
    print(f"{'='*100}")
    header = f"{'Scenario':<28s}" + "".join(f" {r:>8s}" for r in all_rules) + f" {'TOTAL':>9s}"
    print(header)
    print("-" * 100)

    for m in results_list:
        total = sum(m["rule_pnl"].values())
        row = f"{m['name']:<28s}"
        for r in all_rules:
            val = m["rule_pnl"].get(r, 0.0)
            row += f" {val:>8.1f}"
        row += f" {total:>9.1f}"
        print(row)

    # --- Save CSV ---
    out_rows = []
    for m in results_list:
        row = {"Scenario": m["name"], "CAGR_%": m["cagr"], "Max_DD_%": m["max_dd"],
               "Total_Return_%": m["total_ret"], "End_Equity": m["end_equity"],
               "Avg_TQQQ_%": m["avg_tqqq"], "Avg_SQQQ_%": m["avg_sqqq"],
               "Long_PnL": m["long_pnl"], "Short_PnL": m["short_pnl"]}
        for r in all_rules:
            row[f"PnL_{r}"] = m["rule_pnl"].get(r, 0.0)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join(_SCRIPT_DIR, "sweep_v2_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


main()
