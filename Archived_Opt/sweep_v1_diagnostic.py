"""
Optimization Sweep V1 - Broad diagnostic.

Tests each proposed change in isolation and combined.
Outputs a comparison table of CAGR, Max DD, Total Return, and per-rule P&L.

Key finding: L1 is the only profitable rule. Whipsaw=1 is catastrophic.
Stacking unvalidated changes (scenario 7) destroyed performance.
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

    # Win/loss from rule changes
    trades = extract_positioned_trades(results)
    n_trades = len(trades)
    winners = (trades["pnl"] > 0).sum() if n_trades > 0 else 0
    win_rate = winners / n_trades * 100 if n_trades > 0 else 0

    # Per-rule P&L
    rule_pnl = {}
    for rule, grp in trades.groupby("rule"):
        rule_pnl[rule] = round(grp["pnl"].sum(), 2)

    avg_tqqq = results["Target_TQQQ_Pct"].mean()
    avg_sqqq = results["Target_SQQQ_Pct"].mean()

    return {
        "total_ret": round(total_ret, 1),
        "cagr": round(cagr, 1),
        "max_dd": round(max_dd, 1),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "avg_tqqq": round(avg_tqqq, 1),
        "avg_sqqq": round(avg_sqqq, 1),
        "rule_pnl": rule_pnl,
        "end_equity": round(end_eq, 2),
    }


def extract_positioned_trades(results):
    """Quick trade extraction - just rule changes with P&L."""
    trades = []
    entry_idx = 0
    cur_rule = results["Active_Rule"].iloc[0]

    for i in range(1, len(results)):
        rule = results["Active_Rule"].iloc[i]
        if rule != cur_rule:
            trades.append(_trade_pnl(results, entry_idx, i - 1, cur_rule))
            entry_idx = i
            cur_rule = rule
    trades.append(_trade_pnl(results, entry_idx, len(results) - 1, cur_rule))

    df = pd.DataFrame(trades)
    # Filter to positioned trades only
    return df[df["side"] != "cash"].reset_index(drop=True)


def _trade_pnl(results, entry_idx, exit_idx, rule):
    if rule in ("L1", "L2", "L3", "L4"):
        side = "long"
    elif rule in ("S1", "S2", "S3"):
        side = "short"
    else:
        side = "cash"

    entry_eq = results["Equity"].iloc[entry_idx]
    exit_eq = results["Equity"].iloc[exit_idx]

    # Approximate P&L from equity change attributed to this position
    pnl = 0.0
    for j in range(entry_idx, exit_idx + 1):
        if j == 0:
            continue
        eq_prev = results["Equity"].iloc[j - 1]
        if eq_prev == 0:
            continue
        if side == "long":
            held_pct = results["Target_TQQQ_Pct"].iloc[j - 1]
            p_cur = results["TQQQ_Close"].iloc[j]
            p_prev = results["TQQQ_Close"].iloc[j - 1]
        elif side == "short":
            held_pct = results["Target_SQQQ_Pct"].iloc[j - 1]
            p_cur = results["SQQQ_Close"].iloc[j]
            p_prev = results["SQQQ_Close"].iloc[j - 1]
        else:
            continue
        if p_prev > 0:
            pnl += held_pct / 100.0 * eq_prev * (p_cur / p_prev - 1)

    return {"rule": rule, "side": side, "pnl": round(pnl, 2)}


def run_scenario(name, cfg, ndx, tqqq, sqqq):
    """Run engine and return metrics dict."""
    engine = StrategyEngine(cfg)
    results = engine.run(ndx, tqqq, sqqq)
    metrics = compute_metrics(results)
    metrics["name"] = name
    return metrics


def main():
    print("Loading data...")
    ndx, tqqq, sqqq = load_data()

    scenarios = []

    # =========================================================
    # 0. BASELINE (default config)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    scenarios.append(("0. Baseline", cfg))

    # =========================================================
    # 1. Kill L4 (set L4 sizes to match L3 - effectively L3 handles that zone)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.l4_fresh_size = 0.0
    cfg.l4_extended_size = 0.0
    scenarios.append(("1. Remove L4", cfg))

    # =========================================================
    # 2. Reduce L2 sizing (50/40 -> 30/25)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.l2_extended_size = 30.0
    cfg.l2_normal_size = 25.0
    scenarios.append(("2. Reduce L2 sizing", cfg))

    # =========================================================
    # 3. Shrink S3 sizes (halved)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.s3_bb_upper_override_size = 25.0
    cfg.s3_ext_high_decel_size = 20.0
    cfg.s3_ext_high_accel_size = 10.0
    cfg.s3_ext_low_decel_size = 15.0
    cfg.s3_ext_low_accel_size = 8.0
    scenarios.append(("3. Shrink S3 (half)", cfg))

    # =========================================================
    # 4. L1 more aggressive (accel 100%, decel_ext 60%, decel_norm 80%)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.l1_accel_size = 100.0
    cfg.l1_decel_extended_size = 60.0
    cfg.l1_decel_normal_size = 80.0
    scenarios.append(("4. Boost L1", cfg))

    # =========================================================
    # 5. Whipsaw 1-day confirm instead of 2
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.whipsaw_confirm_days = 1
    scenarios.append(("5. Whipsaw 1-day", cfg))

    # =========================================================
    # 6. Soften trend health (0.75 -> 0.85)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.trend_health_weak_mult = 0.85
    scenarios.append(("6. Trend health 0.85x", cfg))

    # =========================================================
    # 7. COMBINED: all changes together
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.l4_fresh_size = 0.0
    cfg.l4_extended_size = 0.0
    cfg.l2_extended_size = 30.0
    cfg.l2_normal_size = 25.0
    cfg.s3_bb_upper_override_size = 25.0
    cfg.s3_ext_high_decel_size = 20.0
    cfg.s3_ext_high_accel_size = 10.0
    cfg.s3_ext_low_decel_size = 15.0
    cfg.s3_ext_low_accel_size = 8.0
    cfg.l1_accel_size = 100.0
    cfg.l1_decel_extended_size = 60.0
    cfg.l1_decel_normal_size = 80.0
    cfg.whipsaw_confirm_days = 1
    cfg.trend_health_weak_mult = 0.85
    scenarios.append(("7. ALL COMBINED", cfg))

    # =========================================================
    # 8. COMBINED minus whipsaw change (in case whipsaw hurts)
    # =========================================================
    cfg = DEFAULT_CONFIG.copy()
    cfg.l4_fresh_size = 0.0
    cfg.l4_extended_size = 0.0
    cfg.l2_extended_size = 30.0
    cfg.l2_normal_size = 25.0
    cfg.s3_bb_upper_override_size = 25.0
    cfg.s3_ext_high_decel_size = 20.0
    cfg.s3_ext_high_accel_size = 10.0
    cfg.s3_ext_low_decel_size = 15.0
    cfg.s3_ext_low_accel_size = 8.0
    cfg.l1_accel_size = 100.0
    cfg.l1_decel_extended_size = 60.0
    cfg.l1_decel_normal_size = 80.0
    cfg.trend_health_weak_mult = 0.85
    scenarios.append(("8. COMBINED (keep 2d whip)", cfg))

    # =========================================================
    # Run all scenarios
    # =========================================================
    results_list = []
    for name, cfg in scenarios:
        print(f"  Running: {name}...")
        metrics = run_scenario(name, cfg, ndx, tqqq, sqqq)
        results_list.append(metrics)

    # =========================================================
    # Print comparison table
    # =========================================================
    print(f"\n{'='*100}")
    print("OPTIMIZATION RESULTS COMPARISON")
    print(f"{'='*100}")

    header = f"{'Scenario':<30s} {'CAGR':>6s} {'MaxDD':>7s} {'TotRet':>8s} {'EndEq':>9s} {'Trades':>6s} {'WinR':>5s} {'AvgTQ':>6s} {'AvgSQ':>6s}"
    print(header)
    print("-" * 100)

    for m in results_list:
        row = (f"{m['name']:<30s} {m['cagr']:>5.1f}% {m['max_dd']:>6.1f}% "
               f"{m['total_ret']:>7.1f}% {m['end_equity']:>9.2f} "
               f"{m['n_trades']:>6d} {m['win_rate']:>4.1f}% "
               f"{m['avg_tqqq']:>5.1f}% {m['avg_sqqq']:>5.1f}%")
        print(row)

    # Per-rule P&L comparison
    all_rules = sorted(set(r for m in results_list for r in m["rule_pnl"].keys()))
    print(f"\n{'='*100}")
    print("PER-RULE P&L ($)")
    print(f"{'='*100}")

    header = f"{'Scenario':<30s}" + "".join(f" {r:>8s}" for r in all_rules) + f" {'TOTAL':>9s}"
    print(header)
    print("-" * 100)

    for m in results_list:
        total = sum(m["rule_pnl"].values())
        row = f"{m['name']:<30s}"
        for r in all_rules:
            val = m["rule_pnl"].get(r, 0.0)
            row += f" {val:>8.1f}"
        row += f" {total:>9.1f}"
        print(row)

    # Save to CSV
    out_rows = []
    for m in results_list:
        row = {
            "Scenario": m["name"],
            "CAGR_%": m["cagr"],
            "Max_DD_%": m["max_dd"],
            "Total_Return_%": m["total_ret"],
            "End_Equity": m["end_equity"],
            "Trades": m["n_trades"],
            "Win_Rate_%": m["win_rate"],
            "Avg_TQQQ_%": m["avg_tqqq"],
            "Avg_SQQQ_%": m["avg_sqqq"],
        }
        for r in all_rules:
            row[f"PnL_{r}"] = m["rule_pnl"].get(r, 0.0)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join(_SCRIPT_DIR, "sweep_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
