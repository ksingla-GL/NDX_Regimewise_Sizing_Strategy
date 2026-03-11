"""
Quick backtest runner - loads data, runs engine, prints summary + per-rule P&L.
"""

import pandas as pd
import numpy as np
import os
from config import DEFAULT_CONFIG
from engine import StrategyEngine

INPUT_DIR = os.path.join(os.path.dirname(__file__), "Inputs")


def load_data():
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)
    tqqq = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)
    sqqq = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)
    return ndx["Close"], tqqq["Close"], sqqq["Close"], tqqq["Low"], sqqq["Low"]


def print_summary(results: pd.DataFrame):
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Period       : {results.index[0].date()} to {results.index[-1].date()}")
    print(f"Trading days : {len(results)}")

    start_eq = results["Equity"].iloc[0]
    end_eq = results["Equity"].iloc[-1]
    total_ret = (end_eq / start_eq - 1) * 100
    years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
    max_dd = results["Drawdown_Pct"].min()

    print(f"\nStart equity : {start_eq:.2f}")
    print(f"End equity   : {end_eq:.2f}")
    print(f"Total return : {total_ret:.1f}%")
    print(f"CAGR         : {cagr:.1f}%")
    print(f"Max Drawdown : {max_dd:.1f}%")

    print(f"\nRegime Distribution:")
    for r, c in results["Regime"].value_counts().sort_index().items():
        print(f"  {r}: {c} days ({c/len(results)*100:.1f}%)")

    print(f"\nActive Rule Distribution:")
    for r, c in results["Active_Rule"].value_counts().sort_index().items():
        print(f"  {r}: {c} days ({c/len(results)*100:.1f}%)")

    ws_days = results["Whipsaw_Active"].sum()
    risk_days = (results["Risk_Mode"] != "").sum()
    print(f"\nWhipsaw cash days: {ws_days}")
    print(f"Risk mode days   : {risk_days}")
    if "Stop_Triggered" in results.columns:
        print(f"Stop-loss fires  : {results['Stop_Triggered'].sum()}")

    print(f"\nAvg TQQQ allocation: {results['Target_TQQQ_Pct'].mean():.1f}%")
    print(f"Avg SQQQ allocation: {results['Target_SQQQ_Pct'].mean():.1f}%")
    print(f"Avg Cash allocation: {results['Target_Cash_Pct'].mean():.1f}%")


def compute_rule_pnl(results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-rule P&L from daily results.

    For each day, attributes the dollar P&L to the active rule based on
    held position and instrument return. Aggregates by rule.
    """
    n = len(results)
    daily_pnl = []

    for i in range(1, n):
        rule = results["Active_Rule"].iloc[i - 1]  # rule that set yesterday's position
        eq_prev = results["Equity"].iloc[i - 1]
        if eq_prev == 0:
            continue

        # Long side P&L
        tqqq_pct = results["Target_TQQQ_Pct"].iloc[i - 1]
        if tqqq_pct > 0 and results["TQQQ_Close"].iloc[i - 1] > 0:
            tqqq_ret = results["TQQQ_Close"].iloc[i] / results["TQQQ_Close"].iloc[i - 1] - 1
            pnl = tqqq_pct / 100.0 * eq_prev * tqqq_ret
            daily_pnl.append({"Rule": rule, "PnL": pnl})

        # Short side P&L
        sqqq_pct = results["Target_SQQQ_Pct"].iloc[i - 1]
        if sqqq_pct > 0 and results["SQQQ_Close"].iloc[i - 1] > 0:
            sqqq_ret = results["SQQQ_Close"].iloc[i] / results["SQQQ_Close"].iloc[i - 1] - 1
            pnl = sqqq_pct / 100.0 * eq_prev * sqqq_ret
            daily_pnl.append({"Rule": rule, "PnL": pnl})

    df = pd.DataFrame(daily_pnl)
    if len(df) == 0:
        return pd.DataFrame()

    summary = df.groupby("Rule").agg(
        Days=("PnL", "count"),
        Total_PnL=("PnL", "sum"),
        Avg_PnL=("PnL", "mean"),
        Win_Days=("PnL", lambda x: (x > 0).sum()),
    )
    summary["Loss_Days"] = summary["Days"] - summary["Win_Days"]
    summary["Win_Rate"] = (summary["Win_Days"] / summary["Days"] * 100).round(1)
    return summary.sort_index()


if __name__ == "__main__":
    print("Loading data...")
    ndx_close, tqqq_close, sqqq_close, tqqq_low, sqqq_low = load_data()

    print("Running backtest with default config...")
    engine = StrategyEngine(DEFAULT_CONFIG)
    results = engine.run(ndx_close, tqqq_close, sqqq_close, tqqq_low, sqqq_low)

    print_summary(results)

    # Save full results
    out_path = os.path.join(INPUT_DIR, "backtest_results.csv")
    results.to_csv(out_path)
    print(f"\nFull results saved to {out_path}")

    # Per-rule P&L breakdown
    rule_pnl = compute_rule_pnl(results)
    if len(rule_pnl) > 0:
        print(f"\n{'='*60}")
        print("PnL BY RULE (daily attribution)")
        print(f"{'='*60}")
        for rule, row in rule_pnl.iterrows():
            print(f"  {rule:8s}: {row['Days']:4.0f} days, "
                  f"PnL ${row['Total_PnL']:>9.2f}, "
                  f"avg ${row['Avg_PnL']:>+7.2f}/day, "
                  f"WR {row['Win_Rate']:.0f}% ({row['Win_Days']:.0f}W/{row['Loss_Days']:.0f}L)")
        print(f"  {'TOTAL':8s}: PnL ${rule_pnl['Total_PnL'].sum():>9.2f}")
