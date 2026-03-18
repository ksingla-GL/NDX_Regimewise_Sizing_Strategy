"""
Optimization Sweep V3 - Intraday stop-loss tuning.

Base: V2 winner (kill_s3 + vol_filter_35) + intraday SL -3%
      = 25.0% CAGR, -30.0% MaxDD, $2,838

Tests:
  1.  Kill L4 (cash when L4 triggers)
  2.  SL cooldown 1 day (no re-entry day of stop fire)
  3.  SL cooldown 2 days
  4.  Reduce L2 to 25%
  5.  Reduce L2 to 20%
  6.  Asymmetric SL: -3% long, -5% short
  7.  Asymmetric SL: -2% long, -3% short
  8.  Vol-scaled sizing (target 20% annualized)
  9.  Vol-scaled sizing (target 25% annualized)
  10. Kill L4 + cooldown 1d
  11. Kill L4 + cooldown 1d + reduce L2 25%
  12. Kill L4 + vol-scaled 20%
  13. Kill L4 + asymmetric SL (-2%/-3%)
  14. Best combo: kill L4 + cooldown 1d + vol-scaled 20% + asymmetric SL

Winner: Asymmetric SL (-2% TQQQ, -3% SQQQ) = 28.7% CAGR, -24.7% MaxDD.
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
from indicators import compute_all_indicators
from regime import classify_regime
from rules import select_long_rule, size_long, select_short_rule, size_short
from adjustments import apply_all_adjustments

INPUT_DIR = os.path.join(_PROJECT_ROOT, "Inputs")


def run_v3_sl_engine(cfg, ndx_close, tqqq_close, sqqq_close, tqqq_low, sqqq_low,
                  kill_l4=False, sl_cooldown_days=0, reduce_l2_to=None,
                  sl_long_threshold=None, sl_short_threshold=None,
                  vol_scale_target=None,
                  kill_s3=True, vol_filter_short=35.0, sl_threshold=-3.0):
    """Engine with V2 winner base + SL + all V3 structural overrides.

    Args:
        kill_l4: If True, go cash when L4 triggers.
        sl_cooldown_days: Days to stay cash after SL fires (0 = re-enter same day EOD).
        reduce_l2_to: Override L2 sizing to this % (None = use config).
        sl_long_threshold: SL threshold for TQQQ positions (None = use sl_threshold).
        sl_short_threshold: SL threshold for SQQQ positions (None = use sl_threshold).
        vol_scale_target: Target annualized vol % for position scaling (None = disabled).
        kill_s3: Cash in Regime D (V2 feature).
        vol_filter_short: Skip shorting if 20d vol > this (V2 feature, 0 = disabled).
        sl_threshold: Default SL threshold if asymmetric not set.
    """
    ind = compute_all_indicators(ndx_close, cfg)
    common_idx = ind.index.intersection(tqqq_close.index).intersection(sqqq_close.index)
    ind = ind.loc[common_idx]
    tqqq = tqqq_close.loc[common_idx]
    sqqq = sqqq_close.loc[common_idx]
    tqqq_lo = tqqq_low.loc[common_idx]
    sqqq_lo = sqqq_low.loc[common_idx]
    warmup = cfg.ma_long_period + cfg.ma_slope_lookback
    ind = ind.iloc[warmup:]
    tqqq = tqqq.iloc[warmup:]
    sqqq = sqqq.iloc[warmup:]
    tqqq_lo = tqqq_lo.iloc[warmup:]
    sqqq_lo = sqqq_lo.iloc[warmup:]
    n = len(ind)
    dates = ind.index

    # Resolve SL thresholds
    sl_long = sl_long_threshold if sl_long_threshold is not None else sl_threshold
    sl_short = sl_short_threshold if sl_short_threshold is not None else sl_threshold

    out = {
        "Date": dates, "NDX_Close": ind["Close"].values,
        "TQQQ_Close": tqqq.values, "SQQQ_Close": sqqq.values,
        "Regime": [""] * n, "Active_Rule": [""] * n,
        "Base_Size": np.zeros(n), "Adj_Size": np.zeros(n),
        "Target_TQQQ_Pct": np.zeros(n), "Target_SQQQ_Pct": np.zeros(n),
        "Target_Cash_Pct": np.zeros(n), "Whipsaw_Active": [False] * n,
        "Stop_Triggered": [False] * n,
        "Risk_Mode": [""] * n, "Equity": np.zeros(n),
        "Drawdown_Pct": np.zeros(n),
    }

    prev_regime = None
    whipsaw_active = False
    whipsaw_confirm_count = 0
    whipsaw_side = None
    s2_break_day = None
    s2_break_active = False
    prev_above_ma250 = None
    equity = 100.0
    peak_equity = 100.0
    risk_halt = False
    reduce_only_until = -1
    consec_50pct_until = -1
    tqqq_dollars = 0.0
    sqqq_dollars = 0.0
    cash_dollars = 100.0
    last_sl_fire_day = -999  # for cooldown tracking
    stop_count = 0

    for i in range(n):
        row = ind.iloc[i]
        row_dict = row.to_dict()

        # === STEP 0.5: Intraday Stop-Loss ===
        stop_triggered_today = False
        if i > 0:
            equity_at_open = tqqq_dollars + sqqq_dollars + cash_dollars
            if equity_at_open > 0:
                worst_pnl_pct = 0.0
                active_sl = 0.0
                if tqqq_dollars > 0 and tqqq.iloc[i-1] > 0:
                    worst_ret = tqqq_lo.iloc[i] / tqqq.iloc[i-1] - 1
                    worst_pnl_pct = (tqqq_dollars * worst_ret) / equity_at_open * 100
                    active_sl = sl_long
                elif sqqq_dollars > 0 and sqqq.iloc[i-1] > 0:
                    worst_ret = sqqq_lo.iloc[i] / sqqq.iloc[i-1] - 1
                    worst_pnl_pct = (sqqq_dollars * worst_ret) / equity_at_open * 100
                    active_sl = sl_short

                if active_sl < 0 and worst_pnl_pct < active_sl:
                    equity = equity_at_open * (1 + active_sl / 100)
                    held_pct = (tqqq_dollars + sqqq_dollars) / equity_at_open * 100
                    tx_cost = (held_pct / 100.0 * equity) * cfg.tx_cost_bps / 10000.0
                    equity -= tx_cost
                    tqqq_dollars = 0.0
                    sqqq_dollars = 0.0
                    cash_dollars = equity
                    stop_triggered_today = True
                    last_sl_fire_day = i
                    stop_count += 1

        # === STEP 1: Update equity ===
        if i > 0 and not stop_triggered_today:
            tqqq_ret = (tqqq.iloc[i] / tqqq.iloc[i-1] - 1) if tqqq.iloc[i-1] > 0 else 0.0
            sqqq_ret = (sqqq.iloc[i] / sqqq.iloc[i-1] - 1) if sqqq.iloc[i-1] > 0 else 0.0
            tqqq_dollars *= (1 + tqqq_ret)
            sqqq_dollars *= (1 + sqqq_ret)
            equity = tqqq_dollars + sqqq_dollars + cash_dollars
            peak_equity = max(peak_equity, equity)

        held_tqqq_pct = (tqqq_dollars / equity * 100) if equity > 0 else 0.0
        held_sqqq_pct = (sqqq_dollars / equity * 100) if equity > 0 else 0.0

        # === STEP 2: Regime ===
        today_regime = classify_regime(row["Close"], row["MA_20"], row["MA_250"])
        currently_above_ma250 = row["Close"] > row["MA_250"]

        # === S2 break tracking ===
        if prev_above_ma250 is not None:
            if prev_above_ma250 and not currently_above_ma250:
                s2_break_day = i
                s2_break_active = True
        if s2_break_active:
            days_since_break = i - s2_break_day
            if days_since_break > cfg.s2_break_window:
                s2_break_active = False
            if currently_above_ma250 and s2_break_active:
                s2_break_active = False
        else:
            days_since_break = 0

        # === Whipsaw ===
        crossed_ma250 = False
        if prev_regime is not None:
            prev_bull = prev_regime in ("A", "B")
            curr_bull = today_regime in ("A", "B")
            if prev_bull != curr_bull:
                crossed_ma250 = True

        if crossed_ma250 and not whipsaw_active:
            whipsaw_active = True
            whipsaw_confirm_count = 0
            whipsaw_side = "bull" if today_regime in ("A", "B") else "bear"
            traded_pct = held_tqqq_pct + held_sqqq_pct
            tx_cost = (traded_pct / 100.0 * equity) * cfg.tx_cost_bps / 10000.0
            equity -= tx_cost
            out["Regime"][i] = today_regime
            out["Active_Rule"][i] = "WHIPSAW"
            out["Whipsaw_Active"][i] = True
            out["Stop_Triggered"][i] = stop_triggered_today
            out["Target_Cash_Pct"][i] = 100.0
            out["Equity"][i] = equity
            dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
            out["Drawdown_Pct"][i] = dd_pct
            tqqq_dollars = 0.0; sqqq_dollars = 0.0; cash_dollars = equity
            prev_regime = today_regime; prev_above_ma250 = currently_above_ma250
            continue

        if whipsaw_active:
            current_side = "bull" if today_regime in ("A", "B") else "bear"
            if current_side == whipsaw_side:
                whipsaw_confirm_count += 1
            else:
                whipsaw_side = current_side
                whipsaw_confirm_count = 0
            if whipsaw_confirm_count > cfg.whipsaw_confirm_days:
                whipsaw_active = False
            else:
                out["Regime"][i] = today_regime
                out["Active_Rule"][i] = "WHIPSAW"
                out["Whipsaw_Active"][i] = True
                out["Stop_Triggered"][i] = stop_triggered_today
                out["Target_Cash_Pct"][i] = 100.0
                out["Equity"][i] = equity
                dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
                out["Drawdown_Pct"][i] = dd_pct
                tqqq_dollars = 0.0; sqqq_dollars = 0.0; cash_dollars = equity
                prev_regime = today_regime; prev_above_ma250 = currently_above_ma250
                continue

        s2_eligible = (s2_break_active and today_regime == "C" and row["ROC_ACCEL"] < 0)

        # Realized vol (20-day annualized)
        if i >= 20:
            recent_rets = ind["Close"].iloc[max(0, i-20):i].pct_change().dropna()
            realized_vol = recent_rets.std() * np.sqrt(252) * 100 if len(recent_rets) > 1 else 0.0
        else:
            realized_vol = 0.0

        # === STEP 6: Rule selection with overrides ===
        side = None; rule = None; base_size = 0.0

        # SL cooldown: force cash if within cooldown period
        in_sl_cooldown = (sl_cooldown_days > 0 and (i - last_sl_fire_day) <= sl_cooldown_days)

        if in_sl_cooldown:
            pass  # stay cash
        elif today_regime in ("A", "B"):
            side = "long"
            rule = select_long_rule(today_regime, row["Close"], row["BB_Lower"], row["EXT_20"], cfg)
            if rule:
                base_size = size_long(rule, row_dict, cfg)
            # Kill L4
            if kill_l4 and rule == "L4":
                side = None; rule = None; base_size = 0.0
            # Reduce L2
            if reduce_l2_to is not None and rule == "L2":
                base_size = reduce_l2_to
        elif today_regime == "D" and kill_s3:
            pass  # cash in Regime D
        elif today_regime in ("C", "D"):
            # Vol filter for shorts
            if vol_filter_short > 0 and today_regime == "C" and realized_vol > vol_filter_short:
                pass  # skip short
            else:
                side = "short"
                rule = select_short_rule(today_regime, s2_eligible, cfg)
                if rule:
                    base_size = size_short(rule, row_dict, days_since_break, row["BB_Expansion"], cfg)

        # Vol-scaled sizing
        if vol_scale_target is not None and base_size > 0 and realized_vol > 0:
            vol_scale = vol_scale_target / realized_vol
            vol_scale = max(0.25, min(vol_scale, 2.0))  # clamp [0.25x, 2.0x]
            base_size *= vol_scale

        if rule and base_size > 0:
            adj_size = apply_all_adjustments(base_size, side, today_regime,
                                              row["MA_250_Slope"], row["Momentum_Bullish"],
                                              row["Momentum_Bearish"], cfg)
        else:
            adj_size = 0.0

        # === Risk management ===
        risk_mode = ""
        dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        if dd_pct <= cfg.drawdown_halt:
            risk_halt = True
        if risk_halt:
            adj_size = 0.0; risk_mode = "HALT"
        else:
            dd_reduce_active = dd_pct <= cfg.drawdown_reduce
            consec_reduce_active = i <= consec_50pct_until
            if dd_reduce_active or consec_reduce_active:
                adj_size *= cfg.drawdown_reduce_mult
                if dd_reduce_active and consec_reduce_active:
                    risk_mode = "DD_REDUCE+CONSEC"
                elif dd_reduce_active:
                    risk_mode = "DD_REDUCE"
                else:
                    risk_mode = "CONSEC_REDUCE"
            if i <= reduce_only_until:
                if side == "long" and adj_size > held_tqqq_pct:
                    adj_size = held_tqqq_pct
                elif side == "short" and adj_size > held_sqqq_pct:
                    adj_size = held_sqqq_pct
                risk_mode = "REDUCE_ONLY" if not risk_mode else risk_mode + "+RO"

        if side == "long":
            adj_size = min(adj_size, cfg.max_tqqq)
            target_tqqq = adj_size; target_sqqq = 0.0
        elif side == "short":
            adj_size = min(adj_size, cfg.max_sqqq)
            target_tqqq = 0.0; target_sqqq = adj_size
        else:
            target_tqqq = 0.0; target_sqqq = 0.0

        if not risk_halt and not risk_mode:
            tqqq_delta = abs(target_tqqq - held_tqqq_pct)
            sqqq_delta = abs(target_sqqq - held_sqqq_pct)
            if tqqq_delta + sqqq_delta < cfg.min_trade_threshold:
                target_tqqq = held_tqqq_pct; target_sqqq = held_sqqq_pct

        target_cash = 100.0 - target_tqqq - target_sqqq
        traded_pct = abs(target_tqqq - held_tqqq_pct) + abs(target_sqqq - held_sqqq_pct)
        if traded_pct > 0 and cfg.tx_cost_bps > 0:
            tx_cost = (traded_pct / 100.0 * equity) * cfg.tx_cost_bps / 10000.0
            equity -= tx_cost

        out["Regime"][i] = today_regime
        out["Active_Rule"][i] = rule or "CASH"
        out["Base_Size"][i] = base_size; out["Adj_Size"][i] = adj_size
        out["Target_TQQQ_Pct"][i] = target_tqqq
        out["Target_SQQQ_Pct"][i] = target_sqqq
        out["Target_Cash_Pct"][i] = target_cash
        out["Whipsaw_Active"][i] = False
        out["Stop_Triggered"][i] = stop_triggered_today
        out["Risk_Mode"][i] = risk_mode
        out["Equity"][i] = equity
        dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
        out["Drawdown_Pct"][i] = dd_pct

        if i > 0 and out["Equity"][i-1] > 0:
            daily_ret = (equity / out["Equity"][i-1] - 1) * 100
            if daily_ret < cfg.daily_loss_reduce_only:
                reduce_only_until = max(reduce_only_until, i + 1)
            if i >= 2 and out["Equity"][i-2] > 0:
                prev_daily_ret = (out["Equity"][i-1] / out["Equity"][i-2] - 1) * 100
                if daily_ret < cfg.daily_loss_consecutive and prev_daily_ret < cfg.daily_loss_consecutive:
                    consec_50pct_until = max(consec_50pct_until, i + cfg.daily_loss_consec_duration)

        tqqq_dollars = target_tqqq / 100.0 * equity
        sqqq_dollars = target_sqqq / 100.0 * equity
        cash_dollars = equity - tqqq_dollars - sqqq_dollars
        prev_regime = today_regime; prev_above_ma250 = currently_above_ma250

    result = pd.DataFrame(out)
    result.set_index("Date", inplace=True)
    return result, stop_count


def compute_metrics(results):
    start_eq = results["Equity"].iloc[0]
    end_eq = results["Equity"].iloc[-1]
    years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100
    max_dd = results["Drawdown_Pct"].min()
    return cagr, max_dd, end_eq


def compute_rule_pnl(results):
    """Per-rule total P&L via daily attribution."""
    pnl = {}
    for i in range(1, len(results)):
        rule = results["Active_Rule"].iloc[i - 1]
        eq_prev = results["Equity"].iloc[i - 1]
        if eq_prev == 0:
            continue
        tqqq_pct = results["Target_TQQQ_Pct"].iloc[i - 1]
        if tqqq_pct > 0 and results["TQQQ_Close"].iloc[i - 1] > 0:
            ret = results["TQQQ_Close"].iloc[i] / results["TQQQ_Close"].iloc[i - 1] - 1
            pnl[rule] = pnl.get(rule, 0.0) + tqqq_pct / 100.0 * eq_prev * ret
        sqqq_pct = results["Target_SQQQ_Pct"].iloc[i - 1]
        if sqqq_pct > 0 and results["SQQQ_Close"].iloc[i - 1] > 0:
            ret = results["SQQQ_Close"].iloc[i] / results["SQQQ_Close"].iloc[i - 1] - 1
            pnl[rule] = pnl.get(rule, 0.0) + sqqq_pct / 100.0 * eq_prev * ret
    return pnl


if __name__ == "__main__":
    ndx = pd.read_csv(os.path.join(INPUT_DIR, "NDX.csv"), index_col=0, parse_dates=True)["Close"]
    tqqq_df = pd.read_csv(os.path.join(INPUT_DIR, "TQQQ.csv"), index_col=0, parse_dates=True)
    sqqq_df = pd.read_csv(os.path.join(INPUT_DIR, "SQQQ.csv"), index_col=0, parse_dates=True)
    tqqq = tqqq_df["Close"]
    sqqq = sqqq_df["Close"]
    tqqq_lo = tqqq_df["Low"]
    sqqq_lo = sqqq_df["Low"]
    cfg = DEFAULT_CONFIG.copy()

    scenarios = [
        ("0.  V2+SL-3% (base)",       {}),
        ("1.  + Kill L4",              {"kill_l4": True}),
        ("2.  + SL cooldown 1d",       {"sl_cooldown_days": 1}),
        ("3.  + SL cooldown 2d",       {"sl_cooldown_days": 2}),
        ("4.  + L2 -> 25%",            {"reduce_l2_to": 25.0}),
        ("5.  + L2 -> 20%",            {"reduce_l2_to": 20.0}),
        ("6.  Asym SL -3%L/-5%S",      {"sl_long_threshold": -3.0, "sl_short_threshold": -5.0}),
        ("7.  Asym SL -2%L/-3%S",      {"sl_long_threshold": -2.0, "sl_short_threshold": -3.0}),
        ("8.  Vol-scale tgt 20%",      {"vol_scale_target": 20.0}),
        ("9.  Vol-scale tgt 25%",      {"vol_scale_target": 25.0}),
        ("10. KillL4 + cool1d",        {"kill_l4": True, "sl_cooldown_days": 1}),
        ("11. KillL4 + cool1d + L2 25%", {"kill_l4": True, "sl_cooldown_days": 1, "reduce_l2_to": 25.0}),
        ("12. KillL4 + vol-scale 20%", {"kill_l4": True, "vol_scale_target": 20.0}),
        ("13. KillL4 + asym -2/-3",    {"kill_l4": True, "sl_long_threshold": -2.0, "sl_short_threshold": -3.0}),
        ("14. KillL4+cool1d+vol20+asym", {"kill_l4": True, "sl_cooldown_days": 1, "vol_scale_target": 20.0,
                                          "sl_long_threshold": -2.0, "sl_short_threshold": -3.0}),
    ]

    print(f"{'Scenario':<32s} {'CAGR':>6s} {'MaxDD':>7s} {'EndEq':>9s} {'Stops':>5s}  Rule PnL")
    print("=" * 110)

    rows = []
    for name, kwargs in scenarios:
        results, stops = run_v3_sl_engine(cfg, ndx, tqqq, sqqq, tqqq_lo, sqqq_lo, **kwargs)
        cagr, max_dd, end_eq = compute_metrics(results)
        pnl = compute_rule_pnl(results)

        pnl_parts = []
        for r in ["L1", "L2", "L3", "L4", "S1", "S2"]:
            if r in pnl:
                pnl_parts.append(f"{r}:{pnl[r]:+.0f}")
        pnl_str = "  ".join(pnl_parts)

        print(f"{name:<32s} {cagr:>5.1f}% {max_dd:>6.1f}% {end_eq:>9.2f} {stops:>5d}  {pnl_str}")

        row = {"Scenario": name, "CAGR_%": round(cagr, 1), "Max_DD_%": round(max_dd, 1),
               "End_Equity": round(end_eq, 2), "Stops": stops}
        for r in ["L1", "L2", "L3", "L4", "S1", "S2", "S3"]:
            row[f"PnL_{r}"] = round(pnl.get(r, 0.0), 2)
        rows.append(row)

    # Save results
    df = pd.DataFrame(rows)
    out_path = os.path.join(_SCRIPT_DIR, "sweep_v3_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
