"""
Strategy Engine - Daily Decision Flow Orchestrator
Walks through historical data day-by-day, tracks all state, produces signals.
Per Strategy_Rules Section 9.
"""

import pandas as pd
import numpy as np
from config import StrategyConfig, DEFAULT_CONFIG
from indicators import compute_all_indicators
from regime import classify_regime
from rules import select_long_rule, size_long, select_short_rule, size_short
from adjustments import apply_all_adjustments


class StrategyEngine:
    def __init__(self, cfg: StrategyConfig = None):
        self.cfg = cfg or DEFAULT_CONFIG

    def run(self, ndx_close: pd.Series, tqqq_close: pd.Series,
            sqqq_close: pd.Series, tqqq_low: pd.Series = None,
            sqqq_low: pd.Series = None) -> pd.DataFrame:
        """Run the full strategy on historical data.

        Args:
            ndx_close: $NDX daily close (DatetimeIndex)
            tqqq_close: TQQQ daily close (DatetimeIndex)
            sqqq_close: SQQQ daily close (DatetimeIndex)
            tqqq_low: TQQQ daily low (optional, for intraday stop-loss)
            sqqq_low: SQQQ daily low (optional, for intraday stop-loss)

        Returns:
            DataFrame with daily signals, positions, and equity curve.
        """
        cfg = self.cfg

        # Compute indicators
        ind = compute_all_indicators(ndx_close, cfg)

        # Align all data to common dates
        common_idx = ind.index.intersection(tqqq_close.index).intersection(sqqq_close.index)
        ind = ind.loc[common_idx]
        tqqq = tqqq_close.loc[common_idx]
        sqqq = sqqq_close.loc[common_idx]
        tqqq_lo = tqqq_low.loc[common_idx] if tqqq_low is not None else None
        sqqq_lo = sqqq_low.loc[common_idx] if sqqq_low is not None else None

        # Skip warmup (need ma_long_period + slope lookback days)
        warmup = cfg.ma_long_period + cfg.ma_slope_lookback
        ind = ind.iloc[warmup:]
        tqqq = tqqq.iloc[warmup:]
        sqqq = sqqq.iloc[warmup:]
        if tqqq_lo is not None:
            tqqq_lo = tqqq_lo.iloc[warmup:]
        if sqqq_lo is not None:
            sqqq_lo = sqqq_lo.iloc[warmup:]

        n = len(ind)
        dates = ind.index

        # Output arrays
        out = {
            "Date": dates,
            "NDX_Close": ind["Close"].values,
            "TQQQ_Close": tqqq.values,
            "SQQQ_Close": sqqq.values,
            "Regime": [""] * n,
            "Active_Rule": [""] * n,
            "Base_Size": np.zeros(n),
            "Adj_Size": np.zeros(n),
            "Target_TQQQ_Pct": np.zeros(n),
            "Target_SQQQ_Pct": np.zeros(n),
            "Target_Cash_Pct": np.zeros(n),
            "Whipsaw_Active": [False] * n,
            "Stop_Triggered": [False] * n,
            "Risk_Mode": [""] * n,
            "Equity": np.zeros(n),
            "Drawdown_Pct": np.zeros(n),
        }

        # State variables
        prev_regime = None
        whipsaw_active = False
        whipsaw_confirm_count = 0
        whipsaw_side = None  # 'bull' or 'bear' - the new side we're waiting to confirm

        # S2 break tracking
        s2_break_day = None  # loop index of the break day
        s2_break_active = False
        prev_above_ma250 = None  # was previous close > MA_250?

        # Risk state
        equity = 100.0
        peak_equity = 100.0
        risk_halt = False
        reduce_only_until = -1     # day index
        consec_50pct_until = -1    # day index

        # Position tracking (actual dollar amounts, not target percentages)
        # This avoids equity drift when min trade threshold blocks rebalancing:
        # dollar positions naturally drift with returns, giving precise P&L.
        tqqq_dollars = 0.0
        sqqq_dollars = 0.0
        cash_dollars = 100.0

        for i in range(n):
            row = ind.iloc[i]
            row_dict = row.to_dict()

            # =============================================================
            # STEP 0.5: Intraday Stop-Loss Check (Section 8.6)
            # Before applying close-based returns, check if the day's low
            # would have breached the stop threshold. If so, exit at the
            # threshold level (not the low - assume fill at trigger price).
            # The EOD system then runs normally and may re-enter at close.
            # =============================================================
            stop_triggered_today = False
            if cfg.intraday_stop_enabled and i > 0:
                equity_at_open = tqqq_dollars + sqqq_dollars + cash_dollars
                if equity_at_open > 0:
                    worst_pnl_pct = 0.0
                    if tqqq_dollars > 0 and tqqq_lo is not None and tqqq.iloc[i - 1] > 0:
                        worst_ret = tqqq_lo.iloc[i] / tqqq.iloc[i - 1] - 1
                        worst_pnl_pct = (tqqq_dollars * worst_ret) / equity_at_open * 100
                    elif sqqq_dollars > 0 and sqqq_lo is not None and sqqq.iloc[i - 1] > 0:
                        worst_ret = sqqq_lo.iloc[i] / sqqq.iloc[i - 1] - 1
                        worst_pnl_pct = (sqqq_dollars * worst_ret) / equity_at_open * 100

                    if worst_pnl_pct < cfg.intraday_stop_threshold:
                        # Stop fires - exit at threshold level
                        equity = equity_at_open * (1 + cfg.intraday_stop_threshold / 100)
                        # Deduct tx cost for liquidation
                        held_pct = (tqqq_dollars + sqqq_dollars) / equity_at_open * 100
                        tx_cost = (held_pct / 100.0 * equity) * cfg.tx_cost_bps / 10000.0
                        equity -= tx_cost
                        tqqq_dollars = 0.0
                        sqqq_dollars = 0.0
                        cash_dollars = equity
                        stop_triggered_today = True

            # =============================================================
            # STEP 1: Update equity from YESTERDAY's held position
            # The position was set at yesterday's close; today's price
            # change determines P&L. This must happen FIRST, before any
            # signal logic, so that whipsaw/cross days capture their P&L.
            # Skip if intraday stop already fired (we exited mid-day).
            # =============================================================
            if i > 0 and not stop_triggered_today:
                tqqq_ret = (tqqq.iloc[i] / tqqq.iloc[i - 1] - 1) if tqqq.iloc[i - 1] > 0 else 0.0
                sqqq_ret = (sqqq.iloc[i] / sqqq.iloc[i - 1] - 1) if sqqq.iloc[i - 1] > 0 else 0.0

                # Apply returns to actual dollar positions (not percentages)
                tqqq_dollars *= (1 + tqqq_ret)
                sqqq_dollars *= (1 + sqqq_ret)
                # cash_dollars unchanged (no return on cash)
                equity = tqqq_dollars + sqqq_dollars + cash_dollars
                peak_equity = max(peak_equity, equity)

            # Compute actual held percentages from dollar positions
            # (these drift naturally with returns - more accurate than stale targets)
            held_tqqq_pct = (tqqq_dollars / equity * 100) if equity > 0 else 0.0
            held_sqqq_pct = (sqqq_dollars / equity * 100) if equity > 0 else 0.0

            # (Daily loss triggers are checked AFTER signal computation
            #  in Step 11b so they only affect the NEXT day per PDF 8.2)

            # =============================================================
            # STEP 2: Classify regime
            # =============================================================
            today_regime = classify_regime(row["Close"], row["MA_20"], row["MA_250"])
            currently_above_ma250 = row["Close"] > row["MA_250"]

            # =============================================================
            # STEP 3: S2 break tracking
            # Must happen BEFORE whipsaw continue so the break is recorded
            # even on the cross day. Per PDF Section 5.4: "The 10-day
            # window counts from the break date, including days spent in
            # the whipsaw filter."
            # =============================================================
            if prev_above_ma250 is not None:
                if prev_above_ma250 and not currently_above_ma250:
                    # Fresh break below MA_250
                    s2_break_day = i
                    s2_break_active = True

            if s2_break_active:
                days_since_break = i - s2_break_day
                if days_since_break > cfg.s2_break_window:
                    s2_break_active = False
                # If price reclaims MA_250 during S2 window, invalidate
                if currently_above_ma250 and s2_break_active:
                    s2_break_active = False
            else:
                days_since_break = 0

            # =============================================================
            # STEP 4: Whipsaw filter
            # Per PDF Section 6.3:
            # - Cross day: exit to cash immediately. Do NOT count as confirm.
            # - Days 1-2 after cross: waiting period, need 2 consecutive
            #   closes on new side.
            # - Day 3 (after 2 confirms): enter position.
            # =============================================================
            crossed_ma250 = False
            if prev_regime is not None:
                prev_bull = prev_regime in ("A", "B")
                curr_bull = today_regime in ("A", "B")
                if prev_bull != curr_bull:
                    crossed_ma250 = True

            if crossed_ma250 and not whipsaw_active:
                # MA_250 cross detected - exit to cash, start filter
                whipsaw_active = True
                whipsaw_confirm_count = 0
                whipsaw_side = "bull" if today_regime in ("A", "B") else "bear"
                # Cross day: hold cash, do NOT count as a confirm day
                # Deduct tx cost for selling held position to cash
                traded_pct = held_tqqq_pct + held_sqqq_pct
                tx_cost = (traded_pct / 100.0 * equity) * cfg.tx_cost_bps / 10000.0
                equity -= tx_cost
                out["Regime"][i] = today_regime
                out["Active_Rule"][i] = "WHIPSAW"
                out["Whipsaw_Active"][i] = True
                out["Target_TQQQ_Pct"][i] = 0.0
                out["Target_SQQQ_Pct"][i] = 0.0
                out["Target_Cash_Pct"][i] = 100.0
                out["Equity"][i] = equity
                dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
                out["Drawdown_Pct"][i] = dd_pct
                tqqq_dollars = 0.0
                sqqq_dollars = 0.0
                cash_dollars = equity
                prev_regime = today_regime
                prev_above_ma250 = currently_above_ma250
                continue

            if whipsaw_active:
                current_side = "bull" if today_regime in ("A", "B") else "bear"
                if current_side == whipsaw_side:
                    whipsaw_confirm_count += 1
                else:
                    # Crossed back - reset filter, track new side
                    whipsaw_side = current_side
                    whipsaw_confirm_count = 0  # don't count cross-back day

                if whipsaw_confirm_count > cfg.whipsaw_confirm_days:
                    # Filter clears: we had N consecutive confirms,
                    # now on the (N+1)th day we enter the new position.
                    whipsaw_active = False
                    # Fall through to normal signal computation below
                else:
                    # Still waiting - hold cash
                    out["Regime"][i] = today_regime
                    out["Active_Rule"][i] = "WHIPSAW"
                    out["Whipsaw_Active"][i] = True
                    out["Target_TQQQ_Pct"][i] = 0.0
                    out["Target_SQQQ_Pct"][i] = 0.0
                    out["Target_Cash_Pct"][i] = 100.0
                    out["Equity"][i] = equity
                    dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
                    out["Drawdown_Pct"][i] = dd_pct
                    tqqq_dollars = 0.0
                    sqqq_dollars = 0.0
                    cash_dollars = equity
                    prev_regime = today_regime
                    prev_above_ma250 = currently_above_ma250
                    continue

            # =============================================================
            # STEP 5: S2 eligibility check
            # =============================================================
            s2_eligible = (s2_break_active and
                           today_regime == "C" and
                           row["ROC_ACCEL"] < 0)

            # =============================================================
            # STEP 6: Rule selection & position sizing
            # =============================================================
            side = None
            rule = None
            base_size = 0.0

            if today_regime in ("A", "B"):
                side = "long"
                rule = select_long_rule(today_regime, row["Close"],
                                        row["BB_Lower"], row["EXT_20"], cfg)
                if rule:
                    base_size = size_long(rule, row_dict, cfg)
            elif today_regime in ("C", "D"):
                side = "short"
                rule = select_short_rule(today_regime, s2_eligible, cfg)
                if rule:
                    base_size = size_short(rule, row_dict, days_since_break,
                                           row["BB_Expansion"], cfg)

            # =============================================================
            # STEP 7: Post-rule adjustments (Section 7)
            # =============================================================
            if rule and base_size > 0:
                adj_size = apply_all_adjustments(
                    base_size, side, today_regime,
                    row["MA_250_Slope"],
                    row["Momentum_Bullish"],
                    row["Momentum_Bearish"],
                    cfg
                )
            else:
                adj_size = 0.0

            # =============================================================
            # STEP 8: Risk management (Section 8)
            # Per Section 8.4 precedence:
            #  - Halt overrides everything.
            #  - DD_REDUCE and CONSEC_REDUCE apply ONE 50% reduction
            #    (not stacked), with the longer duration.
            #  - Reduce-only stacks: apply 50% AND enforce no increases.
            # =============================================================
            risk_mode = ""
            dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0

            if dd_pct <= cfg.drawdown_halt:
                risk_halt = True

            if risk_halt:
                adj_size = 0.0
                risk_mode = "HALT"
            else:
                # DD_REDUCE and CONSEC_REDUCE: single 50% if EITHER active
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

                # Reduce-only mode (Section 8.2) stacks with above
                if i <= reduce_only_until:
                    if side == "long" and adj_size > held_tqqq_pct:
                        adj_size = held_tqqq_pct
                    elif side == "short" and adj_size > held_sqqq_pct:
                        adj_size = held_sqqq_pct
                    risk_mode = "REDUCE_ONLY" if not risk_mode else risk_mode + "+RO"

            # =============================================================
            # STEP 9: Apply caps & set targets
            # =============================================================
            if side == "long":
                adj_size = min(adj_size, cfg.max_tqqq)
                target_tqqq = adj_size
                target_sqqq = 0.0
            elif side == "short":
                adj_size = min(adj_size, cfg.max_sqqq)
                target_tqqq = 0.0
                target_sqqq = adj_size
            else:
                target_tqqq = 0.0
                target_sqqq = 0.0

            # =============================================================
            # STEP 10: Min trade threshold (Section 8.3)
            # Signal-driven rebalancing only; risk actions are exempt.
            # =============================================================
            if not risk_halt and not risk_mode:
                tqqq_delta = abs(target_tqqq - held_tqqq_pct)
                sqqq_delta = abs(target_sqqq - held_sqqq_pct)
                total_delta = tqqq_delta + sqqq_delta
                if total_delta < cfg.min_trade_threshold:
                    target_tqqq = held_tqqq_pct
                    target_sqqq = held_sqqq_pct

            target_cash = 100.0 - target_tqqq - target_sqqq

            # =============================================================
            # STEP 10b: Deduct transaction cost
            # Cost = traded_amount * bps. Applied at rebalance (close).
            # =============================================================
            traded_pct = abs(target_tqqq - held_tqqq_pct) + abs(target_sqqq - held_sqqq_pct)
            if traded_pct > 0 and cfg.tx_cost_bps > 0:
                tx_cost = (traded_pct / 100.0 * equity) * cfg.tx_cost_bps / 10000.0
                equity -= tx_cost

            # =============================================================
            # STEP 11: Store outputs
            # =============================================================
            out["Regime"][i] = today_regime
            out["Active_Rule"][i] = rule or "CASH"
            out["Base_Size"][i] = base_size
            out["Adj_Size"][i] = adj_size
            out["Target_TQQQ_Pct"][i] = target_tqqq
            out["Target_SQQQ_Pct"][i] = target_sqqq
            out["Target_Cash_Pct"][i] = target_cash
            out["Whipsaw_Active"][i] = False
            out["Stop_Triggered"][i] = stop_triggered_today
            out["Risk_Mode"][i] = risk_mode
            out["Equity"][i] = equity
            dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
            out["Drawdown_Pct"][i] = dd_pct

            # =============================================================
            # STEP 11b: Check daily loss triggers for FUTURE days
            # Per PDF 8.2: ">15% single-day loss -> reduce-only NEXT day"
            # and "2 consecutive >10% days -> 50% for next 5 days."
            # Detected here (after today's signal is finalized) so they
            # only affect tomorrow's signal, not today's.
            # =============================================================
            if i > 0 and out["Equity"][i - 1] > 0:
                daily_ret = (equity / out["Equity"][i - 1] - 1) * 100
                if daily_ret < cfg.daily_loss_reduce_only:
                    reduce_only_until = max(reduce_only_until, i + 1)
                if i >= 2 and out["Equity"][i - 2] > 0:
                    prev_daily_ret = (out["Equity"][i - 1] / out["Equity"][i - 2] - 1) * 100
                    if daily_ret < cfg.daily_loss_consecutive and prev_daily_ret < cfg.daily_loss_consecutive:
                        consec_50pct_until = max(consec_50pct_until, i + cfg.daily_loss_consec_duration)

            # =============================================================
            # STEP 12: Update dollar positions for next day's return calc
            # =============================================================
            tqqq_dollars = target_tqqq / 100.0 * equity
            sqqq_dollars = target_sqqq / 100.0 * equity
            cash_dollars = equity - tqqq_dollars - sqqq_dollars
            prev_regime = today_regime
            prev_above_ma250 = currently_above_ma250

        result = pd.DataFrame(out)
        result.set_index("Date", inplace=True)
        return result
