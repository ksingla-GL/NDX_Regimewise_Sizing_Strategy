"""
Unit tests for the strategy decision tree.
Each test constructs synthetic NDX/TQQQ/SQQQ data targeting a specific
leaf of the decision tree, runs the engine, and verifies the output.
"""

import unittest
import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG
from engine import StrategyEngine
from indicators import compute_all_indicators
from rules import select_long_rule, size_long, select_short_rule, size_short
from adjustments import apply_all_adjustments


# ===================================================================
# Helpers
# ===================================================================

def _bdays(n: int) -> pd.DatetimeIndex:
    """Generate n business days ending on a recent date."""
    return pd.bdate_range(end="2026-02-27", periods=n)


def _simple_etfs(index, tqqq_vals=None, sqqq_vals=None):
    """Return TQQQ / SQQQ series on *index*.  Defaults: gentle linear."""
    n = len(index)
    tqqq = pd.Series(
        tqqq_vals if tqqq_vals is not None else np.linspace(30, 50, n),
        index=index,
    )
    sqqq = pd.Series(
        sqqq_vals if sqqq_vals is not None else np.linspace(40, 20, n),
        index=index,
    )
    return tqqq, sqqq


WARMUP = DEFAULT_CONFIG.ma_long_period + DEFAULT_CONFIG.ma_slope_lookback  # 270


# ===================================================================
# Early Exit 1 — Data Unavailable
# ===================================================================
class TestEarlyExit_DataUnavailable(unittest.TestCase):
    """Leaf 1: Data unavailable → hold current positions.

    This guard exists in the live-trading flow only (Section 9).
    The backtest engine always has data, so this leaf is not reachable
    in backtesting.  Kept as a skip-marker for completeness.
    """

    @unittest.skip("Live-trading guard; not implemented in backtest engine.")
    def test_placeholder(self):
        pass


# ===================================================================
# Early Exit 2 — Whipsaw Filter (waiting period)
# ===================================================================
class TestEarlyExit_WhipsawWaiting(unittest.TestCase):
    """Leaf 2: Whipsaw filter active (waiting period) → 100 % cash.

    Data: steady uptrend for 370 days, then sharp drop below MA_250.
    On the day AFTER the cross the system is still in the 2-day
    confirmation window → must hold 100 % cash.
    """

    @classmethod
    def _build_data(cls):
        n = 400
        dates = _bdays(n)
        crash_idx = 370                       # last uptrend index

        prices = np.empty(n)
        prices[0] = 10_000.0
        for i in range(1, crash_idx + 1):
            prices[i] = prices[i - 1] * 1.0003
        for i in range(crash_idx + 1, n):     # flat below MA_250
            prices[i] = 10_400.0

        ndx = pd.Series(prices, index=dates)
        tqqq, sqqq = _simple_etfs(dates)
        return ndx, tqqq, sqqq, crash_idx

    def test_waiting_day_is_cash(self):
        ndx, tqqq, sqqq, crash_idx = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        cross_result_idx = crash_idx + 1 - WARMUP  # cross day in results
        waiting_idx = cross_result_idx + 1          # first waiting day

        row = results.iloc[waiting_idx]
        self.assertTrue(bool(row["Whipsaw_Active"]))
        self.assertEqual(row["Active_Rule"], "WHIPSAW")
        self.assertEqual(row["Target_TQQQ_Pct"], 0.0)
        self.assertEqual(row["Target_SQQQ_Pct"], 0.0)
        self.assertAlmostEqual(row["Target_Cash_Pct"], 100.0)

    def test_second_waiting_day_is_cash(self):
        """Confirm the SECOND waiting day is also cash (need >2 confirms)."""
        ndx, tqqq, sqqq, crash_idx = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        cross_result_idx = crash_idx + 1 - WARMUP
        waiting_idx_2 = cross_result_idx + 2

        row = results.iloc[waiting_idx_2]
        self.assertTrue(bool(row["Whipsaw_Active"]))
        self.assertEqual(row["Active_Rule"], "WHIPSAW")
        self.assertEqual(row["Target_TQQQ_Pct"], 0.0)
        self.assertAlmostEqual(row["Target_Cash_Pct"], 100.0)


# ===================================================================
# Early Exit 3 — MA_250 Cross Detected (cross day itself)
# ===================================================================
class TestEarlyExit_MA250Cross(unittest.TestCase):
    """Leaf 3: Close crosses MA_250 → exit to cash, enter whipsaw.

    Same data as whipsaw test; assertions target the CROSS day.
    """

    @classmethod
    def _build_data(cls):
        n = 400
        dates = _bdays(n)
        crash_idx = 370

        prices = np.empty(n)
        prices[0] = 10_000.0
        for i in range(1, crash_idx + 1):
            prices[i] = prices[i - 1] * 1.0003
        for i in range(crash_idx + 1, n):
            prices[i] = 10_400.0

        ndx = pd.Series(prices, index=dates)
        tqqq, sqqq = _simple_etfs(dates)
        return ndx, tqqq, sqqq, crash_idx

    def test_cross_day_exits_to_cash(self):
        ndx, tqqq, sqqq, crash_idx = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        cross_result_idx = crash_idx + 1 - WARMUP

        # Day before cross must be bull regime (A or B)
        pre = results.iloc[cross_result_idx - 1]
        self.assertIn(pre["Regime"], ("A", "B"))
        self.assertGreater(pre["Target_TQQQ_Pct"], 0,
                           "Should have been holding TQQQ before cross")

        # Cross day itself
        row = results.iloc[cross_result_idx]
        self.assertTrue(bool(row["Whipsaw_Active"]))
        self.assertEqual(row["Active_Rule"], "WHIPSAW")
        self.assertEqual(row["Target_TQQQ_Pct"], 0.0)
        self.assertEqual(row["Target_SQQQ_Pct"], 0.0)
        self.assertAlmostEqual(row["Target_Cash_Pct"], 100.0)

    def test_regime_changes_on_cross_day(self):
        """Verify regime flips from bull to bear on the cross day."""
        ndx, tqqq, sqqq, crash_idx = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        cross_result_idx = crash_idx + 1 - WARMUP

        pre_regime = results.iloc[cross_result_idx - 1]["Regime"]
        post_regime = results.iloc[cross_result_idx]["Regime"]

        self.assertIn(pre_regime, ("A", "B"))
        self.assertIn(post_regime, ("C", "D"))


# ===================================================================
# Early Exit 4 — Circuit Breaker HALT (≥ 50 % drawdown)
# ===================================================================
class TestEarlyExit_CircuitBreakerHalt(unittest.TestCase):
    """Leaf 4: Portfolio drawdown ≥ 50 % from peak → HALT, 100 % cash.

    Data: NDX stays in a steady uptrend (Regime A throughout, no
    MA_250 cross), so no whipsaw is triggered.  TQQQ crashes 90 %
    on a single day, causing > 50 % portfolio drawdown → HALT.
    """

    @classmethod
    def _build_data(cls):
        n = 400
        dates = _bdays(n)
        crash_day = 350                        # index in the prices array

        # NDX: steady uptrend (keeps Regime A, no cross)
        ndx_prices = np.empty(n)
        ndx_prices[0] = 10_000.0
        for i in range(1, n):
            ndx_prices[i] = ndx_prices[i - 1] * 1.0003

        # TQQQ: rises gently, then crashes 90 % on one day
        tqqq_prices = np.empty(n)
        tqqq_prices[0] = 30.0
        for i in range(1, n):
            if i == crash_day:
                tqqq_prices[i] = tqqq_prices[i - 1] * 0.10   # −90 %
            else:
                tqqq_prices[i] = tqqq_prices[i - 1] * 1.001

        ndx = pd.Series(ndx_prices, index=dates)
        tqqq = pd.Series(tqqq_prices, index=dates)
        sqqq = pd.Series(np.linspace(40, 20, n), index=dates)
        return ndx, tqqq, sqqq, crash_day

    def test_halt_triggers_on_crash_day(self):
        ndx, tqqq, sqqq, crash_day = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        halt_result_idx = crash_day - WARMUP   # day 350 → index 80

        row = results.iloc[halt_result_idx]
        self.assertEqual(row["Risk_Mode"], "HALT")
        self.assertEqual(row["Target_TQQQ_Pct"], 0.0)
        self.assertEqual(row["Target_SQQQ_Pct"], 0.0)
        self.assertAlmostEqual(row["Target_Cash_Pct"], 100.0)

    def test_halt_persists_next_day(self):
        """Once HALT fires it stays on — next day must also be 100 % cash."""
        ndx, tqqq, sqqq, crash_day = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        halt_result_idx = crash_day - WARMUP
        next_row = results.iloc[halt_result_idx + 1]

        self.assertEqual(next_row["Risk_Mode"], "HALT")
        self.assertEqual(next_row["Target_TQQQ_Pct"], 0.0)
        self.assertAlmostEqual(next_row["Target_Cash_Pct"], 100.0)

    def test_was_invested_before_halt(self):
        """Sanity: system must have been holding TQQQ before the crash."""
        ndx, tqqq, sqqq, crash_day = self._build_data()
        cfg = DEFAULT_CONFIG.copy()
        results = StrategyEngine(cfg).run(ndx, tqqq, sqqq)

        pre_idx = crash_day - WARMUP - 1
        self.assertGreater(results.iloc[pre_idx]["Target_TQQQ_Pct"], 0)


# ===================================================================
# All 84 sizing leaves  (21 base × 4 adjustment combos)
#
# Adjustment combos per Section 7:
#   H+NM  = healthy trend (1.0×) + no momentum (1.0×)
#   H+M   = healthy trend (1.0×) + momentum   (1.25×)
#   W+NM  = weak trend   (0.75×) + no momentum (1.0×)
#   W+M   = weak trend   (0.75×) + momentum   (1.25×)
#
# NOTE (S1 rows 2-3):  PDF states ROC_ACCEL ≤ 0 for rows 2-3, but
# after first-match row 1 (ROC_ACCEL < 0) only ROC_ACCEL = 0 would
# remain.  Code correctly treats rows 2-3 as the ROC_ACCEL ≥ 0
# partition.  See PDF audit for details.
# ===================================================================

# ---------- leaf definitions ----------
# Long: (label, regime, close, bb_lower, ext_20, row_for_sizing, rule, base)
_LONG_LEAVES = [
    ("L1r1_accel",         "A", 11000, 10800,  1.0, {"ROC_ACCEL":  0.5, "EXT_20":  1.0},          "L1", 80.0),
    ("L1r2_decel_ext",     "A", 11000, 10800,  6.0, {"ROC_ACCEL": -0.5, "EXT_20":  6.0},          "L1", 40.0),
    ("L1r3_decel_normal",  "A", 11000, 10800,  2.0, {"ROC_ACCEL": -0.5, "EXT_20":  2.0},          "L1", 60.0),
    ("L2r1_ext250_high",   "B", 10900, 10800, -1.0, {"EXT_250": 6.0},                              "L2", 50.0),
    ("L2r2_ext250_low",    "B", 10900, 10800, -1.0, {"EXT_250": 3.0},                              "L2", 40.0),
    ("L3r1_moderate",      "B", 10900, 10800, -4.0, {"EXT_20": -4.0},                              "L3", 25.0),
    ("L3r2_extreme",       "B", 10900, 10800, -7.0, {"EXT_20": -7.0},                              "L3", 10.0),
    ("L4r1_fresh",         "B", 10700, 10800, -4.0, {"Days_Below_BB_Lower": 1},                    "L4", 20.0),
    ("L4r2_extended",      "B", 10700, 10800, -4.0, {"Days_Below_BB_Lower": 4},                    "L4", 10.0),
]

# Short: (label, regime, s2_active, row_for_sizing, s2_days, bb_exp, rule, base)
_SHORT_LEAVES = [
    ("S1r1_accel_down",       "C", False, {"ROC_ACCEL": -0.5, "EXT_20": -2.0},                                     0, False, "S1", 60.0),
    ("S1r2_decel_stretched",  "C", False, {"ROC_ACCEL":  0.5, "EXT_20": -6.0},                                     0, False, "S1", 25.0),
    ("S1r3_decel_normal",     "C", False, {"ROC_ACCEL":  0.5, "EXT_20": -2.0},                                     0, False, "S1", 40.0),
    ("S2r1_early_expand",     "C", True,  {},                                                                        3, True,  "S2", 70.0),
    ("S2r2_early_noexpand",   "C", True,  {},                                                                        3, False, "S2", 50.0),
    ("S2r3_late_expand",      "C", True,  {},                                                                        7, True,  "S2", 60.0),
    ("S2r4_late_noexpand",    "C", True,  {},                                                                        7, False, "S2", 50.0),
    ("S3r1_bb_upper",         "D", False, {"Close": 11000, "BB_Upper": 10900, "EXT_20": 1.0, "ROC_ACCEL":  0.5},   0, False, "S3", 50.0),
    ("S3r2_ext_high_decel",   "D", False, {"Close": 10800, "BB_Upper": 10900, "EXT_20": 4.0, "ROC_ACCEL": -0.5},   0, False, "S3", 40.0),
    ("S3r3_ext_high_accel",   "D", False, {"Close": 10800, "BB_Upper": 10900, "EXT_20": 4.0, "ROC_ACCEL":  0.5},   0, False, "S3", 20.0),
    ("S3r4_ext_low_decel",    "D", False, {"Close": 10800, "BB_Upper": 10900, "EXT_20": 1.0, "ROC_ACCEL": -0.5},   0, False, "S3", 30.0),
    ("S3r5_ext_low_accel",    "D", False, {"Close": 10800, "BB_Upper": 10900, "EXT_20": 1.0, "ROC_ACCEL":  0.5},   0, False, "S3", 15.0),
]

# Adjustment combos: (label, healthy_trend, trend_mult, mom_mult)
_ADJ_COMBOS = [
    ("H+NM",  True,  1.0,  1.0),
    ("H+M",   True,  1.0,  1.25),
    ("W+NM",  False, 0.75, 1.0),
    ("W+M",   False, 0.75, 1.25),
]


class TestAllSizingLeaves(unittest.TestCase):
    """21 base leaves × 4 adjustment combos = 84 cases.

    Each subTest verifies:
      1. Rule selection   (select_long/short_rule)
      2. Base sizing      (size_long/short)
      3. Adjusted output  (apply_all_adjustments)
    """

    def setUp(self):
        self.cfg = DEFAULT_CONFIG.copy()

    # ---- long side (9 × 4 = 36) ----
    def test_long_leaves(self):
        cfg = self.cfg
        for lbl, regime, close, bb_lo, ext20, row, exp_rule, base in _LONG_LEAVES:
            # Verify rule selection
            rule = select_long_rule(regime, close, bb_lo, ext20, cfg)
            self.assertEqual(rule, exp_rule, f"{lbl}: rule")
            # Verify base sizing
            self.assertEqual(size_long(rule, row, cfg), base, f"{lbl}: base")

            for adj_lbl, healthy, t_m, m_m in _ADJ_COMBOS:
                with self.subTest(leaf=lbl, adj=adj_lbl):
                    slope = 0.5 if healthy else -0.5  # bull regimes
                    mom_bull = m_m > 1.0
                    expected = round(min(base * t_m * m_m, cfg.max_tqqq), 2)

                    got = apply_all_adjustments(
                        base, "long", regime, slope,
                        momentum_bullish=mom_bull,
                        momentum_bearish=False,
                        cfg=cfg,
                    )
                    self.assertAlmostEqual(got, expected, places=2)

    # ---- short side (12 × 4 = 48) ----
    def test_short_leaves(self):
        cfg = self.cfg
        for lbl, regime, s2a, row, s2d, bbx, exp_rule, base in _SHORT_LEAVES:
            # Verify rule selection
            rule = select_short_rule(regime, s2a, cfg)
            self.assertEqual(rule, exp_rule, f"{lbl}: rule")
            # Verify base sizing
            self.assertEqual(size_short(rule, row, s2d, bbx, cfg), base,
                             f"{lbl}: base")

            for adj_lbl, healthy, t_m, m_m in _ADJ_COMBOS:
                with self.subTest(leaf=lbl, adj=adj_lbl):
                    # Bear regimes: healthy = slope < 0
                    slope = -0.5 if healthy else 0.5
                    mom_bear = m_m > 1.0
                    expected = round(min(base * t_m * m_m, cfg.max_sqqq), 2)

                    got = apply_all_adjustments(
                        base, "short", regime, slope,
                        momentum_bullish=False,
                        momentum_bearish=mom_bear,
                        cfg=cfg,
                    )
                    self.assertAlmostEqual(got, expected, places=2)


if __name__ == "__main__":
    unittest.main()
