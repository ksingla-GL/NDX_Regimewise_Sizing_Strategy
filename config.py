"""
Strategy Configuration - all tunable parameters in one place.
Pass modified copies of DEFAULT_CONFIG to the engine for optimization sweeps.
"""

from dataclasses import dataclass, field, asdict
from copy import deepcopy


@dataclass
class StrategyConfig:
    # --- Moving Averages ---
    ma_short_period: int = 20
    ma_long_period: int = 250

    # --- Rate of Change ---
    roc_lookback: int = 10
    roc_accel_lookback: int = 5

    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std: float = 2.0
    bb_momentum_days: int = 3       # out of last bb_momentum_window days
    bb_momentum_window: int = 5
    bb_expansion_threshold: float = 1.2  # multiplier vs 20-day avg width

    # --- MA_250 Slope ---
    ma_slope_lookback: int = 20

    # --- Whipsaw Filter ---
    whipsaw_confirm_days: int = 2

    # --- EXT thresholds (Long side) ---
    ext20_l2_l3_boundary: float = -3.0    # L2 vs L3 split
    ext20_l3_extreme: float = -6.0        # L3 sizing split
    ext20_l1_trim: float = 5.0            # L1 trim when decelerating

    # --- EXT thresholds (Short side) ---
    ext20_s1_trim: float = -5.0           # S1 trim when stretched
    ext20_s3_boundary: float = 3.0        # S3 sizing split

    # --- Position Sizes (Long) ---
    l1_accel_size: float = 80.0
    l1_decel_extended_size: float = 40.0
    l1_decel_normal_size: float = 60.0
    l2_extended_size: float = 50.0        # EXT_250 > 5%
    l2_normal_size: float = 40.0
    l3_moderate_size: float = 25.0
    l3_extreme_size: float = 10.0
    l4_fresh_size: float = 20.0           # days 1-2 below BB_Lower
    l4_extended_size: float = 10.0        # days 3+ below BB_Lower

    # --- Position Sizes (Short) ---
    s1_accel_size: float = 60.0
    s1_decel_stretched_size: float = 25.0
    s1_decel_normal_size: float = 40.0
    s2_early_expand_size: float = 70.0
    s2_early_noexpand_size: float = 50.0
    s2_late_expand_size: float = 60.0
    s2_late_noexpand_size: float = 50.0
    s3_bb_upper_override_size: float = 50.0
    s3_ext_high_decel_size: float = 40.0
    s3_ext_high_accel_size: float = 20.0
    s3_ext_low_decel_size: float = 30.0
    s3_ext_low_accel_size: float = 15.0

    # --- S2 Break Window ---
    s2_break_window: int = 10
    s2_early_cutoff: int = 5

    # --- EXT_250 threshold for L2 ---
    ext250_l2_threshold: float = 5.0

    # --- Post-Rule Adjustments ---
    trend_health_weak_mult: float = 0.75
    momentum_phase_mult: float = 1.25

    # --- Caps ---
    max_tqqq: float = 100.0
    max_sqqq: float = 80.0

    # --- Trade Filter ---
    min_trade_threshold: float = 5.0  # % of portfolio
    tx_cost_bps: float = 5.0          # transaction cost in basis points

    # --- Risk Management ---
    drawdown_alert: float = -30.0
    drawdown_reduce: float = -40.0
    drawdown_halt: float = -50.0
    drawdown_reduce_mult: float = 0.5

    daily_loss_reduce_only: float = -15.0       # single day
    daily_loss_consecutive: float = -10.0       # 2 consecutive days
    daily_loss_consec_duration: int = 5         # trading days

    # --- Intraday Stop-Loss (optional, Section 8.6) ---
    intraday_stop_enabled: bool = False
    intraday_stop_threshold: float = -10.0
    intraday_monitor_freq_min: int = 5

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = StrategyConfig()
