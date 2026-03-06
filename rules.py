"""
Sub-Strategy Rules Module
Rule selection + position sizing per Strategy_Rules Sections 5.2-5.5.
"""

from config import StrategyConfig


def select_long_rule(regime: str, close: float, bb_lower: float,
                     ext_20: float, cfg: StrategyConfig) -> str:
    """Select the active long rule. Priority: L4 > L3 > L2 > L1.
    Returns rule name or None if no long rule applies."""
    if regime == "B":
        if close < bb_lower:
            return "L4"
        if ext_20 <= cfg.ext20_l2_l3_boundary:
            return "L3"
        return "L2"
    elif regime == "A":
        return "L1"
    return None


def size_long(rule: str, row: dict, cfg: StrategyConfig) -> float:
    """Compute base TQQQ % for the given long rule. Returns 0-100."""
    if rule == "L1":
        if row["ROC_ACCEL"] > 0:
            return cfg.l1_accel_size
        elif row["EXT_20"] > cfg.ext20_l1_trim:
            return cfg.l1_decel_extended_size
        else:
            return cfg.l1_decel_normal_size

    elif rule == "L2":
        if row["EXT_250"] > cfg.ext250_l2_threshold:
            return cfg.l2_extended_size
        else:
            return cfg.l2_normal_size

    elif rule == "L3":
        if row["EXT_20"] > cfg.ext20_l3_extreme:
            return cfg.l3_moderate_size
        else:
            return cfg.l3_extreme_size

    elif rule == "L4":
        if row["Days_Below_BB_Lower"] <= 2:
            return cfg.l4_fresh_size
        else:
            return cfg.l4_extended_size

    return 0.0


def select_short_rule(regime: str, s2_active: bool, cfg: StrategyConfig) -> str:
    """Select the active short rule. Priority: S2 > S1 > S3.
    s2_active: whether S2 conditions are met (fresh break + ROC_ACCEL < 0).
    Returns rule name or None if no short rule applies."""
    if regime == "C":
        if s2_active:
            return "S2"
        return "S1"
    elif regime == "D":
        return "S3"
    return None


def size_short(rule: str, row: dict, s2_days_since_break: int,
               bb_expansion: bool, cfg: StrategyConfig) -> float:
    """Compute base SQQQ % for the given short rule. Returns 0-80."""
    if rule == "S1":
        if row["ROC_ACCEL"] < 0:
            return cfg.s1_accel_size
        elif row["EXT_20"] < cfg.ext20_s1_trim:
            return cfg.s1_decel_stretched_size
        else:
            return cfg.s1_decel_normal_size

    elif rule == "S2":
        early = s2_days_since_break <= cfg.s2_early_cutoff
        if early and bb_expansion:
            return cfg.s2_early_expand_size
        elif early and not bb_expansion:
            return cfg.s2_early_noexpand_size
        elif not early and bb_expansion:
            return cfg.s2_late_expand_size
        else:
            return cfg.s2_late_noexpand_size

    elif rule == "S3":
        # Row 1 override: Close > BB_Upper
        if row["Close"] > row["BB_Upper"]:
            return cfg.s3_bb_upper_override_size
        if row["EXT_20"] >= cfg.ext20_s3_boundary:
            if row["ROC_ACCEL"] <= 0:
                return cfg.s3_ext_high_decel_size
            else:
                return cfg.s3_ext_high_accel_size
        else:
            if row["ROC_ACCEL"] <= 0:
                return cfg.s3_ext_low_decel_size
            else:
                return cfg.s3_ext_low_accel_size

    return 0.0
