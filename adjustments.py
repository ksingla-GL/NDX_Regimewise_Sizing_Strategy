"""
Post-Rule Adjustments Module
Trend Health modifier + Momentum Phase overlay per Strategy_Rules Section 7.
"""

from config import StrategyConfig


def apply_trend_health(base_size: float, regime: str, ma_250_slope: float,
                       cfg: StrategyConfig) -> float:
    """Apply trend health multiplier (Section 7.1)."""
    if regime in ("A", "B"):
        if ma_250_slope > 0:
            return base_size * 1.0
        else:
            return base_size * cfg.trend_health_weak_mult
    elif regime in ("C", "D"):
        if ma_250_slope < 0:
            return base_size * 1.0
        else:
            return base_size * cfg.trend_health_weak_mult
    return base_size


def apply_momentum_phase(size_after_health: float, side: str,
                         momentum_bullish: bool, momentum_bearish: bool,
                         cfg: StrategyConfig) -> float:
    """Apply momentum phase multiplier (Section 7.2).

    side: 'long' or 'short'
    Returns adjusted size capped at max_tqqq or max_sqqq.
    """
    if side == "long" and momentum_bullish:
        result = size_after_health * cfg.momentum_phase_mult
        return min(result, cfg.max_tqqq)
    elif side == "short" and momentum_bearish:
        result = size_after_health * cfg.momentum_phase_mult
        return min(result, cfg.max_sqqq)
    return size_after_health


def apply_all_adjustments(base_size: float, side: str, regime: str,
                          ma_250_slope: float, momentum_bullish: bool,
                          momentum_bearish: bool, cfg: StrategyConfig) -> float:
    """Apply both adjustments sequentially, then cap."""
    size = apply_trend_health(base_size, regime, ma_250_slope, cfg)
    size = apply_momentum_phase(size, side, momentum_bullish, momentum_bearish, cfg)

    # Final caps
    if side == "long":
        size = min(size, cfg.max_tqqq)
    else:
        size = min(size, cfg.max_sqqq)

    return round(size, 2)
