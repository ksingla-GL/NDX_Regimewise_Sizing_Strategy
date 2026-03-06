"""
Regime Classification Module
Determines market regime (A/B/C/D) per Strategy_Rules Section 4.
"""


def classify_regime(close: float, ma_20: float, ma_250: float) -> str:
    """Classify the current regime.

    Section 4.1 boundary rules:
    - Close == MA_250 -> Bear regime
    - Close == MA_20  -> treated as below MA_20

    Returns: 'A', 'B', 'C', or 'D'
    """
    if close > ma_250:
        # Bull regime
        if close > ma_20:
            return "A"  # Strong Uptrend
        else:
            return "B"  # Pullback in Uptrend (Close <= MA_20)
    else:
        # Bear regime (Close <= MA_250)
        # Boundary rule: Close = MA_20 -> treated as below MA_20 -> Regime C
        if close <= ma_20:
            return "C"  # Strong Downtrend
        else:
            return "D"  # Bounce in Downtrend (Close > MA_20)
