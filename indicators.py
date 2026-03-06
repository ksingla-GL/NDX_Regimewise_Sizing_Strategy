"""
Indicator Computation Module
Computes all indicators from $NDX daily close prices per Strategy_Rules Section 3.
All functions take a pandas Series/DataFrame and config, return Series/DataFrame.
"""

import pandas as pd
import numpy as np
from config import StrategyConfig


def compute_all_indicators(ndx_close: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    """Compute all strategy indicators from $NDX close prices.

    Returns DataFrame with columns:
        Close, MA_20, MA_250, EXT_20, EXT_250, ROC_10, ROC_ACCEL,
        MA_250_Slope, BB_Upper, BB_Lower, BB_Width, BB_Width_Avg
    """
    df = pd.DataFrame({"Close": ndx_close})

    # 3.1 Core Indicators
    df["MA_20"] = df["Close"].rolling(cfg.ma_short_period).mean()
    df["MA_250"] = df["Close"].rolling(cfg.ma_long_period).mean()

    df["EXT_20"] = (df["Close"] - df["MA_20"]) / df["MA_20"] * 100
    df["EXT_250"] = (df["Close"] - df["MA_250"]) / df["MA_250"] * 100

    df["ROC_10"] = (df["Close"] - df["Close"].shift(cfg.roc_lookback)) / df["Close"].shift(cfg.roc_lookback) * 100
    df["ROC_ACCEL"] = df["ROC_10"] - df["ROC_10"].shift(cfg.roc_accel_lookback)

    ma250_shifted = df["MA_250"].shift(cfg.ma_slope_lookback)
    df["MA_250_Slope"] = (df["MA_250"] - ma250_shifted) / ma250_shifted * 100

    # 3.2 Bollinger Bands
    bb_std = df["Close"].rolling(cfg.bb_period).std()
    df["BB_Upper"] = df["MA_20"] + cfg.bb_std * bb_std
    df["BB_Lower"] = df["MA_20"] - cfg.bb_std * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["MA_20"] * 100
    df["BB_Width_Avg"] = df["BB_Width"].rolling(cfg.bb_period).mean()

    # 3.3 Momentum Phase Detection
    # Upper Band Ride: Close > BB_Upper for >= N of last M days
    above_upper = (df["Close"] > df["BB_Upper"]).astype(int)
    df["Upper_Band_Count"] = above_upper.rolling(cfg.bb_momentum_window).sum()
    df["Upper_Band_Ride"] = df["Upper_Band_Count"] >= cfg.bb_momentum_days

    # Lower Band Ride: Close < BB_Lower for >= N of last M days
    below_lower = (df["Close"] < df["BB_Lower"]).astype(int)
    df["Lower_Band_Count"] = below_lower.rolling(cfg.bb_momentum_window).sum()
    df["Lower_Band_Ride"] = df["Lower_Band_Count"] >= cfg.bb_momentum_days

    # BB Expansion
    df["BB_Expansion"] = df["BB_Width"] > (df["BB_Width_Avg"] * cfg.bb_expansion_threshold)

    # Momentum Phases
    df["Momentum_Bullish"] = df["Upper_Band_Ride"] & df["BB_Expansion"]
    df["Momentum_Bearish"] = df["Lower_Band_Ride"] & df["BB_Expansion"]

    # Days below BB_Lower (consecutive count for L4 sizing)
    below_bb = df["Close"] < df["BB_Lower"]
    consec = pd.Series(0, index=df.index)
    count = 0
    for i in range(len(df)):
        if below_bb.iloc[i]:
            count += 1
        else:
            count = 0
        consec.iloc[i] = count
    df["Days_Below_BB_Lower"] = consec

    return df
