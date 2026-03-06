"""
Data Acquisition Module
Pulls historical daily data for $NDX, TQQQ, and SQQQ from Yahoo Finance.
Saves cleaned CSVs to Inputs/ folder.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Inputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = {
    "^NDX": "NDX",
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",
}

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def pull_data():
    """Download OHLCV data for all tickers and save as CSVs."""
    summary = {}

    for yahoo_ticker, label in TICKERS.items():
        print(f"Downloading {label} ({yahoo_ticker})...")
        df = yf.download(yahoo_ticker, start=START_DATE, end=END_DATE, auto_adjust=True)

        if df.empty:
            print(f"  WARNING: No data returned for {label}")
            continue

        # Flatten multi-level columns if present (yfinance >= 0.2.31)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep standard OHLCV columns
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in expected_cols if c in df.columns]]

        # Sort by date ascending
        df.sort_index(inplace=True)

        # Drop rows where Close is NaN
        before = len(df)
        df.dropna(subset=["Close"], inplace=True)
        dropped = before - len(df)

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{label}.csv")
        df.to_csv(out_path)

        summary[label] = {
            "rows": len(df),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "dropped_nan": dropped,
            "close_min": round(df["Close"].min(), 2),
            "close_max": round(df["Close"].max(), 2),
            "file": out_path,
        }
        print(f"  Saved {len(df)} rows -> {out_path}")

    return summary


def validate_data():
    """Load saved CSVs and run basic quality checks."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    for label in TICKERS.values():
        path = os.path.join(OUTPUT_DIR, f"{label}.csv")
        if not os.path.exists(path):
            print(f"\n{label}: FILE MISSING")
            continue

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"\n--- {label} ---")
        print(f"  Date range : {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Total rows : {len(df)}")
        print(f"  Columns    : {list(df.columns)}")

        # Check for NaN in Close
        nan_close = df["Close"].isna().sum()
        print(f"  NaN Close  : {nan_close}")

        # Check for duplicate dates
        dupes = df.index.duplicated().sum()
        print(f"  Dupe dates : {dupes}")

        # Check for gaps > 5 calendar days (rough weekend/holiday check)
        date_diffs = df.index.to_series().diff().dt.days
        big_gaps = date_diffs[date_diffs > 5]
        print(f"  Gaps > 5d   : {len(big_gaps)}")
        if len(big_gaps) > 0:
            for dt, gap in big_gaps.items():
                print(f"    {dt.date()}: {int(gap)} day gap")

        # Basic stats
        print(f"  Close range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
        print(f"  Last close : {df['Close'].iloc[-1]:.2f} ({df.index[-1].date()})")

    # Check date alignment between tickers
    print(f"\n--- Date Alignment ---")
    dfs = {}
    for label in TICKERS.values():
        path = os.path.join(OUTPUT_DIR, f"{label}.csv")
        if os.path.exists(path):
            dfs[label] = pd.read_csv(path, index_col=0, parse_dates=True)

    if len(dfs) == 3:
        common_start = max(df.index.min() for df in dfs.values())
        common_end = min(df.index.max() for df in dfs.values())
        print(f"  Common range: {common_start.date()} to {common_end.date()}")

        # Count rows in common range for each
        for label, df in dfs.items():
            mask = (df.index >= common_start) & (df.index <= common_end)
            print(f"  {label} rows in common range: {mask.sum()}")

        # Check if all dates match
        ndx_dates = set(dfs["NDX"].loc[common_start:common_end].index)
        tqqq_dates = set(dfs["TQQQ"].loc[common_start:common_end].index)
        sqqq_dates = set(dfs["SQQQ"].loc[common_start:common_end].index)

        only_ndx = ndx_dates - tqqq_dates
        only_tqqq = tqqq_dates - ndx_dates
        if only_ndx:
            print(f"  Dates in NDX but not TQQQ: {len(only_ndx)}")
        if only_tqqq:
            print(f"  Dates in TQQQ but not NDX: {len(only_tqqq)}")

        if ndx_dates == tqqq_dates == sqqq_dates:
            print("  All 3 tickers have identical trading dates in common range.")
        else:
            print("  WARNING: Date mismatch between tickers. Will need alignment.")


if __name__ == "__main__":
    print("Pulling data from Yahoo Finance...")
    print(f"Period: {START_DATE} to {END_DATE}\n")

    summary = pull_data()
    print("\n--- Download Summary ---")
    for label, info in summary.items():
        print(f"  {label}: {info['rows']} rows, {info['start']} to {info['end']}")

    validate_data()
