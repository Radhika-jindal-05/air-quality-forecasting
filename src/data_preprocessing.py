"""
Data Preprocessing Module for Air Quality Forecasting.

This module handles:
1. Loading the raw Indian Air Quality dataset (encoding="latin1", low_memory=False).
2. Parsing and cleaning temporal and spatial identifiers.
3. Converting pollutant columns to numeric and sanitizing negative values.
4. Aggregating multi-station observations to city-level daily averages.
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "data.csv")

POLLUTANT_COLUMNS = ["so2", "no2", "rspm", "pm2_5", "ws", "rh", "co", "ozone", "nh3", "spm"]


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw air quality dataset from CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    df = pd.read_csv(
        path,
        encoding="latin1",
        low_memory=False
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw air quality observations:
    - Parse dates into datetime objects.
    - Remove rows with missing date, state, or location.
    - Convert pollutant columns to numeric.
    - Sanitize negative pollutant values to NaN.
    """
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows missing essential identifiers or invalid dates
    df = df.dropna(subset=["date", "state", "location"]).copy()

    # Clean location and state string whitespace
    df["location"] = df["location"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()

    # Convert pollutant columns to numeric and sanitize invalid negatives
    for col in POLLUTANT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    # Sort chronologically by location and date
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple monitoring station observations per (location, state, date)
    into city-level daily mean pollutant concentrations.
    """
    available_pollutants = [col for col in POLLUTANT_COLUMNS if col in df.columns]

    agg_df = (
        df.groupby(["location", "state", "date"])[available_pollutants]
        .mean()
        .reset_index()
    )

    # Ensure chronological order per location
    agg_df = agg_df.sort_values(["location", "date"]).reset_index(drop=True)

    return agg_df


def load_and_preprocess(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Execute end-to-end data loading, cleaning, and daily city-level aggregation.
    """
    raw_df = load_data(path)
    clean_df = clean_data(raw_df)
    daily_df = aggregate_daily(clean_df)
    return daily_df


if __name__ == "__main__":
    print("Executing Data Preprocessing Pipeline...")
    daily_df = load_and_preprocess()
    print(f"Preprocessed daily aggregated shape: {daily_df.shape}")
    print(f"Unique locations: {daily_df['location'].nunique()}")
    print(f"Unique states: {daily_df['state'].nunique()}")
    print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")
    print("\nMissing values in daily aggregated dataset:")
    print(daily_df.isnull().sum())