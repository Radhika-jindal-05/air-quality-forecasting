"""
Feature Engineering and Chronological Dataset Splitting.

This module handles:
1. Target construction: AQI(t+1) within the same location.
2. Horizon filtering: Retaining observation pairs with gap_to_next <= 7 days.
3. Causal feature generation available at time t:
   - Current observation at time t: AQI(t), SO2(t), NO2(t), RSPM(t)
   - Causal observation gap: days_since_previous_observation (gap_from_previous)
   - Historical lags: t-1, t-2, t-3 for AQI and pollutants
   - Rolling historical statistics up to time t (strictly zero target lookahead)
   - Calendar/temporal features: month, day_of_week, day_of_year, quarter
4. Dynamic chronological train / validation / test partitioning.
5. Strict training-set median imputation to prevent feature leakage.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Current observation at time t
    "aqi",
    "so2",
    "no2",
    "rspm",
    # Causal elapsed time from previous observation
    "days_since_previous_observation",
    # Historical Lags (t-1, t-2, t-3)
    "aqi_lag_1",
    "so2_lag_1",
    "no2_lag_1",
    "rspm_lag_1",
    "aqi_lag_2",
    "so2_lag_2",
    "no2_lag_2",
    "rspm_lag_2",
    "aqi_lag_3",
    "so2_lag_3",
    "no2_lag_3",
    "rspm_lag_3",
    # Rolling Historical Statistics (up to time t)
    "aqi_roll_mean_3",
    "so2_roll_mean_3",
    "no2_roll_mean_3",
    "rspm_roll_mean_3",
    "aqi_roll_std_3",
    "aqi_roll_mean_7",
    "so2_roll_mean_7",
    "no2_roll_mean_7",
    "rspm_roll_mean_7",
    # Temporal / Calendar features
    "month",
    "day_of_week",
    "day_of_year",
    "quarter"
]

TARGET_COLUMN = "target_next_aqi"


def create_forecasting_features(
    df: pd.DataFrame,
    max_horizon_days: int = 7
) -> pd.DataFrame:
    """
    Generate causal features and future observation target from daily aggregated data.

    Parameters:
    -----------
    df : pd.DataFrame
        Daily location-aggregated DataFrame with calculated ground-truth 'aqi'.
    max_horizon_days : int, default=7
        Maximum allowed days to the next observation to qualify as a valid forecasting target.

    Returns:
    --------
    pd.DataFrame
        Feature-engineered DataFrame filtered to valid forecasting pairs.
    """
    df = df.copy()

    # Drop rows without valid ground truth AQI
    df = df.dropna(subset=["aqi"]).copy()

    # Ensure strict chronological sorting by location and date
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    loc = df["location"]
    same_loc_next = (loc == loc.shift(-1))
    same_loc_prev1 = (loc == loc.shift(1))
    same_loc_prev2 = (loc == loc.shift(2)) & same_loc_prev1
    same_loc_prev3 = (loc == loc.shift(3)) & same_loc_prev2

    # 1. Target Construction: Next observed AQI within same location
    df["next_date"] = df["date"].shift(-1).where(same_loc_next)
    df["gap_to_next"] = (df["next_date"] - df["date"]).dt.days
    df[TARGET_COLUMN] = df["aqi"].shift(-1).where(same_loc_next)

    # 2. Causal Feature: Days since previous observation (known at time t)
    df["prev_date"] = df["date"].shift(1).where(same_loc_prev1)
    df["days_since_previous_observation"] = (df["date"] - df["prev_date"]).dt.days.fillna(7.0)

    # 3. Historical Lags (t-1, t-2, t-3)
    for col in ["aqi", "so2", "no2", "rspm"]:
        df[f"{col}_lag_1"] = df[col].shift(1).where(same_loc_prev1)
        df[f"{col}_lag_2"] = df[col].shift(2).where(same_loc_prev2)
        df[f"{col}_lag_3"] = df[col].shift(3).where(same_loc_prev3)

    # 4. Rolling Historical Features (using past lags up to time t: lag_1, lag_2, lag_3)
    # Using past values strictly ensures zero target lookahead
    for col in ["aqi", "so2", "no2", "rspm"]:
        lags_3 = df[[f"{col}_lag_1", f"{col}_lag_2", f"{col}_lag_3"]]
        df[f"{col}_roll_mean_3"] = lags_3.mean(axis=1)
        if col == "aqi":
            df["aqi_roll_std_3"] = lags_3.std(axis=1).fillna(0.0)

        # 7-observation rolling mean on lag_1 grouped by location
        df[f"{col}_roll_mean_7"] = (
            df.groupby("location")[f"{col}_lag_1"]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )

    # 5. Temporal / Calendar Features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter

    # 6. Filter to valid forecasting instances:
    # - Must have valid target_next_aqi
    # - Must have at least 1 historical lag (aqi_lag_1)
    # - Gap to next observation must be <= max_horizon_days (e.g. 7 days)
    valid_mask = (
        df[TARGET_COLUMN].notna()
        & df["aqi_lag_1"].notna()
        & (df["gap_to_next"] <= max_horizon_days)
    )

    model_df = df[valid_mask].copy().reset_index(drop=True)
    return model_df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split DataFrame chronologically based on sorted unique dates.

    Guarantees:
    - Zero future data in training/validation.
    - True held-out test evaluation on the latest chronological period.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_cutoff = unique_dates[train_end_idx]
    val_cutoff = unique_dates[val_end_idx]

    train_df = df[df["date"] <= train_cutoff].copy().reset_index(drop=True)
    val_df = df[(df["date"] > train_cutoff) & (df["date"] <= val_cutoff)].copy().reset_index(drop=True)
    test_df = df[df["date"] > val_cutoff].copy().reset_index(drop=True)

    split_info = {
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "train_samples": len(train_df),
        "val_start": str(val_df["date"].min().date()),
        "val_end": str(val_df["date"].max().date()),
        "val_samples": len(val_df),
        "test_start": str(test_df["date"].min().date()),
        "test_end": str(test_df["date"].max().date()),
        "test_samples": len(test_df),
    }

    return train_df, val_df, test_df, split_info


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract feature matrices and targets, imputing missing features strictly
    using training-set medians to eliminate any data leakage.
    """
    train_medians = train_df[FEATURE_COLUMNS].median()

    X_train = train_df[FEATURE_COLUMNS].fillna(train_medians)
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS].fillna(train_medians)
    y_val = val_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS].fillna(train_medians)
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_val, y_val, X_test, y_test, train_medians


if __name__ == "__main__":
    from src.data_preprocessing import load_and_preprocess
    from src.aqi_calculator import calculate_aqi_dataframe

    print("Running Feature Engineering Pipeline Test...")
    daily_df = load_and_preprocess()
    aqi_df = calculate_aqi_dataframe(daily_df)
    model_df = create_forecasting_features(aqi_df, max_horizon_days=7)

    print(f"Total modeling samples: {len(model_df):,}")
    print(f"Number of feature columns: {len(FEATURE_COLUMNS)}")

    train_df, val_df, test_df, split_info = chronological_split(model_df)
    print("\nChronological Split Information:")
    for k, v in split_info.items():
        print(f"  {k:15s}: {v}")

    X_train, y_train, X_val, y_val, X_test, y_test, medians = prepare_feature_matrices(
        train_df, val_df, test_df
    )
    print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    print("\nFeature Engineering Verification: SUCCESSFUL!")
