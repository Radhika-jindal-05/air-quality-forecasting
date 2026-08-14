"""
Model Training and Baseline Comparison Pipeline.

This module executes:
1. End-to-end data loading, cleaning, daily aggregation, and CPCB AQI calculation.
2. Causal feature engineering and chronological train/validation/test splitting.
3. Fitting and benchmarking:
   - Baseline 1: Naive Persistence (predicts current observation AQI(t))
   - Baseline 2: Linear Regression
   - Baseline 3: Ridge Regression
   - Baseline 4: Random Forest Regressor
   - Primary Model: XGBoost Regressor (with validation set early stopping)
4. Persisting the trained XGBoost model and preprocessing artifacts to models/xgboost_model.pkl.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from xgboost import XGBRegressor

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_preprocess
from src.aqi_calculator import calculate_aqi_dataframe
from src.feature_engineering import (
    create_forecasting_features,
    chronological_split,
    prepare_feature_matrices,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

MODELS_DIR = "models"
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
METADATA_JSON_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred))
    }


def train_and_evaluate_all():
    """
    Run complete training, baseline comparison, and model serialization workflow.
    """
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("        STARTING AIR QUALITY FORECASTING TRAINING PIPELINE        ")
    print("=" * 70)

    # 1. Load, Preprocess & Compute AQI
    print("\n[Step 1/5] Loading data, cleaning & aggregating to daily city level...")
    daily_df = load_and_preprocess()
    print(f"  -> Daily aggregated records: {len(daily_df):,}")

    print("\n[Step 2/5] Calculating deterministic CPCB ground-truth AQI...")
    aqi_df = calculate_aqi_dataframe(daily_df)
    valid_aqi_cnt = aqi_df['aqi'].notna().sum()
    print(f"  -> Valid ground-truth AQI records: {valid_aqi_cnt:,} ({valid_aqi_cnt/len(aqi_df)*100:.2f}%)")

    # 2. Feature Engineering & Chronological Split
    print("\n[Step 3/5] Engineering causal features and filtering valid horizons (<= 7 days)...")
    model_df = create_forecasting_features(aqi_df, max_horizon_days=7)
    print(f"  -> Valid modeling samples: {len(model_df):,}")

    print("\n[Step 4/5] Partitioning dataset chronologically (70% Train / 15% Val / 15% Test)...")
    train_df, val_df, test_df, split_info = chronological_split(model_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    print("  -> Chronological Split Boundaries:")
    print(f"     Train: {split_info['train_start']} to {split_info['train_end']} ({split_info['train_samples']:,} samples)")
    print(f"     Val:   {split_info['val_start']} to {split_info['val_end']} ({split_info['val_samples']:,} samples)")
    print(f"     Test:  {split_info['test_start']} to {split_info['test_end']} ({split_info['test_samples']:,} samples)")

    X_train, y_train, X_val, y_val, X_test, y_test, train_medians = prepare_feature_matrices(
        train_df, val_df, test_df
    )

    # 3. Fit Baseline Models
    print("\n[Step 5/5] Training and evaluating baseline models on validation set...")
    val_results = {}

    # Model 1: Persistence / Naive Baseline (predicts current AQI)
    y_val_pred_naive = val_df["aqi"].values
    val_results["Naive (Persistence)"] = evaluate_predictions(y_val, y_val_pred_naive)
    print(f"  [1/5] Naive Baseline   | MAE: {val_results['Naive (Persistence)']['MAE']:6.2f} | RMSE: {val_results['Naive (Persistence)']['RMSE']:6.2f} | R2: {val_results['Naive (Persistence)']['R2']:6.4f}")

    # Model 2: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    val_results["Linear Regression"] = evaluate_predictions(y_val, lr.predict(X_val))
    print(f"  [2/5] Linear Regression| MAE: {val_results['Linear Regression']['MAE']:6.2f} | RMSE: {val_results['Linear Regression']['RMSE']:6.2f} | R2: {val_results['Linear Regression']['R2']:6.4f}")

    # Model 3: Ridge Regression
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    val_results["Ridge Regression"] = evaluate_predictions(y_val, ridge.predict(X_val))
    print(f"  [3/5] Ridge Regression | MAE: {val_results['Ridge Regression']['MAE']:6.2f} | RMSE: {val_results['Ridge Regression']['RMSE']:6.2f} | R2: {val_results['Ridge Regression']['R2']:6.4f}")

    # Model 4: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_results["Random Forest"] = evaluate_predictions(y_val, rf.predict(X_val))
    print(f"  [4/5] Random Forest    | MAE: {val_results['Random Forest']['MAE']:6.2f} | RMSE: {val_results['Random Forest']['RMSE']:6.2f} | R2: {val_results['Random Forest']['R2']:6.4f}")

    # Model 5: XGBoost Regressor
    print("  [5/5] Fitting XGBoost Regressor with early stopping on validation set...")
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    val_results["XGBoost Regressor"] = evaluate_predictions(y_val, xgb.predict(X_val))
    print(f"  [5/5] XGBoost Regressor| MAE: {val_results['XGBoost Regressor']['MAE']:6.2f} | RMSE: {val_results['XGBoost Regressor']['RMSE']:6.2f} | R2: {val_results['XGBoost Regressor']['R2']:6.4f} (best iter: {xgb.best_iteration})")

    # Feature Importances
    feature_importances = {
        col: float(val)
        for col, val in zip(FEATURE_COLUMNS, xgb.feature_importances_)
    }
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    # Location statistics for Streamlit city selector
    loc_counts = model_df.groupby("location").size().to_dict()
    supported_cities = sorted([loc for loc, count in loc_counts.items() if count >= 100])

    # Save Model Artifact Bundle
    artifact_bundle = {
        "model": xgb,
        "feature_columns": FEATURE_COLUMNS,
        "train_medians": train_medians.to_dict(),
        "split_info": split_info,
        "val_results": val_results,
        "feature_importances": sorted_importances,
        "supported_cities": supported_cities,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(artifact_bundle, MODEL_PKL_PATH)
    print(f"\nSuccessfully serialized model artifact bundle to: {MODEL_PKL_PATH}")

    # Also save metadata JSON
    metadata = {
        "split_info": split_info,
        "validation_metrics": val_results,
        "feature_importances": sorted_importances,
        "feature_columns": FEATURE_COLUMNS,
        "supported_cities_count": len(supported_cities),
        "total_modeling_samples": len(model_df)
    }
    with open(METADATA_JSON_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata JSON to: {METADATA_JSON_PATH}")

    elapsed = time.time() - start_time
    print(f"\nPipeline completed successfully in {elapsed:.2f} seconds.")

    return artifact_bundle, (train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    train_and_evaluate_all()
