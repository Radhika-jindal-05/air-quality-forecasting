"""
Model Evaluation and Diagnostic Visualization Module.

Evaluates all models strictly on the frozen, held-out chronological test set:
1. Primary regression metrics: MAE, RMSE, R², Median Absolute Error (MedAE).
2. Derived CPCB category classification metrics: Accuracy, Macro F1, Weighted F1, Confusion Matrix.
3. Granular Error Analysis:
   - Error by forecast horizon gap (1 day, 2 days, 3 days, 4-7 days)
   - Error by AQI category (Good to Severe)
   - Error across representative top cities
4. Generates and persists publication-quality diagnostic charts:
   - actual_vs_predicted.png
   - residuals_distribution.png
   - time_series_forecast.png
   - feature_importance.png
   - model_comparison_bar.png
   - confusion_matrix_aqi.png
5. Persists test results to models/test_evaluation_results.json and models/error_analysis.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_preprocess
from src.aqi_calculator import calculate_aqi_dataframe, get_aqi_category
from src.feature_engineering import (
    create_forecasting_features,
    chronological_split,
    prepare_feature_matrices,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

MODELS_DIR = "models"
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")
RESULTS_JSON_PATH = os.path.join(MODELS_DIR, "test_evaluation_results.json")
ERROR_ANALYSIS_JSON_PATH = os.path.join(MODELS_DIR, "error_analysis.json")

# Styling settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute primary regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred))
    }


def evaluate_classification(y_true_cont: np.ndarray, y_pred_cont: np.ndarray) -> dict:
    """Compute classification metrics for continuous predictions mapped to CPCB buckets."""
    cat_true = get_aqi_category(y_true_cont)
    cat_pred = get_aqi_category(y_pred_cont)

    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    
    acc = float(accuracy_score(cat_true, cat_pred))
    macro_f1 = float(f1_score(cat_true, cat_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(cat_true, cat_pred, labels=labels, average="weighted", zero_division=0))
    report = classification_report(cat_true, cat_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(cat_true, cat_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }


def compute_error_analysis(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute granular error analysis across forecast gaps, categories, and top cities."""
    df_err = test_df.copy()
    df_err["y_true"] = y_true
    df_err["y_pred"] = y_pred
    df_err["abs_err"] = np.abs(y_true - y_pred)
    df_err["category_true"] = get_aqi_category(y_true)

    # 1. By Forecast Gap
    df_err["gap_bucket"] = pd.cut(
        df_err["gap_to_next"],
        bins=[0, 1, 2, 3, 7],
        labels=["1 Day (Strict Daily)", "2 Days", "3 Days (Semi-weekly)", "4-7 Days"]
    )
    gap_stats = df_err.groupby("gap_bucket", observed=False).agg(
        Sample_Count=("abs_err", "count"),
        MAE=("abs_err", "mean"),
        Median_AE=("abs_err", "median"),
        RMSE=("y_true", lambda s: root_mean_squared_error(s, df_err.loc[s.index, "y_pred"]))
    ).reset_index()

    # 2. By AQI Category
    cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cat_stats = df_err.groupby("category_true", observed=False).agg(
        Sample_Count=("abs_err", "count"),
        Percentage=("abs_err", lambda s: len(s) / len(df_err) * 100),
        MAE=("abs_err", "mean"),
        Median_AE=("abs_err", "median")
    ).reindex(cat_order).reset_index()

    # 3. By Top Cities
    top_cities = df_err["location"].value_counts().head(10).index
    city_stats = df_err[df_err["location"].isin(top_cities)].groupby("location").agg(
        Test_Samples=("abs_err", "count"),
        Mean_AQI=("y_true", "mean"),
        MAE=("abs_err", "mean"),
        RMSE=("y_true", lambda s: root_mean_squared_error(s, df_err.loc[s.index, "y_pred"])),
        R2=("y_true", lambda s: r2_score(s, df_err.loc[s.index, "y_pred"]))
    ).sort_values("Test_Samples", ascending=False).reset_index()

    return {
        "by_gap": gap_stats.to_dict(orient="records"),
        "by_category": cat_stats.to_dict(orient="records"),
        "by_top_cities": city_stats.to_dict(orient="records")
    }


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Generate Actual vs Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sample_idx = np.random.RandomState(42).choice(len(y_true), size=min(10000, len(y_true)), replace=False)
    
    ax.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.25, color="#1f77b4", s=15, edgecolors="none")
    max_val = max(np.percentile(y_true, 99.5), np.percentile(y_pred, 99.5))
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="1:1 Perfect Forecast Line")
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual Next-Period AQI")
    ax.set_ylabel("Predicted Next-Period AQI (XGBoost)")
    ax.set_title("Actual vs. Predicted AQI on Held-Out Test Set")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Generate Residuals Distribution plot."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    
    clipped_res = np.clip(residuals, -100, 100)
    sns.histplot(clipped_res, bins=60, kde=True, color="#2ca02c", ax=ax, edgecolor="black", alpha=0.6)
    
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error Line")
    ax.axvline(mean_err, color="black", linestyle=":", lw=1.5, label=f"Mean Error: {mean_err:.2f}")
    
    ax.set_xlabel("Prediction Residual (Actual AQI - Predicted AQI)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Test Residual Distribution (Mean Error: {mean_err:.2f}, Test RMSE: {root_mean_squared_error(y_true, y_pred):.2f})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importances(importances: dict, save_path: str):
    """Generate Top 15 Feature Importance chart."""
    top_items = list(importances.items())[:15]
    features = [k for k, v in top_items][::-1]
    scores = [v * 100 for k, v in top_items][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(features, scores, color="#3470a3", edgecolor="black")
    
    ax.set_xlabel("Relative Importance (%)")
    ax.set_title("Top 15 Most Informative Features (XGBoost Regressor)")
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(comparison_dict: dict, save_path: str):
    """Generate Model Comparison Bar Chart."""
    models = list(comparison_dict.keys())
    maes = [comparison_dict[m]["MAE"] for m in models]
    rmses = [comparison_dict[m]["RMSE"] for m in models]
    r2s = [comparison_dict[m]["R2"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    rects1 = ax1.bar(x - width, maes, width, label="MAE (lower is better)", color="#4575b4")
    rects2 = ax1.bar(x, rmses, width, label="RMSE (lower is better)", color="#d73027")

    ax1.set_ylabel("Error (AQI Units)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, r2s, color="#1a9850", marker="o", linewidth=2.5, markersize=8, label="R² Score (higher is better)")
    ax2.set_ylabel("R² Score", color="#1a9850")
    ax2.tick_params(axis="y", labelcolor="#1a9850")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="upper right")

    plt.title("Benchmarking Forecasting Models on Held-Out Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm_data: list, labels: list, save_path: str):
    """Generate Confusion Matrix Heatmap for AQI Categories."""
    cm = np.array(cm_data)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={'label': 'Normalized Ratio'})
    
    ax.set_xlabel("Predicted AQI Category")
    ax.set_ylabel("Actual AQI Category")
    ax.set_title("Derived AQI Category Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_time_series_samples(test_df: pd.DataFrame, y_pred: np.ndarray, save_path: str):
    """Generate actual vs forecast time-series for representative cities."""
    eval_df = test_df.copy()
    eval_df["predicted_aqi"] = y_pred

    cities = ["Jaipur", "Nashik", "Pune", "Delhi"]
    fig, axes = plt.subplots(len(cities), 1, figsize=(12, 10), sharex=False)

    for i, city in enumerate(cities):
        city_df = eval_df[eval_df["location"] == city].sort_values("date")
        if len(city_df) > 0:
            axes[i].plot(city_df["date"], city_df[TARGET_COLUMN], label="Actual Next AQI", color="#1f77b4", lw=1.5)
            axes[i].plot(city_df["date"], city_df["predicted_aqi"], label="Forecasted Next AQI (XGBoost)", color="#ff7f0e", lw=1.2, linestyle="--")
            axes[i].set_title(f"Forecast Timeline: {city} (Held-Out Test Period)")
            axes[i].set_ylabel("AQI")
            axes[i].legend(loc="upper right", fontsize=8)
        else:
            axes[i].text(0.5, 0.5, f"No test records for {city}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_full_evaluation():
    """
    Execute full evaluation protocol across all models on held-out test data.
    """
    print("=" * 70)
    print("        RUNNING HELD-OUT TEST EVALUATION & DIAGNOSTICS          ")
    print("=" * 70)

    # 1. Load pipeline and data
    daily_df = load_and_preprocess()
    aqi_df = calculate_aqi_dataframe(daily_df)
    model_df = create_forecasting_features(aqi_df, max_horizon_days=7)
    train_df, val_df, test_df, split_info = chronological_split(model_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_medians = prepare_feature_matrices(
        train_df, val_df, test_df
    )

    print(f"\nHeld-Out Test Date Span: {split_info['test_start']} to {split_info['test_end']}")
    print(f"Held-Out Test Samples:   {len(test_df):,} instances across {test_df['location'].nunique()} cities")

    # 2. Fit and evaluate all models on Test Set
    test_results = {}

    # Naive Baseline
    y_test_naive = test_df["aqi"].values
    test_results["Naive (Persistence)"] = evaluate_regression(y_test.values, y_test_naive)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    test_results["Linear Regression"] = evaluate_regression(y_test.values, lr.predict(X_test))

    # Ridge Regression
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    test_results["Ridge Regression"] = evaluate_regression(y_test.values, ridge.predict(X_test))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    test_results["Random Forest"] = evaluate_regression(y_test.values, rf.predict(X_test))

    # XGBoost
    if os.path.exists(MODEL_PKL_PATH):
        artifact = joblib.load(MODEL_PKL_PATH)
        xgb_model = artifact["model"]
    else:
        from src.train import train_and_evaluate_all
        artifact, _ = train_and_evaluate_all()
        xgb_model = artifact["model"]

    y_test_xgb = xgb_model.predict(X_test)
    test_results["XGBoost Regressor"] = evaluate_regression(y_test.values, y_test_xgb)

    # 3. Print Final Benchmark Table
    print("\n" + "-" * 75)
    print(f"{'Model':25s} | {'MAE':8s} | {'RMSE':8s} | {'R²':8s} | {'MedAE':8s}")
    print("-" * 75)
    for model_name, metrics in test_results.items():
        print(f"{model_name:25s} | {metrics['MAE']:8.2f} | {metrics['RMSE']:8.2f} | {metrics['R2']:8.4f} | {metrics['MedAE']:8.2f}")
    print("-" * 75)

    # 4. Classification Metrics for XGBoost
    clf_metrics = evaluate_classification(y_test.values, y_test_xgb)
    print(f"\nDerived AQI Category Metrics (XGBoost Continuous Forecast -> CPCB Buckets):")
    print(f"  Accuracy:    {clf_metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1:    {clf_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {clf_metrics['weighted_f1']:.4f}")

    # 5. Compute Detailed Error Analysis
    error_analysis = compute_error_analysis(test_df, y_test.values, y_test_xgb)

    # 6. Generate Diagnostic Plots
    print("\nGenerating and saving publication-grade diagnostic plots...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    plot_actual_vs_predicted(y_test.values, y_test_xgb, os.path.join(MODELS_DIR, "actual_vs_predicted.png"))
    plot_residuals(y_test.values, y_test_xgb, os.path.join(MODELS_DIR, "residuals_distribution.png"))
    plot_feature_importances(artifact["feature_importances"], os.path.join(MODELS_DIR, "feature_importance.png"))
    plot_model_comparison(test_results, os.path.join(MODELS_DIR, "model_comparison_bar.png"))
    plot_confusion_matrix(clf_metrics["confusion_matrix"], clf_metrics["labels"], os.path.join(MODELS_DIR, "confusion_matrix_aqi.png"))
    plot_time_series_samples(test_df, y_test_xgb, os.path.join(MODELS_DIR, "time_series_forecast.png"))

    # 7. Save results to JSON
    full_eval_payload = {
        "test_split_info": split_info,
        "test_regression_metrics": test_results,
        "xgb_classification_metrics": {
            "accuracy": clf_metrics["accuracy"],
            "macro_f1": clf_metrics["macro_f1"],
            "weighted_f1": clf_metrics["weighted_f1"],
            "report": clf_metrics["classification_report"]
        },
        "test_rmse": float(root_mean_squared_error(y_test.values, y_test_xgb)),
        "mean_prediction_error": float(np.mean(y_test.values - y_test_xgb))
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(full_eval_payload, f, indent=2)

    with open(ERROR_ANALYSIS_JSON_PATH, "w") as f:
        json.dump(error_analysis, f, indent=2)

    print(f"\nSaved full test evaluation results to: {RESULTS_JSON_PATH}")
    print(f"Saved granular error analysis to:      {ERROR_ANALYSIS_JSON_PATH}")
    print("Evaluation completed successfully!")

    return full_eval_payload


if __name__ == "__main__":
    run_full_evaluation()
