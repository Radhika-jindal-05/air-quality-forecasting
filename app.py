"""
Streamlit Web Application: Air Quality Forecasting using XGBoost.

This application demonstrates:
1. Deterministic calculation of current AQI via official CPCB piecewise breakpoints.
2. Next-Period AQI Forecasting using XGBoost trained on temporal lags and rolling patterns.
3. City-level historical forecasts across 250+ Indian cities.
4. Custom pollutant scenario prediction with input validation.
5. Model benchmark transparency and diagnostic metrics.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.aqi_calculator import calculate_aqi_scalar, get_aqi_category, calc_so2_subindex, calc_no2_subindex, calc_rspm_subindex
from src.feature_engineering import FEATURE_COLUMNS

# Page configuration
st.set_page_config(
    page_title="Air Quality Forecasting | XGBoost",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, responsive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .aqi-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
    }
    .badge-good { background-color: #22c55e; }
    .badge-satisfactory { background-color: #84cc16; color: #1e293b; }
    .badge-moderate { background-color: #eab308; color: #1e293b; }
    .badge-poor { background-color: #f97316; }
    .badge-very-poor { background-color: #ef4444; }
    .badge-severe { background-color: #991b1b; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load model bundle and test evaluation metrics."""
    model_path = os.path.join("models", "xgboost_model.pkl")
    results_path = os.path.join("models", "test_evaluation_results.json")

    if not os.path.exists(model_path):
        st.error(f"Trained model not found at '{model_path}'. Please run 'python src/train.py' first.")
        st.stop()

    artifact = joblib.load(model_path)

    test_results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            test_results = json.load(f)

    return artifact, test_results


@st.cache_data
def load_city_historical_summary():
    """Load latest records per city for historical forecasting tab."""
    from src.data_preprocessing import load_and_preprocess
    from src.aqi_calculator import calculate_aqi_dataframe
    from src.feature_engineering import create_forecasting_features

    daily_df = load_and_preprocess()
    aqi_df = calculate_aqi_dataframe(daily_df)
    model_df = create_forecasting_features(aqi_df, max_horizon_days=7)
    return model_df


def get_badge_html(category: str) -> str:
    """Return colored HTML badge based on CPCB category."""
    cat_lower = category.lower().replace(" ", "-")
    badge_class = f"badge-{cat_lower}" if f"badge-{cat_lower}" in ["badge-good", "badge-satisfactory", "badge-moderate", "badge-poor", "badge-very-poor", "badge-severe"] else "badge-moderate"
    return f'<span class="aqi-badge {badge_class}">{category}</span>'


def main():
    artifact, test_results = load_artifacts()
    model = artifact["model"]
    train_medians = artifact["train_medians"]
    supported_cities = artifact.get("supported_cities", [])
    residual_std = test_results.get("test_residual_std", 28.57)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode:",
        ["🏙️ City Historical Forecast", "🧪 Custom Scenario Forecast", "📊 Model Diagnostics & Methodology"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Methodology Overview")
    st.sidebar.info(
        "**Deterministic Ground Truth:** Current AQI is computed using the official CPCB piecewise sub-index formula.\n\n"
        "**Machine Learning Forecasting:** XGBoost forecasts the **Next-Period AQI** (within $\le 7$ days) from temporal dynamics and historical pollutant lags."
    )

    # Header
    st.markdown('<div class="main-header">Air Quality Forecasting using XGBoost</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-pollutant temporal forecasting pipeline evaluated on Indian Ambient Air Quality data (CPCB IND-AQI Standard)</div>', unsafe_allow_html=True)

    # TAB 1: City Historical Forecast
    if app_mode == "🏙️ City Historical Forecast":
        st.subheader("City-Level Next-Period Air Quality Forecast")
        st.write("Select an Indian city with continuous historical monitoring records to view recent pollutant trends and generate the next-period AQI forecast.")

        with st.spinner("Loading city historical observations..."):
            model_df = load_city_historical_summary()

        available_cities = sorted([c for c in supported_cities if c in model_df["location"].unique()])
        if not available_cities:
            available_cities = sorted(model_df["location"].unique())

        default_city_idx = available_cities.index("Delhi") if "Delhi" in available_cities else 0
        selected_city = st.selectbox("Select City / Monitoring Location:", available_cities, index=default_city_idx)

        city_df = model_df[model_df["location"] == selected_city].sort_values("date").reset_index(drop=True)

        if len(city_df) < 5:
            st.warning(f"City '{selected_city}' has limited records ({len(city_df)}). Forecast may rely on regional imputation.")

        # Latest observation
        latest_row = city_df.iloc[-1]
        latest_date = latest_row["date"].strftime("%Y-%m-%d")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Observation Date", latest_date)
        with col2:
            st.metric("SO₂ (µg/m³)", f"{latest_row['so2']:.1f}" if pd.notna(latest_row['so2']) else "N/A")
        with col3:
            st.metric("NO₂ (µg/m³)", f"{latest_row['no2']:.1f}" if pd.notna(latest_row['no2']) else "N/A")
        with col4:
            st.metric("RSPM / PM₁₀ (µg/m³)", f"{latest_row['rspm']:.1f}" if pd.notna(latest_row['rspm']) else "N/A")

        # Current AQI vs Next-Period Forecast
        curr_aqi = latest_row["aqi"]
        curr_cat = get_aqi_category(curr_aqi)

        # Prepare feature vector for latest row
        X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS]]).fillna(train_medians)
        pred_next_aqi = float(model.predict(X_latest)[0])
        pred_cat = get_aqi_category(pred_next_aqi)

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("#### 📐 Current Observed AQI (Deterministic CPCB Formula)")
            st.markdown(f"**Observed AQI:** `{curr_aqi:.1f}`")
            st.markdown(f"**Category:** {get_badge_html(curr_cat)}", unsafe_allow_html=True)
            st.caption("Calculated exactly using official CPCB piecewise breakpoints from same-day pollutant sub-indices.")

        with res_col2:
            st.markdown("#### 🔮 Next-Period Forecast (XGBoost ML Model)")
            st.markdown(f"**Forecasted AQI:** `{pred_next_aqi:.1f}`")
            st.markdown(f"**Predicted Category:** {get_badge_html(pred_cat)}", unsafe_allow_html=True)
            st.caption(f"Forecast horizon: next available observation ($\le 7$ days). Global test residual standard error: {residual_std:.2f} AQI units.")

        # Historical Trend Chart
        st.markdown("#### 📈 Historical AQI Trend for " + selected_city)
        chart_data = city_df.tail(60)[["date", "aqi"]].set_index("date")
        st.line_chart(chart_data)

    # TAB 2: Custom Scenario Forecast
    elif app_mode == "🧪 Custom Scenario Forecast":
        st.subheader("Custom Pollutant Scenario Forecaster")
        st.write("Enter hypothetical or current pollutant measurements and recent history to generate both the deterministic current AQI and future forecasted AQI.")

        with st.form("custom_forecast_form"):
            st.markdown("##### 1. Current Observation (Time $t$)")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                in_so2 = st.number_input("Current SO₂ (µg/m³)", min_value=0.0, max_value=1500.0, value=25.0, step=1.0)
            with f_col2:
                in_no2 = st.number_input("Current NO₂ (µg/m³)", min_value=0.0, max_value=800.0, value=45.0, step=1.0)
            with f_col3:
                in_rspm = st.number_input("Current RSPM / PM₁₀ (µg/m³)", min_value=0.0, max_value=1000.0, value=120.0, step=5.0)

            st.markdown("##### 2. Temporal & Causal Context")
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1:
                in_days_prev = st.slider("Days since previous observation", min_value=1, max_value=7, value=2, help="Elapsed days since the last recorded reading (typically 1 for daily, 2-3 for semi-weekly).")
            with c_col2:
                in_month = st.selectbox("Month of Year", list(range(1, 13)), index=9)  # October
            with c_col3:
                in_dow = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
            with c_col4:
                in_quarter = (in_month - 1) // 3 + 1
                st.text(f"Quarter: Q{in_quarter}")

            st.markdown("##### 3. Recent Historical Lags (Past Trends)")
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                in_aqi_lag1 = st.number_input("AQI at $t-1$", min_value=0.0, max_value=500.0, value=115.0, step=5.0)
            with l_col2:
                in_aqi_lag2 = st.number_input("AQI at $t-2$", min_value=0.0, max_value=500.0, value=110.0, step=5.0)
            with l_col3:
                in_aqi_lag3 = st.number_input("AQI at $t-3$", min_value=0.0, max_value=500.0, value=105.0, step=5.0)

            submit_btn = st.form_submit_state = st.form_submit_button("⚡ Compute & Forecast AQI")

        if submit_btn:
            # Deterministic Current AQI Calculation
            curr_aqi = calculate_aqi_scalar(so2=in_so2, no2=in_no2, rspm=in_rspm)
            curr_cat = get_aqi_category(curr_aqi)

            # Sub-indices breakdown
            si_so2 = calc_so2_subindex(in_so2)
            si_no2 = calc_no2_subindex(in_no2)
            si_rspm = calc_rspm_subindex(in_rspm)

            # Build feature dictionary for XGBoost
            dow_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(in_dow)
            lag_aqis = [in_aqi_lag1, in_aqi_lag2, in_aqi_lag3]

            feature_dict = {
                "aqi": curr_aqi,
                "so2": in_so2,
                "no2": in_no2,
                "rspm": in_rspm,
                "days_since_previous_observation": float(in_days_prev),
                "aqi_lag_1": in_aqi_lag1,
                "so2_lag_1": in_so2,
                "no2_lag_1": in_no2,
                "rspm_lag_1": in_rspm,
                "aqi_lag_2": in_aqi_lag2,
                "so2_lag_2": in_so2,
                "no2_lag_2": in_no2,
                "rspm_lag_2": in_rspm,
                "aqi_lag_3": in_aqi_lag3,
                "so2_lag_3": in_so2,
                "no2_lag_3": in_no2,
                "rspm_lag_3": in_rspm,
                "aqi_roll_mean_3": float(np.mean(lag_aqis)),
                "so2_roll_mean_3": in_so2,
                "no2_roll_mean_3": in_no2,
                "rspm_roll_mean_3": in_rspm,
                "aqi_roll_std_3": float(np.std(lag_aqis)),
                "aqi_roll_mean_7": float(np.mean(lag_aqis)),
                "so2_roll_mean_7": in_so2,
                "no2_roll_mean_7": in_no2,
                "rspm_roll_mean_7": in_rspm,
                "month": in_month,
                "day_of_week": dow_num,
                "day_of_year": in_month * 30,
                "quarter": in_quarter
            }

            input_row = pd.DataFrame([feature_dict])[FEATURE_COLUMNS].fillna(train_medians)
            pred_aqi = float(model.predict(input_row)[0])
            pred_cat = get_aqi_category(pred_aqi)

            st.markdown("### 📋 Prediction Results")
            out_col1, out_col2 = st.columns(2)

            with out_col1:
                st.markdown("#### 📐 Current Observed AQI (CPCB Deterministic)")
                st.metric("Calculated AQI (Time $t$)", f"{curr_aqi:.1f}")
                st.markdown(f"**Health Category:** {get_badge_html(curr_cat)}", unsafe_allow_html=True)
                st.markdown(f"- **SO₂ Sub-Index:** `{si_so2:.1f}`\n- **NO₂ Sub-Index:** `{si_no2:.1f}`\n- **RSPM Sub-Index:** `{si_rspm:.1f}`")

            with out_col2:
                st.markdown("#### 🔮 Forecasted Next-Period AQI (XGBoost ML)")
                st.metric("Forecasted AQI (Time $t+1$)", f"{pred_aqi:.1f}")
                st.markdown(f"**Predicted Category:** {get_badge_html(pred_cat)}", unsafe_allow_html=True)
                st.info(f"**Diagnostic Error Context:** Global test RMSE is **{residual_std:.2f} AQI units** (held-out test benchmark).")

    # TAB 3: Model Diagnostics & Methodology
    elif app_mode == "📊 Model Diagnostics & Methodology":
        st.subheader("Model Performance Diagnostics & Evaluation Rigor")

        st.markdown("#### 🏆 Benchmark Results on Held-Out Test Set (2014–2015)")
        st.write("Chronological split: Models evaluated strictly on the latest 15% chronological dates (61,851 samples across 253 cities).")

        reg_metrics = test_results.get("test_regression_metrics", {})
        if reg_metrics:
            metrics_df = pd.DataFrame(reg_metrics).T
            st.dataframe(metrics_df.style.format({
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R2": "{:.4f}",
                "MedAE": "{:.2f}"
            }))

        st.markdown("---")
        st.markdown("#### 📊 Diagnostic Plots")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if os.path.exists("models/model_comparison_bar.png"):
                st.image("models/model_comparison_bar.png", caption="Baseline Model Comparison (Held-Out Test Set)")
            if os.path.exists("models/actual_vs_predicted.png"):
                st.image("models/actual_vs_predicted.png", caption="Actual vs Predicted Next-Period AQI")

        with p_col2:
            if os.path.exists("models/feature_importance.png"):
                st.image("models/feature_importance.png", caption="Top Feature Importances (XGBoost)")
            if os.path.exists("models/residuals_distribution.png"):
                st.image("models/residuals_distribution.png", caption="Test Set Prediction Residuals")

        if os.path.exists("models/time_series_forecast.png"):
            st.image("models/time_series_forecast.png", caption="Sample City Timelines (Actual vs XGBoost Forecast)")


if __name__ == "__main__":
    main()
