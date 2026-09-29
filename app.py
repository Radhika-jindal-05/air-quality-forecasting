"""
Streamlit Web Application: Air Quality Forecasting using XGBoost.

An interactive, responsive dashboard for:
1. Deterministic calculation of current AQI via official CPCB piecewise breakpoints.
2. Next-Period AQI Forecasting using XGBoost trained on temporal lags and rolling patterns.
3. City-level historical forecasts across 250+ Indian cities with interactive Plotly timelines.
4. Custom pollutant scenario simulation with preset scenarios and health advisories.
5. Interactive model diagnostics, error distributions, and benchmark transparency.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Custom CSS for polished, responsive, and modern dashboard styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .aqi-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        color: white;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .badge-good { background-color: #22c55e; }
    .badge-satisfactory { background-color: #84cc16; color: #1e293b; }
    .badge-moderate { background-color: #eab308; color: #1e293b; }
    .badge-poor { background-color: #f97316; }
    .badge-very-poor { background-color: #ef4444; }
    .badge-severe { background-color: #991b1b; }
    
    .health-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .health-good { background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534; }
    .health-satisfactory { background-color: #f7fee7; border-left: 5px solid #84cc16; color: #3f6212; }
    .health-moderate { background-color: #fefce8; border-left: 5px solid #eab308; color: #854d0e; }
    .health-poor { background-color: #fff7ed; border-left: 5px solid #f97316; color: #9a3412; }
    .health-very-poor { background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b; }
    .health-severe { background-color: #450a0a; border-left: 5px solid #991b1b; color: #fecaca; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load model bundle and test evaluation metrics."""
    model_path = os.path.join("models", "xgboost_model.pkl")
    results_path = os.path.join("models", "test_evaluation_results.json")
    error_analysis_path = os.path.join("models", "error_analysis.json")

    if not os.path.exists(model_path):
        st.error(f"Trained model not found at '{model_path}'. Please run 'python src/train.py' first.")
        st.stop()

    artifact = joblib.load(model_path)

    test_results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            test_results = json.load(f)

    error_analysis = {}
    if os.path.exists(error_analysis_path):
        with open(error_analysis_path, "r") as f:
            error_analysis = json.load(f)

    return artifact, test_results, error_analysis


@st.cache_data
def load_city_historical_summary():
    """Load latest records per city for historical forecasting tab."""
    snapshot_parquet = os.path.join("data", "city_snapshots.parquet")
    snapshot_csv_gz = os.path.join("data", "city_snapshots.csv.gz")
    
    if os.path.exists(snapshot_parquet):
        return pd.read_parquet(snapshot_parquet)
    elif os.path.exists(snapshot_csv_gz):
        return pd.read_csv(snapshot_csv_gz, compression="gzip")

    # Fallback to raw dataset if snapshot not found
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
    badge_class = f"badge-{cat_lower}" if f"badge-{cat_lower}" in [
        "badge-good", "badge-satisfactory", "badge-moderate",
        "badge-poor", "badge-very-poor", "badge-severe"
    ] else "badge-moderate"
    return f'<span class="aqi-badge {badge_class}">{category}</span>'


def get_health_advisory(category: str) -> str:
    """Return official CPCB health statement for each AQI bucket."""
    advisories = {
        "Good": ("health-good", "🌿 Minimal impact. Air quality is ideal for all outdoor activities and physical exercise."),
        "Satisfactory": ("health-satisfactory", "🍃 Minor breathing discomfort may occur to sensitive individuals (asthma patients / elderly)."),
        "Moderate": ("health-moderate", "⚠️ Breathing discomfort to people with lungs, asthma and heart diseases. Children and elderly should limit prolonged exertion."),
        "Poor": ("health-poor", "😷 Breathing discomfort to most people on prolonged exposure. Sensitive groups should avoid outdoor exertion and wear N95 masks."),
        "Very Poor": ("health-very-poor", "🚨 Respiratory illness on prolonged exposure. High risk of cardiovascular and lung aggravation. Limit outdoor activity strictly."),
        "Severe": ("health-severe", "🛑 Health emergency. Affects healthy people and seriously impacts those with existing diseases. Close outdoor activities immediately.")
    }
    css_class, msg = advisories.get(category, ("health-moderate", "Air quality parameters are moderate."))
    return f'<div class="health-card {css_class}"><strong>Health Advisory:</strong> {msg}</div>'


def create_aqi_gauge(aqi_val: float, title: str) -> go.Figure:
    """Create an interactive, colorful radial AQI Gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(max(aqi_val, 0), 500),
        title={"text": f"<b>{title}</b>", "font": {"size": 17, "color": "#1e293b"}},
        number={"font": {"size": 36, "color": "#0f172a", "family": "Inter"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#94a3b8", "tickvals": [0, 50, 100, 200, 300, 400, 500]},
            "bar": {"color": "#1e293b", "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#cbd5e1",
            "steps": [
                {"range": [0, 50], "color": "#22c55e"},
                {"range": [50, 100], "color": "#84cc16"},
                {"range": [100, 200], "color": "#eab308"},
                {"range": [200, 300], "color": "#f97316"},
                {"range": [300, 400], "color": "#ef4444"},
                {"range": [400, 500], "color": "#991b1b"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.8,
                "value": aqi_val
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig


def create_city_timeline_plot(city_df: pd.DataFrame, pred_next_aqi: float, selected_city: str) -> go.Figure:
    """Interactive timeline plot showing historical AQI, CPCB bands, and future forecast point."""
    plot_df = city_df.tail(60).copy()
    
    fig = go.Figure()

    # Category shaded background bands
    fig.add_hrect(y0=0, y1=50, fillcolor="#22c55e", opacity=0.08, line_width=0, annotation_text="Good", annotation_position="top left")
    fig.add_hrect(y0=50, y1=100, fillcolor="#84cc16", opacity=0.08, line_width=0, annotation_text="Satisfactory", annotation_position="top left")
    fig.add_hrect(y0=100, y1=200, fillcolor="#eab308", opacity=0.08, line_width=0, annotation_text="Moderate", annotation_position="top left")
    fig.add_hrect(y0=200, y1=300, fillcolor="#f97316", opacity=0.08, line_width=0, annotation_text="Poor", annotation_position="top left")
    fig.add_hrect(y0=300, y1=400, fillcolor="#ef4444", opacity=0.08, line_width=0, annotation_text="Very Poor", annotation_position="top left")
    fig.add_hrect(y0=400, y1=500, fillcolor="#991b1b", opacity=0.08, line_width=0, annotation_text="Severe", annotation_position="top left")

    # Historical Observed AQI Line
    fig.add_trace(go.Scatter(
        x=plot_df["date"],
        y=plot_df["aqi"],
        mode="lines+markers",
        name="Observed Ground-Truth AQI",
        line=dict(color="#0284c7", width=3),
        marker=dict(size=6, color="#0369a1"),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Observed AQI:</b> %{y:.1f}<extra></extra>"
    ))

    # Connection line to forecast
    latest_date = plot_df["date"].iloc[-1]
    latest_aqi = plot_df["aqi"].iloc[-1]
    
    forecast_date = latest_date + pd.Timedelta(days=1)
    
    fig.add_trace(go.Scatter(
        x=[latest_date, forecast_date],
        y=[latest_aqi, pred_next_aqi],
        mode="lines",
        name="Transition Horizon",
        line=dict(color="#f43f5e", width=2.5, dash="dot"),
        hoverinfo="skip"
    ))

    # Next-period forecast point
    fig.add_trace(go.Scatter(
        x=[forecast_date],
        y=[pred_next_aqi],
        mode="markers",
        name="🔮 XGBoost Next-Period Forecast",
        marker=dict(size=14, color="#f43f5e", symbol="star", line=dict(width=2, color="#881337")),
        hovertemplate="<b>Next-Period Forecast:</b> %{y:.1f}<br><b>Horizon:</b> $\le 7$ Days<extra></extra>"
    ))

    # Uncertainty error band (+/- 28.57 RMSE)
    fig.add_trace(go.Scatter(
        x=[forecast_date, forecast_date],
        y=[max(0, pred_next_aqi - 28.57), min(500, pred_next_aqi + 28.57)],
        mode="lines",
        name="±1 RMSE Uncertainty (±28.57)",
        line=dict(color="#fb7185", width=4),
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"<b>Historical AQI Progression & Next-Period Forecast for {selected_city}</b>",
        title_font={"size": 17, "color": "#1e293b"},
        xaxis_title="Date",
        yaxis_title="Air Quality Index (AQI)",
        yaxis=dict(range=[0, max(plot_df["aqi"].max() + 50, pred_next_aqi + 60, 220)], gridcolor="#f1f5f9"),
        xaxis=dict(gridcolor="#f1f5f9"),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )
    return fig


def create_subindex_breakdown_plot(so2_si: float, no2_si: float, rspm_si: float) -> go.Figure:
    """Create interactive bar chart showing pollutant sub-indices with dominant pollutant callout."""
    pollutants = ["SO₂ Sub-Index", "NO₂ Sub-Index", "RSPM / PM₁₀ Sub-Index"]
    values = [so2_si, no2_si, rspm_si]
    
    max_val = max(values)
    colors = ["#38bdf8" if v < max_val else "#f43f5e" for v in values]
    
    fig = go.Figure(go.Bar(
        x=pollutants,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="auto",
        hovertemplate="<b>%{x}:</b> %{y:.1f} AQI units<extra></extra>"
    ))
    
    fig.add_hline(y=max_val, line_dash="dash", line_color="#e11d48", annotation_text=f"Max Sub-Index = Overall AQI ({max_val:.1f})", annotation_position="top right")
    
    fig.update_layout(
        title="<b>Pollutant Sub-Index Breakdown (Deterministic CPCB)</b>",
        title_font={"size": 15, "color": "#1e293b"},
        yaxis_title="Sub-Index Value",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=45, b=20),
        yaxis=dict(range=[0, max(max_val * 1.25, 100)], gridcolor="#f1f5f9")
    )
    return fig


def main():
    artifact, test_results, error_analysis = load_artifacts()
    model = artifact["model"]
    train_medians = artifact["train_medians"]
    supported_cities = artifact.get("supported_cities", [])
    residual_std = test_results.get("test_residual_std", 28.57)

    # Sidebar Navigation & Context
    st.sidebar.markdown("## 🌤️ Navigation")
    app_mode = st.sidebar.radio(
        "Select Section:",
        ["🏙️ City Historical Forecast", "🧪 Custom Scenario Simulation", "📊 Model Benchmark & Diagnostics"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Core Model Performance")
    st.sidebar.markdown(
        """
        - **XGBoost Test RMSE:** `28.57` (vs Persistence `36.88`)
        - **R² Score:** `0.6582` (66% variance captured)
        - **Category Accuracy:** `72.69%`
        - **Test Dataset:** 61,851 unseen future instances
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Methodology Definition")
    st.sidebar.info(
        "**1. Deterministic Ground Truth:** Current AQI is computed using official CPCB piecewise sub-index equations.\n\n"
        "**2. Machine Learning Forecasting:** XGBoost forecasts the **Next-Period AQI** ($\le 7$ days) from 30 causal lag, rolling, and seasonal features."
    )

    # Top Title
    st.markdown('<div class="main-header">Air Quality Forecasting using XGBoost</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Methodologically rigorous temporal forecasting across 300+ Indian monitoring locations under CPCB IND-AQI standards</div>', unsafe_allow_html=True)

    # Top KPI Banner
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="stat-card"><div class="stat-number">28.57</div><div class="stat-label">XGBoost Test RMSE</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="stat-card"><div class="stat-number">0.6582</div><div class="stat-label">Test R² Score</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="stat-card"><div class="stat-number">72.69%</div><div class="stat-label">Category Accuracy</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(supported_cities)}</div><div class="stat-label">Supported Cities</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 1: City Historical Forecast
    # =========================================================================
    if app_mode == "🏙️ City Historical Forecast":
        st.subheader("🏙️ City Historical Replay & Next-Period Forecast")
        st.write("Explore continuous historical monitoring timelines across 250+ Indian cities and generate next-period forecasts.")

        with st.spinner("Loading city historical observations..."):
            model_df = load_city_historical_summary()

        available_cities = sorted([c for c in supported_cities if c in model_df["location"].unique()])
        if not available_cities:
            available_cities = sorted(model_df["location"].unique())

        default_city_idx = available_cities.index("Delhi") if "Delhi" in available_cities else 0
        
        sel_c1, sel_c2 = st.columns([3, 1])
        with sel_c1:
            selected_city = st.selectbox("Select Monitoring Location / City:", available_cities, index=default_city_idx)
        with sel_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.caption(f"Showing validated historical records for **{selected_city}**.")

        city_df = model_df[model_df["location"] == selected_city].sort_values("date").reset_index(drop=True)

        if len(city_df) < 5:
            st.warning(f"City '{selected_city}' has limited records ({len(city_df)}).")

        # Latest observation
        latest_row = city_df.iloc[-1]
        latest_date = latest_row["date"].strftime("%Y-%m-%d")

        # Current AQI vs Next-Period Forecast
        curr_aqi = latest_row["aqi"]
        curr_cat = get_aqi_category(curr_aqi)

        # Prepare feature vector for latest row
        X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS]]).fillna(train_medians)
        pred_next_aqi = float(model.predict(X_latest)[0])
        pred_cat = get_aqi_category(pred_next_aqi)

        # Delta calculation
        aqi_delta = pred_next_aqi - curr_aqi
        delta_symbol = "↗️ Expected to rise by" if aqi_delta > 0 else "↘️ Expected to improve by"
        delta_color = "normal" if abs(aqi_delta) < 5 else ("inverse" if aqi_delta > 0 else "off")

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Latest Observation Date", latest_date)
        with m2:
            st.metric("Latest SO₂", f"{latest_row['so2']:.1f} µg/m³" if pd.notna(latest_row['so2']) else "N/A")
        with m3:
            st.metric("Latest NO₂", f"{latest_row['no2']:.1f} µg/m³" if pd.notna(latest_row['no2']) else "N/A")
        with m4:
            st.metric("Latest RSPM (PM₁₀)", f"{latest_row['rspm']:.1f} µg/m³" if pd.notna(latest_row['rspm']) else "N/A")

        st.markdown("---")

        # Gauge Row
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.plotly_chart(create_aqi_gauge(curr_aqi, f"Current Observed AQI: {curr_aqi:.1f}"), use_container_width=True)
            st.markdown(f"<div style='text-align: center;'><strong>Category:</strong> {get_badge_html(curr_cat)}</div>", unsafe_allow_html=True)
            st.caption("Calculated deterministically from same-day pollutant sub-indices via official CPCB breakpoints.")

        with g_col2:
            st.plotly_chart(create_aqi_gauge(pred_next_aqi, f"Next-Period Forecast: {pred_next_aqi:.1f}"), use_container_width=True)
            st.markdown(f"<div style='text-align: center;'><strong>Predicted Category:</strong> {get_badge_html(pred_cat)} &nbsp; (<i>{delta_symbol} {abs(aqi_delta):.1f} pts</i>)</div>", unsafe_allow_html=True)
            st.caption(f"Machine learning forecast for next available observation ($\le 7$ days). Global test RMSE: ±{residual_std:.2f}.")

        # Health Advisory Banner
        st.markdown(get_health_advisory(pred_cat), unsafe_allow_html=True)

        # Interactive Timeline Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(create_city_timeline_plot(city_df, pred_next_aqi, selected_city), use_container_width=True)

    # =========================================================================
    # TAB 2: Custom Scenario Simulation
    # =========================================================================
    elif app_mode == "🧪 Custom Scenario Simulation":
        st.subheader("🧪 Real-Time Custom Scenario Forecaster")
        st.write("Input current pollutant measurements, temporal context, and recent lag history to simulate both the deterministic ground-truth AQI and the ML forecast.")

        # Preset Scenarios
        st.markdown("##### ⚡ Quick Preset Scenarios")
        preset_cols = st.columns(4)
        
        default_so2, default_no2, default_rspm = 25.0, 45.0, 120.0
        default_days_prev = 2
        default_month = 10
        default_lag1, default_lag2, default_lag3 = 115.0, 110.0, 105.0

        if "preset_loaded" not in st.session_state:
            st.session_state.preset_loaded = "moderate"

        with preset_cols[0]:
            if st.button("🌫️ Delhi Winter Smog", use_container_width=True):
                st.session_state.so2 = 28.0
                st.session_state.no2 = 98.0
                st.session_state.rspm = 310.0
                st.session_state.days_prev = 1
                st.session_state.month = 11
                st.session_state.lag1 = 260.0
                st.session_state.lag2 = 220.0
                st.session_state.lag3 = 180.0
                st.rerun()

        with preset_cols[1]:
            if st.button("🌧️ Monsoon Clean Washout", use_container_width=True):
                st.session_state.so2 = 8.0
                st.session_state.no2 = 18.0
                st.session_state.rspm = 38.0
                st.session_state.days_prev = 1
                st.session_state.month = 7
                st.session_state.lag1 = 45.0
                st.session_state.lag2 = 42.0
                st.session_state.lag3 = 50.0
                st.rerun()

        with preset_cols[2]:
            if st.button("🚗 Urban Traffic Peak", use_container_width=True):
                st.session_state.so2 = 22.0
                st.session_state.no2 = 78.0
                st.session_state.rspm = 145.0
                st.session_state.days_prev = 2
                st.session_state.month = 4
                st.session_state.lag1 = 125.0
                st.session_state.lag2 = 118.0
                st.session_state.lag3 = 110.0
                st.rerun()

        with preset_cols[3]:
            if st.button("🏭 Industrial Accumulation", use_container_width=True):
                st.session_state.so2 = 95.0
                st.session_state.no2 = 140.0
                st.session_state.rspm = 230.0
                st.session_state.days_prev = 3
                st.session_state.month = 1
                st.session_state.lag1 = 195.0
                st.session_state.lag2 = 175.0
                st.session_state.lag3 = 150.0
                st.rerun()

        with st.form("custom_forecast_form"):
            st.markdown("##### 1. Current Observation (Time $t$)")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                in_so2 = st.number_input("Current SO₂ (µg/m³)", min_value=0.0, max_value=1500.0, value=st.session_state.get("so2", 25.0), step=1.0)
            with f_col2:
                in_no2 = st.number_input("Current NO₂ (µg/m³)", min_value=0.0, max_value=800.0, value=st.session_state.get("no2", 45.0), step=1.0)
            with f_col3:
                in_rspm = st.number_input("Current RSPM / PM₁₀ (µg/m³)", min_value=0.0, max_value=1000.0, value=st.session_state.get("rspm", 120.0), step=5.0)

            st.markdown("##### 2. Temporal & Causal Context")
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1:
                in_days_prev = st.slider("Days since previous observation", min_value=1, max_value=7, value=st.session_state.get("days_prev", 2), help="Elapsed days since the last recorded reading.")
            with c_col2:
                in_month = st.selectbox("Month of Year", list(range(1, 13)), index=st.session_state.get("month", 10) - 1)
            with c_col3:
                in_dow = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
            with c_col4:
                in_quarter = (in_month - 1) // 3 + 1
                st.text(f"Quarter: Q{in_quarter}")

            st.markdown("##### 3. Recent Historical Lags (Past Trends)")
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                in_aqi_lag1 = st.number_input("AQI at $t-1$", min_value=0.0, max_value=500.0, value=st.session_state.get("lag1", 115.0), step=5.0)
            with l_col2:
                in_aqi_lag2 = st.number_input("AQI at $t-2$", min_value=0.0, max_value=500.0, value=st.session_state.get("lag2", 110.0), step=5.0)
            with l_col3:
                in_aqi_lag3 = st.number_input("AQI at $t-3$", min_value=0.0, max_value=500.0, value=st.session_state.get("lag3", 105.0), step=5.0)

            submit_btn = st.form_submit_button("⚡ Compute & Forecast AQI", use_container_width=True)

        if submit_btn or "so2" in st.session_state:
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

            st.markdown("### 📋 Prediction Results & Analysis")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.plotly_chart(create_aqi_gauge(curr_aqi, f"Current Observed AQI: {curr_aqi:.1f}"), use_container_width=True)
                st.markdown(f"<div style='text-align: center;'><strong>Current Category:</strong> {get_badge_html(curr_cat)}</div>", unsafe_allow_html=True)
            
            with res_col2:
                st.plotly_chart(create_aqi_gauge(pred_aqi, f"Next-Period Forecast: {pred_aqi:.1f}"), use_container_width=True)
                st.markdown(f"<div style='text-align: center;'><strong>Forecasted Category:</strong> {get_badge_html(pred_cat)}</div>", unsafe_allow_html=True)

            # Health Statement
            st.markdown(get_health_advisory(pred_cat), unsafe_allow_html=True)

            # Sub-index breakdown chart
            st.plotly_chart(create_subindex_breakdown_plot(si_so2, si_no2, si_rspm), use_container_width=True)

    # =========================================================================
    # TAB 3: Model Diagnostics & Methodology
    # =========================================================================
    elif app_mode == "📊 Model Benchmark & Diagnostics":
        st.subheader("📊 Model Performance Diagnostics & Evaluation Rigor")
        st.write("Comprehensive benchmarks evaluated strictly on the frozen chronological held-out test set (61,851 instances across 253 cities, 2014–2015).")

        # Benchmark Comparison Table & Interactive Bar Chart
        st.markdown("#### 🏆 Benchmark Results on Held-Out Test Set")
        reg_metrics = test_results.get("test_regression_metrics", {})
        
        if reg_metrics:
            models_list = list(reg_metrics.keys())
            mae_vals = [reg_metrics[m]["MAE"] for m in models_list]
            rmse_vals = [reg_metrics[m]["RMSE"] for m in models_list]
            r2_vals = [reg_metrics[m]["R2"] for m in models_list]
            medae_vals = [reg_metrics[m]["MedAE"] for m in models_list]

            # Interactive Plotly Benchmark Chart
            fig_bench = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_bench.add_trace(go.Bar(
                x=models_list,
                y=mae_vals,
                name="MAE (AQI Units)",
                marker_color="#38bdf8",
                hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f} AQI<extra></extra>"
            ), secondary_y=False)

            fig_bench.add_trace(go.Bar(
                x=models_list,
                y=rmse_vals,
                name="RMSE (AQI Units)",
                marker_color="#f43f5e",
                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f} AQI<extra></extra>"
            ), secondary_y=False)

            fig_bench.add_trace(go.Scatter(
                x=models_list,
                y=r2_vals,
                name="R² Score",
                mode="lines+markers",
                line=dict(color="#10b981", width=3),
                marker=dict(size=9, color="#059669"),
                hovertemplate="<b>%{x}</b><br>R² Score: %{y:.4f}<extra></extra>"
            ), secondary_y=True)

            fig_bench.update_layout(
                title="<b>Benchmarking 5 Forecasting Models (Held-Out Test Set)</b>",
                title_font={"size": 16, "color": "#1e293b"},
                barmode="group",
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig_bench.update_yaxes(title_text="Error (AQI Units)", secondary_y=False, gridcolor="#f1f5f9")
            fig_bench.update_yaxes(title_text="R² Score", range=[0, 1.0], secondary_y=True, gridcolor="#f1f5f9")

            st.plotly_chart(fig_bench, use_container_width=True)

            # Formatted Data Table
            metrics_df = pd.DataFrame(reg_metrics).T
            st.dataframe(metrics_df.style.format({
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R2": "{:.4f}",
                "MedAE": "{:.2f}"
            }), use_container_width=True)

        st.markdown("---")

        # Feature Importance & Horizon Error
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.markdown("#### 🏆 Top 12 Feature Importances")
            feat_imp = artifact.get("feature_importances", {})
            if feat_imp:
                top_12 = list(feat_imp.items())[:12]
                f_names = [k for k, v in top_12][::-1]
                f_scores = [v * 100 for k, v in top_12][::-1]

                fig_fi = go.Figure(go.Bar(
                    x=f_scores,
                    y=f_names,
                    orientation="h",
                    marker=dict(color=f_scores, colorscale="Blues", showscale=False),
                    hovertemplate="<b>%{y}</b>: %{x:.2f}% relative gain<extra></extra>"
                ))
                fig_fi.update_layout(
                    title="<b>XGBoost Feature Importance (%)</b>",
                    title_font={"size": 14, "color": "#1e293b"},
                    xaxis_title="Relative Importance (%)",
                    plot_bgcolor="white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=20),
                    xaxis=dict(gridcolor="#f1f5f9")
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        with d_col2:
            st.markdown("#### ⏱️ Error by Forecast Horizon Gap")
            by_gap = error_analysis.get("by_gap", [])
            if by_gap:
                gap_df = pd.DataFrame(by_gap)
                fig_gap = go.Figure(go.Bar(
                    x=gap_df["gap_bucket"],
                    y=gap_df["RMSE"],
                    marker_color="#6366f1",
                    text=[f"{v:.1f}" for v in gap_df["RMSE"]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>RMSE: %{y:.2f} AQI<br>Sample Count: %{customdata:,}<extra></extra>",
                    customdata=gap_df["Sample_Count"]
                ))
                fig_gap.update_layout(
                    title="<b>RMSE across Observation Horizons</b>",
                    title_font={"size": 14, "color": "#1e293b"},
                    yaxis_title="RMSE (AQI Units)",
                    plot_bgcolor="white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=20),
                    yaxis=dict(gridcolor="#f1f5f9")
                )
                st.plotly_chart(fig_gap, use_container_width=True)

        st.markdown("---")

        # Category Error & Classification Confusion Matrix
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            st.markdown("#### 🎯 Error by AQI Category")
            by_cat = error_analysis.get("by_category", [])
            if by_cat:
                cat_df = pd.DataFrame(by_cat)
                fig_cat = go.Figure(go.Bar(
                    x=cat_df["category_true"],
                    y=cat_df["MAE"],
                    marker_color=["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444", "#991b1b"],
                    text=[f"{v:.1f}" for v in cat_df["MAE"]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>MAE: %{y:.1f} AQI<br>Data Share: %{customdata:.2f}%<extra></extra>",
                    customdata=cat_df["Percentage"]
                ))
                fig_cat.update_layout(
                    title="<b>MAE per CPCB Health Category</b>",
                    title_font={"size": 14, "color": "#1e293b"},
                    yaxis_title="Mean Absolute Error (AQI Units)",
                    plot_bgcolor="white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=20),
                    yaxis=dict(gridcolor="#f1f5f9")
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        with c_col2:
            st.markdown("#### 🟦 Normalized Confusion Matrix")
            clf_metrics = test_results.get("xgb_classification_metrics", {})
            cm_data = clf_metrics.get("report", {})
            labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
            
            # Row normalized matrix from evaluate.py
            cm_norm_data = [
                [0.543, 0.432, 0.026, 0.000, 0.000, 0.000],
                [0.054, 0.795, 0.151, 0.000, 0.000, 0.000],
                [0.002, 0.202, 0.780, 0.016, 0.000, 0.000],
                [0.001, 0.033, 0.666, 0.290, 0.008, 0.001],
                [0.000, 0.065, 0.451, 0.418, 0.060, 0.005],
                [0.000, 0.130, 0.457, 0.391, 0.022, 0.000]
            ]
            
            fig_cm = px.imshow(
                cm_norm_data,
                x=labels,
                y=labels,
                color_continuous_scale="Blues",
                text_auto=".2f",
                aspect="auto"
            )
            fig_cm.update_layout(
                title="<b>Derived Category Confusion Matrix</b>",
                title_font={"size": 14, "color": "#1e293b"},
                xaxis_title="Predicted Category",
                yaxis_title="Actual Category",
                height=360,
                margin=dict(l=10, r=10, t=40, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📖 Mathematical Formulation & CPCB Standard")
        
        with st.expander("🔍 Click to view official CPCB Breakpoint Matrix & Formula"):
            st.markdown(r"""
            **Sub-Index Formula ($I_p$):**
            $$I_p = \frac{I_{\text{Hi}} - I_{\text{Lo}}}{B_{\text{Hi}} - B_{\text{Lo}}} (C_p - B_{\text{Lo}}) + I_{\text{Lo}}$$
            
            **Overall AQI Criterion:**
            $$\text{AQI} = \max\left(I_{\text{SO}_2}, I_{\text{NO}_2}, I_{\text{RSPM}}\right)$$
            
            | Category | AQI Range | SO₂ (µg/m³) | NO₂ (µg/m³) | RSPM / PM₁₀ (µg/m³) |
            | :--- | :--- | :--- | :--- | :--- |
            | **Good** | 0 – 50 | 0 – 40 | 0 – 40 | 0 – 50 |
            | **Satisfactory** | 51 – 100 | 41 – 80 | 41 – 80 | 51 – 100 |
            | **Moderate** | 101 – 200 | 81 – 380 | 81 – 180 | 101 – 250 |
            | **Poor** | 201 – 300 | 381 – 800 | 181 – 280 | 251 – 350 |
            | **Very Poor** | 301 – 400 | 801 – 1600 | 281 – 400 | 351 – 430 |
            | **Severe** | 401 – 500 | 1600+ | 400+ | 430+ |
            """)


if __name__ == "__main__":
    main()
