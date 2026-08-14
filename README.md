# Air Quality Forecasting using XGBoost

An interview-ready, methodologically rigorous machine learning project for **Next-Period Air Quality Index (AQI) Forecasting** across 300+ Indian monitoring locations, evaluated on 29 years of Indian Ambient Air Quality data under the Central Pollution Control Board (CPCB) standard.

---

## 1. Overview
Air quality management requires proactive decision-making. While the standard Air Quality Index (AQI) is computed deterministically from observed pollutant concentrations, environmental authorities and citizens need to anticipate air quality levels **before** severe pollution events occur. This project develops an end-to-end temporal forecasting system that predicts the future-period AQI ($\le 7$-day horizon) by exploiting nonlinear relationships and temporal interactions in historical pollutant measurements, multi-day lag structures, past rolling patterns, and calendar encodings.

---

## 2. Problem Statement
Given a time series of ambient air quality measurements for a city:
$$\mathcal{D}_{\text{loc}} = \{(t_i, \text{SO}_{2, i}, \text{NO}_{2, i}, \text{RSPM}_i, \text{AQI}_i)\}_{i=1}^N$$
The goal is to forecast the future Air Quality Index value $\text{AQI}_{t+1}$ at the next observation date within a maximum horizon of 7 days:
$$\widehat{\text{AQI}}_{t+1} = f\left(\mathbf{x}_t; \Theta\right)$$
where $\mathbf{x}_t$ consists strictly of causally available information up to observation time $t$.

---

## 3. Why Machine Learning if AQI already has a formula?

> [!IMPORTANT]
> **The Central Motivation & Deterministic Distinction:**
> The CPCB National Air Quality Index (IND-AQI) is calculated **deterministically** from pollutant concentrations using predefined piecewise linear breakpoint equations:
> $$\text{AQI} = \max\left(I_{\text{SO}_2}, I_{\text{NO}_2}, I_{\text{PM}_{10}}\right)$$
> Therefore, predicting the current AQI directly from the same same-day pollutant concentrations is **not a meaningful machine-learning task**—it merely learns an existing mathematical function and produces artificially inflated $>99\%$ accuracy.
> 
> This project instead uses the deterministic CPCB calculation to establish the **observed ground-truth AQI**, and applies machine learning to a fundamentally different problem: **forecasting the AQI of the next available observation** using current conditions, historical pollutant measurements, past rolling trends, and previous AQI values.

```
                    Historical AQ Data
                           │
                           ▼
                  Data Cleaning
                           │
                           ▼
             Location-Day Aggregation
                           │
                           ▼
                 CPCB AQI Calculator
                           │
                           ▼
                 ┌─────────┴─────────┐
                 │                   │
           AQI at time t        AQI at t+1
                 │                   │
                 │                TARGET
                 │
        ┌────────┴────────┐
        │                 │
    Current values    Historical lags
        │                 │
        └────────┬────────┘
                 │
                 ▼
          Temporal Features
                 │
                 ▼
       Chronological Split
        70% / 15% / 15%
                 │
                 ▼
       Baseline + XGBoost
                 │
                 ▼
        Held-out Test Set
                 │
        ┌────────┴────────┐
        ▼                 ▼
   AQI Forecast       Evaluation
        │                 │
        └────────┬────────┘
                 ▼
             Streamlit
```

---

## 4. Dataset
- **Source:** Historical Indian Ambient Air Quality Dataset (CPCB / NAMP / MoEFCC).
- **Scale:** 435,742 raw records across 34 Indian States/UTs, 304 unique locations, and 745 monitoring stations.
- **Date Span:** 1987-01-01 to 2015-12-31 (29 years).
- **Data Quality & Coverage Audit:**
  - `pm2_5`: Missing in **97.86%** of records $\rightarrow$ Excluded from primary features.
  - `spm`: Missing in **54.48%** of records $\rightarrow$ Retained purely as metadata/optional; **not** substituted for RSPM in ground truth calculation.
  - `so2` (7.95% missing), `no2` (3.73% missing), `rspm` (9.23% missing) $\rightarrow$ Form the core multi-pollutant feature space.

---

## 5. Methodology & Pipeline
1. **Data Preprocessing:** Standardized date parsing, non-negative range sanitization, and daily city aggregation.
2. **Deterministic Ground Truth:** Official CPCB sub-index calculations for $SO_2, NO_2,$ and $RSPM/PM_{10}$.
3. **Causal Feature Engineering:** Current observation at $t$, historical lags ($t-1, t-2, t-3$), past rolling history up to $t$, and `days_since_previous_observation`.
4. **Target Construction:** Next observed AQI ($t+1$) filtered to horizons $\le 7$ days. Future gap `gap_to_next` is **never** fed to the model.
5. **Dynamic Chronological Splitting:** 70% Train, 15% Validation, 15% Held-Out Test on sorted unique dates.
6. **Multi-Model Benchmarking:** Naive Persistence, Linear Regression, Ridge, Random Forest, and XGBoost.
7. **Comprehensive Evaluation & Diagnostics:** Regression, classification metrics, error breakdown by gap/category/city, and Streamlit app.

---

## 6. Data Preprocessing
Implemented in [`src/data_preprocessing.py`](file:///C:/RADHIKA-PROJECTS/air-quality-forecasting/src/data_preprocessing.py):
- Loaded with `encoding="latin1"` and `low_memory=False`.
- Dropped invalid dates and records missing `state` or `location`.
- Sanitized negative values to NaN.
- Aggregated multiple monitoring stations per `(location, state, date)` to daily mean concentrations, yielding **306,994 daily city records** across 304 locations.

---

## 7. AQI Calculation (CPCB Breakpoint Standard)
Implemented in [`src/aqi_calculator.py`](file:///C:/RADHIKA-PROJECTS/air-quality-forecasting/src/aqi_calculator.py):
Sub-indices $I_p$ are computed using the official CPCB piecewise linear interpolation:
$$I_p = \frac{I_{Hi} - I_{Lo}}{B_{Hi} - B_{Lo}} \times (C_p - B_{Lo}) + I_{Lo}$$

### CPCB Sub-Index Breakpoint Matrix ($\mu g/m^3$, 24-hr avg):
| Category | AQI Range ($I_{Lo} - I_{Hi}$) | $\text{SO}_2$ ($B_{Lo} - B_{Hi}$) | $\text{NO}_2$ ($B_{Lo} - B_{Hi}$) | $\text{PM}_{10}$ / $\text{RSPM}$ ($B_{Lo} - B_{Hi}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Good** | 0 – 50 | 0 – 40 | 0 – 40 | 0 – 50 |
| **Satisfactory** | 51 – 100 | 41 – 80 | 41 – 80 | 51 – 100 |
| **Moderate** | 101 – 200 | 81 – 380 | 81 – 180 | 101 – 250 |
| **Poor** | 201 – 300 | 381 – 800 | 181 – 280 | 251 – 350 |
| **Very Poor** | 301 – 400 | 801 – 1600 | 281 – 400 | 351 – 430 |
| **Severe** | 401 – 500 | 1600+ | 400+ | 430+ |

**Overall AQI Criterion:**
$$\text{AQI} = \max\left(I_{\text{SO}_2}, I_{\text{NO}_2}, I_{\text{RSPM}}\right)$$
Requires valid particulate matter ($RSPM$) and at least 2 total valid pollutant sub-indices. Produces **277,303 valid ground-truth AQI values (90.33% coverage)**.

---

## 8. Feature Engineering & Causal Guarantee
Implemented in [`src/feature_engineering.py`](file:///C:/RADHIKA-PROJECTS/air-quality-forecasting/src/feature_engineering.py):

> [!IMPORTANT]
> **Verified Zero Temporal Leakage:**
> - Input features $\mathbf{x}_t$ only use data available at observation time $t$.
> - `target_next_aqi` and `gap_to_next` (future elapsed time to $t+1$) are **strictly excluded** from input features.
> - The model instead receives `days_since_previous_observation` ($gap\_from\_previous$), which is causally known at time $t$.

### Feature Space (30 Features):
1. **Current Observation ($t$):** $\text{AQI}(t), \text{SO}_2(t), \text{NO}_2(t), \text{RSPM}(t)$.
2. **Causal Elapsed Time:** `days_since_previous_observation`.
3. **Historical Lags ($t-1, t-2, t-3$):** Lags for AQI and all individual pollutants.
4. **Past Rolling Statistics:** 3-observation and 7-observation rolling mean and std computed strictly on past lags ($\le t$).
5. **Temporal & Calendar:** `month`, `day_of_week`, `day_of_year`, `quarter`.
6. **Feature Imputation:** Missing feature values are imputed using strictly training-set medians.

---

## 9. Forecasting Strategy
- **Target Variable:** $\text{AQI}_{t+1}$ (the AQI of the next observed record within the same location).
- **Forecasting Horizon Filter:** $gap\_to\_next \le 7$ days (filtering out long monitoring interruptions $>7$ days, which comprise 2.35% of pairs).
- **Modeling Dataset:** **269,278 valid chronological forecasting instances**.

---

## 10. Models
1. **Naive Persistence Baseline:** Predicts future AQI as current AQI ($\widehat{\text{AQI}}_{t+1} = \text{AQI}_t$).
2. **Linear Regression:** Standard least-squares multivariate linear baseline.
3. **Ridge Regression:** L2 regularized linear model ($\alpha = 10.0$).
4. **Random Forest Regressor:** Non-linear ensemble ($100$ trees, `max_depth=12`).
5. **XGBoost Regressor (Primary):** Gradient-boosted decision trees (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping on validation set).

---

## 11. Evaluation Metrics
Evaluated on the frozen, held-out chronological test set:
- **Mean Absolute Error (MAE):** Average magnitude of forecast errors in AQI units.
- **Root Mean Squared Error (RMSE):** Penalizes larger forecast deviations.
- **Coefficient of Determination ($R^2$):** Proportion of future AQI variance explained by the model.
- **Median Absolute Error (MedAE):** Robust to heavy-tailed pollutant spikes.
- **Derived AQI Category Classification:** Accuracy, Macro F1, and Weighted F1 when continuous forecasts are mapped to the 6 CPCB categories.

---

## 12. Results & Benchmarks

### Dynamic Chronological Split Boundaries:
- **Training Set (70%):** `2003-01-02` to `2012-02-27` (154,628 samples)
- **Validation Set (15%):** `2012-02-28` to `2014-01-28` (52,799 samples)
- **Held-Out Test Set (15%):** `2014-01-29` to `2015-12-30` (61,851 samples across 253 cities)

### Frozen Held-Out Test Set Benchmark Results:
| Model | MAE (AQI units) | RMSE (AQI units) | $R^2$ Score | MedAE (AQI units) |
| :--- | :---: | :---: | :---: | :---: |
| **Naive (Persistence Baseline)** | 22.00 | 36.88 | 0.4304 | 12.00 |
| **Linear Regression** | 18.98 | 29.31 | 0.6402 | 12.72 |
| **Ridge Regression** | 18.98 | 29.31 | 0.6402 | 12.72 |
| **Random Forest Regressor** | 18.11 | 28.95 | 0.6489 | 11.27 |
| **XGBoost Regressor (Final)** | **18.05** | **28.57** | **0.6582** | **11.40** |

- **Test RMSE:** **28.57 AQI units**
- **Mean Prediction Error:** **-0.65 AQI units**

### Derived AQI Category Metrics (XGBoost):
- **Category Classification Accuracy:** **72.69%**
- **Macro F1-Score:** **0.4372**
- **Weighted F1-Score:** **0.7215**

---

## 13. Granular Error Analysis & Limitations

### 1. Error by Forecast Horizon Gap
| Observation Gap | Test Samples | MAE (AQI units) | MedAE (AQI units) | RMSE (AQI units) |
| :--- | :---: | :---: | :---: | :---: |
| **1 Day (Strict Daily)** | 31,512 (50.9%) | 20.49 | 13.57 | 31.40 |
| **2 Days** | 12,265 (19.8%) | 17.28 | 10.73 | 27.81 |
| **3 Days (Semi-Weekly)** | 9,203 (14.9%) | 14.21 | 8.72 | 22.55 |
| **4–7 Days** | 8,871 (14.3%) | 14.40 | 9.09 | 24.29 |

> [!NOTE]
> **Observation Gap Interpretation:**
> The relationship between observation gap and error is non-monotonic in this dataset. This may reflect differences in monitoring frequency, city composition, pollution variability, and observation patterns across locations; therefore, the results should not be interpreted as evidence that longer forecasting horizons are inherently easier.

### 2. Error by Ground-Truth AQI Category
| Category | Test Samples | % of Test Data | MAE (AQI units) | MedAE (AQI units) |
| :--- | :---: | :---: | :---: | :---: |
| **Good (0–50)** | 11,821 | 19.11% | 16.72 | 11.73 |
| **Satisfactory (51–100)** | 25,965 | 41.98% | 13.88 | 9.29 |
| **Moderate (101–200)** | 22,423 | 36.25% | 19.33 | 13.41 |
| **Poor (201–300)** | 1,412 | 2.28% | 60.35 | 51.95 |
| **Very Poor (301–400)** | 184 | 0.30% | 141.86 | 136.16 |
| **Severe (>400)** | 46 | 0.07% | 290.05 | 275.59 |

### 3. Confusion Matrix Breakdown (Actual vs. Predicted Category):
| Actual Category | Pred Good (%) | Pred Satisfactory (%) | Pred Moderate (%) | Pred Poor (%) | Pred Very Poor (%) | Pred Severe (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Good** | **54.3%** | 43.2% | 2.6% | 0.0% | 0.0% | 0.0% |
| **Satisfactory** | 5.4% | **79.5%** | 15.1% | 0.0% | 0.0% | 0.0% |
| **Moderate** | 0.2% | 20.2% | **78.0%** | 1.6% | 0.0% | 0.0% |
| **Poor** | 0.1% | 3.3% | 66.6% | **29.0%** | 0.8% | 0.1% |
| **Very Poor** | 0.0% | 6.5% | 45.1% | 41.8% | **6.0%** | 0.5% |
| **Severe** | 0.0% | 13.0% | 45.7% | 39.1% | 2.2% | **0.0%** |

### 🎯 Key Interview Discussion: Extreme Category Limitation & Class Imbalance
> **Key Finding & Limitation:**
> The model performs substantially better in the common Good–Moderate range but struggles with rare extreme events. In particular, only **6.0% of Very Poor observations** and **0% of Severe observations** are correctly classified into their exact extreme buckets (frequently predicted as Moderate or Poor).
>
> **Technical Explanation:**
> Good, Satisfactory, and Moderate categories account for **97.34%** of all test observations, where the model achieves substantially lower prediction errors (MAE approximately 14–19 AQI units). In contrast, extreme tail events (Very Poor and Severe) represent less than **0.37%** of the dataset (only 230 combined samples out of 61,851). Standard regression loss functions optimize for the global distribution, causing the model to exhibit regression-to-the-mean during extreme tail spikes. Accuracy (72.69%) and Weighted F1 (0.7215) are dominated by the majority classes, whereas Macro F1 (0.4372) treats all classes equally and thus transparently exposes this limitation.

---

## 14. Streamlit Application
The interactive dashboard in [`app.py`](file:///C:/RADHIKA-PROJECTS/air-quality-forecasting/app.py) provides:
1. **City Historical Forecaster:** Select from 250+ Indian cities to view historical AQI timelines, latest pollutant levels, and the next-period predicted AQI with CPCB color badges.
2. **Custom Scenario Forecaster:** Enter hypothetical/current pollutant concentrations to compute the exact current CPCB AQI and forecast the next-period AQI with full sub-index breakdown.
3. **Model Diagnostics & Methodology Tab:** Interactive benchmark comparison table, feature importance visualization, residual error distributions, and methodology documentation.

---

## 15. Project Structure
```
air-quality-forecasting/
├── data/
│   ├── data.csv                        # Raw Indian air quality dataset (kept locally)
│   └── README.md                       # Data dictionary & placement guide
├── notebooks/
│   └── air-quality-forecasting.ipynb  # End-to-end ML narrative & analysis notebook
├── src/
│   ├── __init__.py                     # Package init
│   ├── data_preprocessing.py           # Cleaning & daily location aggregation
│   ├── aqi_calculator.py               # Deterministic CPCB IND-AQI sub-index calculator
│   ├── feature_engineering.py          # Causal lags, rolling history & chronological split
│   ├── train.py                        # Model training, baselines & model serialization
│   └── evaluate.py                     # Held-out evaluation & diagnostic plot generator
├── models/
│   ├── xgboost_model.pkl               # Serialized XGBoost model bundle & train medians
│   ├── model_metadata.json             # Pipeline & validation metadata
│   ├── test_evaluation_results.json    # Exact test benchmark results
│   ├── error_analysis.json             # Granular error analysis data
│   ├── actual_vs_predicted.png         # Diagnostic scatter plot
│   ├── residuals_distribution.png      # Error distribution plot
│   ├── feature_importance.png          # XGBoost feature importance chart
│   ├── model_comparison_bar.png        # Baseline comparison bar chart
│   ├── confusion_matrix_aqi.png        # Category confusion matrix
│   └── time_series_forecast.png        # Sample city timeline forecast
├── app.py                              # Streamlit web application
├── requirements.txt                    # Minimal project dependencies
├── README.md                           # Comprehensive documentation
├── .gitignore                          # Standard git ignore rules (ignoring data/*.csv)
└── LICENSE                             # MIT License
```

---

## 16. Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/air-quality-forecasting.git
cd air-quality-forecasting

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 17. How to Run

### Execute Training Pipeline:
```bash
python src/train.py
```

### Run Evaluation & Diagnostics:
```bash
python src/evaluate.py
```

### Launch Interactive Streamlit Application:
```bash
streamlit run app.py
```

---

## 18. Limitations & Future Work
1. **Extreme Event Detection:** As documented, extreme tail spikes (Very Poor / Severe representing $<0.37\%$ of data) suffer from regression-to-the-mean. Asymmetric loss functions or extreme-value modeling could improve tail capture.
2. **Irregular Observation Gaps:** Historical NAMP manual stations operated ~2 days per week. While modeled via `days_since_previous_observation`, continuous minute/hourly forecasting requires continuous automated stations (CAAQMS).
3. **Missing Fine Particulates ($PM_{2.5}$):** $PM_{2.5}$ was not monitored in earlier decades (97.86% missing in raw data).
4. **Exogenous Weather Integration:** Merging meteorological reanalysis data (wind speed/direction, boundary layer height, temperature) will significantly improve multi-day dispersion forecasting.

---

## 19. License
Distributed under the **MIT License**. See [`LICENSE`](file:///C:/RADHIKA-PROJECTS/air-quality-forecasting/LICENSE) for details.
