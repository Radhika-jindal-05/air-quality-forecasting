# Dataset Documentation: Indian Ambient Air Quality

## Overview
This repository utilizes historical ambient air quality records collected across Indian states and union territories under the **National Air Quality Monitoring Programme (NAMP)**, coordinated by the **Central Pollution Control Board (CPCB)** and the **Ministry of Environment, Forest and Climate Change (MoEFCC)**.

- **Expected Filename:** `data.csv` (placed inside the `data/` directory)
- **File Format:** CSV (`encoding="latin1"`, `low_memory=False`)
- **Total Raw Records:** 435,742 rows $\times$ 13 columns
- **Temporal Coverage:** 1987-01-01 to 2015-12-31 (29 years)
- **Spatial Coverage:** 304 unique monitoring locations across 34 Indian States & UTs (745 monitoring stations).

---

## Schema & Column Descriptions

| Column Name | Data Type | Description | Missingness (%) | Handled In Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `stn_code` | Object | CPCB monitoring station identification code | 33.06% | Used for station-level analysis |
| `sampling_date` | Object | Raw recording date string | 0.00% | Dropped in favor of standardized `date` |
| `state` | Object | Indian State or Union Territory | 0.00% | Spatial identifier |
| `location` | Object | City / Municipality / Monitoring Area | 0.00% | Primary spatial aggregation level |
| `agency` | Object | State Pollution Control Board or monitoring agency | 34.30% | Administrative metadata |
| `type` | Object | Area classification (Residential, Industrial, etc.) | 1.24% | Station metadata |
| `so2` | Float | Sulphur Dioxide concentration ($\mu g/m^3$, 24-hr avg) | 7.95% | Primary model feature & sub-index |
| `no2` | Float | Nitrogen Dioxide concentration ($\mu g/m^3$, 24-hr avg) | 3.73% | Primary model feature & sub-index |
| `rspm` | Float | Respirable Suspended Particulate Matter / $PM_{10}$ ($\mu g/m^3$) | 9.23% | Primary particulate feature & sub-index |
| `spm` | Float | Suspended Particulate Matter ($\mu g/m^3$, older standard) | 54.48% | Kept separate; not substituted for RSPM |
| `pm2_5` | Float | Fine Particulate Matter $PM_{2.5}$ ($\mu g/m^3$) | 97.86% | Excluded from primary model due to sparsity |
| `date` | Object | Standardized ISO date string (`YYYY-MM-DD`) | 0.00% | Parsed to datetime; invalid rows dropped |

---

## Data Placement Instructions
1. Obtain the dataset `data.csv`.
2. Place `data.csv` directly inside the `data/` folder:
   ```
   air-quality-forecasting/
   └── data/
       ├── data.csv
       └── README.md
   ```
3. Run `python src/data_preprocessing.py` to verify data loading and daily city-level aggregation.
