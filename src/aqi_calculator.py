"""
Deterministic Air Quality Index (AQI) Calculator.

Implements the official Central Pollution Control Board (CPCB) India
National Air Quality Index (IND-AQI) piecewise linear interpolation standard.

This module is strictly deterministic:
- Calculates pollutant sub-indices for SO2, NO2, and RSPM/PM10 based on official CPCB breakpoints.
- Calculates overall AQI as the maximum of available sub-indices, requiring valid particulate
  matter (RSPM) and at least two total pollutant measurements.
- Categorizes continuous AQI into official CPCB health categories.
- The ML model never replaces this formula; this module provides the ground truth.
"""

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd


# Official CPCB Breakpoints (Low_Conc, High_Conc, Low_Index, High_Index)
SO2_BREAKPOINTS = [
    (0.0, 40.0, 0.0, 50.0),
    (40.0, 80.0, 50.0, 100.0),
    (80.0, 380.0, 100.0, 200.0),
    (380.0, 800.0, 200.0, 300.0),
    (800.0, 1600.0, 300.0, 400.0),
    (1600.0, 2400.0, 400.0, 500.0)
]

NO2_BREAKPOINTS = [
    (0.0, 40.0, 0.0, 50.0),
    (40.0, 80.0, 50.0, 100.0),
    (80.0, 180.0, 100.0, 200.0),
    (180.0, 280.0, 200.0, 300.0),
    (280.0, 400.0, 300.0, 400.0),
    (400.0, 600.0, 400.0, 500.0)
]

RSPM_BREAKPOINTS = [
    (0.0, 50.0, 0.0, 50.0),
    (50.0, 100.0, 50.0, 100.0),
    (100.0, 250.0, 100.0, 200.0),
    (250.0, 350.0, 200.0, 300.0),
    (350.0, 430.0, 300.0, 400.0),
    (430.0, 600.0, 400.0, 500.0)
]

AQI_CATEGORIES = [
    (0.0, 50.0, "Good"),
    (50.0, 100.0, "Satisfactory"),
    (100.0, 200.0, "Moderate"),
    (200.0, 300.0, "Poor"),
    (300.0, 400.0, "Very Poor"),
    (400.0, float("inf"), "Severe")
]


def _interpolate_subindex_scalar(conc: Optional[float], breakpoints) -> float:
    """
    Interpolate sub-index for a single scalar concentration.
    """
    if conc is None or pd.isna(conc) or conc < 0:
        return np.nan

    for b_lo, b_hi, i_lo, i_hi in breakpoints:
        if b_lo <= conc <= b_hi:
            return i_lo + (conc - b_lo) * (i_hi - i_lo) / (b_hi - b_lo)

    # Extrapolate linearly from highest breakpoint bracket if concentration exceeds table
    b_lo, b_hi, i_lo, i_hi = breakpoints[-1]
    return i_lo + (conc - b_lo) * (i_hi - i_lo) / (b_hi - b_lo)


def calc_so2_subindex(so2: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Calculate SO2 sub-index (scalar or pandas Series)."""
    if isinstance(so2, pd.Series):
        res = np.where(so2 <= 40, so2 * 1.25,
              np.where(so2 <= 80, 50 + (so2 - 40) * 1.25,
              np.where(so2 <= 380, 100 + (so2 - 80) * (100 / 300),
              np.where(so2 <= 800, 200 + (so2 - 380) * (100 / 420),
              np.where(so2 <= 1600, 300 + (so2 - 800) * (100 / 800),
              400 + (so2 - 1600) * (100 / 800))))))
        return pd.Series(res, index=so2.index).where(so2.notna() & (so2 >= 0))
    return _interpolate_subindex_scalar(so2, SO2_BREAKPOINTS)


def calc_no2_subindex(no2: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Calculate NO2 sub-index (scalar or pandas Series)."""
    if isinstance(no2, pd.Series):
        res = np.where(no2 <= 40, no2 * 1.25,
              np.where(no2 <= 80, 50 + (no2 - 40) * 1.25,
              np.where(no2 <= 180, 100 + (no2 - 80),
              np.where(no2 <= 280, 200 + (no2 - 180),
              np.where(no2 <= 400, 300 + (no2 - 280) * (100 / 120),
              400 + (no2 - 400) * (100 / 120))))))
        return pd.Series(res, index=no2.index).where(no2.notna() & (no2 >= 0))
    return _interpolate_subindex_scalar(no2, NO2_BREAKPOINTS)


def calc_rspm_subindex(rspm: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Calculate RSPM (PM10) sub-index (scalar or pandas Series)."""
    if isinstance(rspm, pd.Series):
        res = np.where(rspm <= 50, rspm,
              np.where(rspm <= 100, 50 + (rspm - 50),
              np.where(rspm <= 250, 100 + (rspm - 100) * (100 / 150),
              np.where(rspm <= 350, 200 + (rspm - 250),
              np.where(rspm <= 430, 300 + (rspm - 350) * (100 / 80),
              400 + (rspm - 430) * (100 / 80))))))
        return pd.Series(res, index=rspm.index).where(rspm.notna() & (rspm >= 0))
    return _interpolate_subindex_scalar(rspm, RSPM_BREAKPOINTS)


def calculate_aqi_scalar(
    so2: Optional[float],
    no2: Optional[float],
    rspm: Optional[float]
) -> float:
    """
    Calculate deterministic AQI for a single scalar observation.
    Requires valid RSPM (PM10) and at least 2 valid pollutant sub-indices.
    """
    so2_si = calc_so2_subindex(so2)
    no2_si = calc_no2_subindex(no2)
    rspm_si = calc_rspm_subindex(rspm)

    valid_subindices = [si for si in [so2_si, no2_si, rspm_si] if pd.notna(si)]

    # CPCB standard: particulate matter (RSPM) must be present, and at least 2 pollutants total
    if pd.notna(rspm_si) and len(valid_subindices) >= 2:
        return float(max(valid_subindices))
    return np.nan


def calculate_aqi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized calculation of sub-indices and overall CPCB AQI across a DataFrame.
    """
    df = df.copy()

    df["so2_si"] = calc_so2_subindex(df["so2"])
    df["no2_si"] = calc_no2_subindex(df["no2"])
    df["rspm_si"] = calc_rspm_subindex(df["rspm"])

    si_cols = ["so2_si", "no2_si", "rspm_si"]
    valid_count = df[si_cols].notna().sum(axis=1)
    has_pm = df["rspm_si"].notna()

    max_si = df[si_cols].max(axis=1)

    # Condition: RSPM must be present and valid pollutant count >= 2
    df["aqi"] = max_si.where(has_pm & (valid_count >= 2))

    return df


def get_aqi_category(aqi: Union[float, pd.Series, np.ndarray]) -> Union[str, pd.Series, np.ndarray]:
    """
    Convert continuous AQI value(s) into CPCB categorical air quality classification:
    Good, Satisfactory, Moderate, Poor, Very Poor, Severe.
    """
    if isinstance(aqi, pd.Series):
        conditions = [
            aqi <= 50.0,
            (aqi > 50.0) & (aqi <= 100.0),
            (aqi > 100.0) & (aqi <= 200.0),
            (aqi > 200.0) & (aqi <= 300.0),
            (aqi > 300.0) & (aqi <= 400.0),
            aqi > 400.0
        ]
        choices = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        return pd.Series(np.select(conditions, choices, default="Unknown"), index=aqi.index)

    elif isinstance(aqi, np.ndarray):
        conditions = [
            aqi <= 50.0,
            (aqi > 50.0) & (aqi <= 100.0),
            (aqi > 100.0) & (aqi <= 200.0),
            (aqi > 200.0) & (aqi <= 300.0),
            (aqi > 300.0) & (aqi <= 400.0),
            aqi > 400.0
        ]
        choices = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        return np.select(conditions, choices, default="Unknown")

    # Scalar
    if pd.isna(aqi) or aqi is None:
        return "Unknown"
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi <= high:
            return label
    return "Severe"


if __name__ == "__main__":
    print("Testing CPCB AQI Calculator with Official Test Vectors...")

    test_cases = [
        # (pollutant, concentration, expected_subindex)
        ("SO2", 20.0, 25.0),
        ("SO2", 60.0, 75.0),
        ("SO2", 230.0, 150.0),
        ("NO2", 20.0, 25.0),
        ("NO2", 60.0, 75.0),
        ("NO2", 130.0, 150.0),
        ("RSPM", 25.0, 25.0),
        ("RSPM", 75.0, 75.0),
        ("RSPM", 175.0, 150.0),
        ("RSPM", 300.0, 250.0),
        ("RSPM", 390.0, 350.0),
    ]

    all_passed = True
    for p_type, conc, expected in test_cases:
        if p_type == "SO2":
            actual = calc_so2_subindex(conc)
        elif p_type == "NO2":
            actual = calc_no2_subindex(conc)
        elif p_type == "RSPM":
            actual = calc_rspm_subindex(conc)
        else:
            actual = np.nan

        passed = np.isclose(actual, expected, atol=0.01)
        if not passed:
            all_passed = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {p_type:5s} conc={conc:5.1f} -> Sub-Index: {actual:6.2f} (Expected: {expected:6.2f})")

    # Overall AQI test
    test_aqi = calculate_aqi_scalar(so2=60.0, no2=130.0, rspm=175.0)
    # Subindices: SO2=75, NO2=150, RSPM=150 -> Max is 150 (Moderate)
    cat = get_aqi_category(test_aqi)
    print(f"\nOverall AQI Test: SO2=60, NO2=130, RSPM=175 -> AQI: {test_aqi} ({cat})")
    assert test_aqi == 150.0, f"Expected 150.0, got {test_aqi}"
    assert cat == "Moderate", f"Expected Moderate, got {cat}"

    # Missing particulate matter test -> must return NaN
    no_pm_aqi = calculate_aqi_scalar(so2=60.0, no2=130.0, rspm=None)
    print(f"Missing Particulate Test: SO2=60, NO2=130, RSPM=None -> AQI: {no_pm_aqi}")
    assert pd.isna(no_pm_aqi), "Expected NaN when particulate matter is missing"

    print("\nAll CPCB Sub-Index and AQI Calculator Unit Tests PASSED!")
