"""
High-Speed Ingestion and Aggregation Module for Kaggle 2010-2024 CAAQMS Dataset.

Extracts all 546 stations from the downloaded archive zip, maps to city/state metadata,
computes daily station averages, and aggregates multi-station cities to clean daily records.
"""

import os
import zipfile
import io
import time
import pandas as pd
import numpy as np

ZIP_PATH = r"C:\Users\LENOVO\Downloads\archive (2).zip"
OUTPUT_CSV_PATH = os.path.join("data", "data.csv")
SNAPSHOT_PARQUET_PATH = os.path.join("data", "city_snapshots.parquet")
SNAPSHOT_CSV_GZ_PATH = os.path.join("data", "city_snapshots.csv.gz")


def ingest_from_zip(zip_path: str = ZIP_PATH) -> pd.DataFrame:
    """
    Stream and aggregate all station CSVs from the zip archive into daily city records.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip archive not found at: {zip_path}")

    start_time = time.time()
    print("=" * 70)
    print("      INGESTING MODERN 2010-2024 INDIAN AIR QUALITY DATASET      ")
    print("=" * 70)

    with zipfile.ZipFile(zip_path, "r") as z:
        print("[Step 1/4] Reading station metadata...")
        with z.open("stations.csv") as f:
            stations_df = pd.read_csv(f)
        
        # Build mapping dictionaries
        station_city_map = dict(zip(stations_df["station_code"], stations_df["city"]))
        station_state_map = dict(zip(stations_df["station_code"], stations_df["state_code"]))

        csv_files = [n for n in z.namelist() if n.endswith(".csv") and n != "stations.csv"]
        total_files = len(csv_files)
        print(f"  -> Found {total_files} monitoring station CSV files in archive.")

        print(f"\n[Step 2/4] Streaming and computing daily averages across {total_files} stations...")
        station_records = []
        
        for idx, fname in enumerate(csv_files, 1):
            stn_code = os.path.splitext(os.path.basename(fname))[0]
            city = station_city_map.get(stn_code, stn_code)
            state = station_state_map.get(stn_code, "Unknown")

            if idx % 100 == 0 or idx == total_files:
                print(f"     Processed {idx}/{total_files} stations ({idx/total_files*100:.1f}%)...")

            try:
                with z.open(fname) as f:
                    df = pd.read_csv(
                        f,
                        usecols=lambda c: c in [
                            "timestamp", "pm2.5", "pm10", "so2", "no2", "co", "ozone", "nh3", "ws", "rh"
                        ]
                    )
                    
                    if "timestamp" not in df.columns or len(df) == 0:
                        continue

                    # Extract YYYY-MM-DD date string fast
                    df["date"] = df["timestamp"].astype(str).str[:10]
                    
                    # Convert pollutant and weather columns to numeric and sanitize negatives
                    numeric_cols = [c for c in ["pm2.5", "pm10", "so2", "no2", "co", "ozone", "nh3", "ws", "rh"] if c in df.columns]
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        df.loc[df[col] < 0, col] = np.nan

                    # Groupby date to get daily station average
                    daily_station = df.groupby("date")[numeric_cols].mean().reset_index()
                    daily_station["location"] = city
                    daily_station["state"] = state
                    daily_station["stn_code"] = stn_code

                    station_records.append(daily_station)
            except Exception as e:
                # Silently skip corrupted single station files
                continue

    print(f"\n[Step 3/4] Merging all station records and aggregating to City-Level daily means...")
    all_stations_df = pd.concat(station_records, ignore_index=True)
    
    # Rename standard columns for consistency
    rename_dict = {
        "pm2.5": "pm2_5",
        "pm10": "rspm"
    }
    all_stations_df = all_stations_df.rename(columns=rename_dict)

    # Valid dates only
    all_stations_df["date"] = pd.to_datetime(all_stations_df["date"], errors="coerce")
    all_stations_df = all_stations_df.dropna(subset=["date", "location"]).copy()

    # Aggregate multiple stations per (location, state, date)
    agg_cols = [c for c in ["pm2_5", "rspm", "so2", "no2", "co", "ozone", "nh3", "ws", "rh"] if c in all_stations_df.columns]
    
    city_daily_df = (
        all_stations_df.groupby(["location", "state", "date"])[agg_cols]
        .mean()
        .reset_index()
    )

    # Sort chronologically by location and date
    city_daily_df = city_daily_df.sort_values(["location", "date"]).reset_index(drop=True)

    print(f"\n[Step 4/4] Saving clean daily aggregated dataset to {OUTPUT_CSV_PATH}...")
    os.makedirs("data", exist_ok=True)
    city_daily_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    # Also save lightweight snapshots for Streamlit cloud deployment
    latest_per_city = city_daily_df.groupby("location").tail(120).reset_index(drop=True)
    latest_per_city.to_parquet(SNAPSHOT_PARQUET_PATH, index=False)
    latest_per_city.to_csv(SNAPSHOT_CSV_GZ_PATH, index=False, compression="gzip")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"  -> INGESTION COMPLETED IN {elapsed:.1f} SECONDS!")
    print(f"  -> Total Daily City Records: {len(city_daily_df):,}")
    print(f"  -> Unique Cities:            {city_daily_df['location'].nunique()}")
    print(f"  -> Unique States:            {city_daily_df['state'].nunique()}")
    print(f"  -> Date Range:               {city_daily_df['date'].min().date()} to {city_daily_df['date'].max().date()}")
    print("=" * 70)

    return city_daily_df


if __name__ == "__main__":
    ingest_from_zip()
