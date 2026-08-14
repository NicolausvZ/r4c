import pandas as pd
import numpy as np
import os
from pyESD.Weatherstation import read_station_csv

print("Loading predictors...")
predictors = pd.read_csv('data/pyesd_predictors/era5_predictors.csv',
                         index_col='time', parse_dates=True)
print(f"Predictors shape: {predictors.shape}")

print("\nLoading station data directly (bypassing broken read_weatherstations)...")
station_dir = 'data/pyesd_stations'
station_files = [f for f in os.listdir(station_dir) if f.endswith('.csv') and f != 'stationnames.csv']

stations = {}
for fname in station_files:
    name = fname.replace('.csv', '')
    try:
        ws = read_station_csv(os.path.join(station_dir, fname), varname='Temperature')
        stations[name] = ws
        print(f"  Loaded: {name}")
    except Exception as e:
        print(f"  FAILED: {name} -> {e}")

print(f"\nStations loaded: {len(stations)}")