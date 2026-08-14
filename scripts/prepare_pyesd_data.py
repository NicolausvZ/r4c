import pandas as pd
import numpy as np
import xarray as xr
import os

os.makedirs('data/pyesd_stations', exist_ok=True)
os.makedirs('data/pyesd_predictors', exist_ok=True)

print("Loading FVH sensor data...")
df = pd.read_excel('data/weather_data.xlsx',
                   sheet_name='fvh_vtt_10_min_laajasalo',
                   skiprows=range(1, 77000),
                   nrows=50000)

df = df.replace(-999.0, np.nan)
df['datetime'] = pd.to_datetime(df['Date time'])
df = df[(df['datetime'] >= '2024-06-18') & (df['datetime'] <= '2024-08-31')]

temp_cols = [c for c in df.columns if 'temperature' in str(c).lower()]
df_hourly = df.set_index('datetime')[temp_cols].resample('1h').mean()

print("Exporting sensor CSVs...")
for col in temp_cols:
    sensor_name = str(col).split('_')[-1]
    sensor_name_clean = sensor_name.replace(' ', '_').replace('/', '_')
    series = df_hourly[col].dropna()
    out = pd.DataFrame({'Date': series.index, 'Temperature': series.values})
    out.to_csv(f'data/pyesd_stations/{sensor_name_clean}.csv', index=False)
    print(f"  Saved: {sensor_name_clean}.csv ({len(out)} rows)")

print("\nLoading ERA5 predictors...")
era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
t2m = era5['t2m'].sel(latitude=60.25, longitude=24.75, method='nearest') - 273.15
d2m = era5['d2m'].sel(latitude=60.25, longitude=24.75, method='nearest') - 273.15

predictors = pd.DataFrame({'time': era5.valid_time.values, 't2m': t2m.values, 'd2m': d2m.values})
predictors = predictors.set_index('time')
predictors_std = (predictors - predictors.mean()) / predictors.std()
predictors_std.index.name = 'time'
predictors_std.to_csv('data/pyesd_predictors/era5_predictors.csv')
print(f"Saved ERA5 predictors: {len(predictors_std)} rows")

print("\nCreating stationnames.csv...")
station_files = [f for f in os.listdir('data/pyesd_stations') if f.endswith('.csv') and f != 'stationnames.csv']
station_files.sort()
with open('data/pyesd_stations/stationnames.csv', 'w') as f:
    f.write('nr,name\n')
    for i, fname in enumerate(station_files):
        f.write(f'{i},{fname.replace(".csv","")}\n')
print("Done!")