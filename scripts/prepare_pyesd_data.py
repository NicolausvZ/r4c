import pandas as pd
import numpy as np
import xarray as xr
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata, match_sensor, sensor_name_from_column

os.makedirs('data/pyesd_era5', exist_ok=True)
os.makedirs('data/pyesd_cache', exist_ok=True)

INSTALL_BUFFER_DAYS = 1
FLAT_DAY_THRESHOLD = 1.0   # °C
MIN_HOURS_FOR_FLAT_TEST = 12

# ---- sensor metadata ----
meta = load_sensor_metadata()
meta.to_csv('data/sensor_locations.csv', index=False)
print(f"Sensor metadata: {len(meta)} sensors")
overrides = meta[meta['name_source'] == 'override']
for _, r in overrides.iterrows():
    print(f"  named by override: {r['sensor_id']} -> '{r['name']}'")
print(f"Install dates: {meta['installed'].min().date()} -> {meta['installed'].max().date()}")

# ---- ERA5 -> one DataArray per variable ----
print("\nLoading ERA5 summer 2024...")
era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})

for var in ['t2m', 'd2m']:
    da = era5[var].astype('float32')
    da = da.drop_vars([c for c in ['number', 'expver'] if c in da.coords])
    da.to_netcdf(f'data/pyesd_era5/{var}.nc')
    print(f"  saved {var}.nc  start={str(da.time.values[0])[:16]}")

print(f"First timestamp is month start: {pd.Timestamp(era5.time.values[0]).is_month_start}")

# ---- FVH sensors ----
print("\nLoading FVH sensor data...")
df = pd.read_excel('data/weather_data.xlsx',
                   sheet_name='fvh_vtt_10_min_laajasalo',
                   skiprows=range(1, 77000), nrows=50000)
df = df.replace(-999.0, np.nan)
df['datetime'] = pd.to_datetime(df['Date time'])
df = df[(df['datetime'] >= '2024-06-01') & (df['datetime'] <= '2024-08-31')]

temp_cols = [c for c in df.columns if 'temperature' in str(c).lower()]
df_hourly = df.set_index('datetime')[temp_cols].resample('1h').mean()

# rename columns to clean sensor names so downstream joins are trivial
rename_map, unmatched = {}, []
for col in temp_cols:
    raw = sensor_name_from_column(col)
    m = match_sensor(raw, meta)
    if m is not None:
        rename_map[col] = m['name']
    else:
        rename_map[col] = raw
        unmatched.append(raw)

df_hourly = df_hourly.rename(columns=rename_map)
sensor_cols = list(rename_map.values())
print(f"Raw hourly: {df_hourly.shape}")
if unmatched:
    print(f"  WARNING unmatched sensors: {unmatched}")
else:
    print("  all sensor columns matched to metadata")

# ---- QC 1: mask pre-installation readings ----
print("\nMasking pre-installation readings...")
default_install = meta['installed'].max()
masked_pre = 0

for name in sensor_cols:
    m = match_sensor(name, meta)
    install = m['installed'] if m is not None and pd.notna(m['installed']) else default_install
    cutoff = install + pd.Timedelta(days=INSTALL_BUFFER_DAYS)
    before = df_hourly[name].notna().sum()
    df_hourly.loc[df_hourly.index < cutoff, name] = np.nan
    masked_pre += before - df_hourly[name].notna().sum()

print(f"  masked {masked_pre} readings before install date")

# ---- QC 2: mask flat days (sensor not measuring outdoor air) ----
print(f"\nFlagging flat days (range < {FLAT_DAY_THRESHOLD} °C, "
      f"min {MIN_HOURS_FOR_FLAT_TEST} h of data)...")
masked_flat = 0

for name in sensor_cols:
    daily = df_hourly[name].groupby(df_hourly.index.normalize())
    day_range = daily.max() - daily.min()
    day_count = daily.count()
    flat = day_range[(day_range < FLAT_DAY_THRESHOLD) &
                     (day_count >= MIN_HOURS_FOR_FLAT_TEST)].index
    if len(flat):
        mask = df_hourly.index.normalize().isin(flat)
        before = df_hourly[name].notna().sum()
        df_hourly.loc[mask, name] = np.nan
        removed = before - df_hourly[name].notna().sum()
        masked_flat += removed
        print(f"  {name}: {len(flat)} flat day(s), {removed} readings removed")

if masked_flat == 0:
    print("  none found")

# ---- report ----
valid = df_hourly.notna().sum()
first_valid = df_hourly.apply(lambda s: s.first_valid_index())

print("\n=== Retained data per sensor ===")
for name in sensor_cols:
    fv = first_valid[name]
    print(f"  {name:28s} {valid[name]:5d} obs   from {str(fv)[:16] if fv is not None else 'n/a'}")

df_hourly.to_pickle('data/fvh_hourly.pkl')
print(f"\nSaved hourly FVH: {df_hourly.shape}  ({int(valid.sum())} valid readings)")