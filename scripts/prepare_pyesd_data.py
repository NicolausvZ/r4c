import pandas as pd
import numpy as np
import xarray as xr
import json, os

os.makedirs('data/pyesd_era5', exist_ok=True)
os.makedirs('data/pyesd_cache', exist_ok=True)

INSTALL_BUFFER_DAYS = 1   # skip partial install day
FLAT_DAY_THRESHOLD = 1.0  # °C; daily range below this = not outdoor air

# ---- sensor metadata: locations AND install dates ----
with open("data/r4c_fvh_all_latest.geojson") as f:
    geo = json.load(f)

locs = []
for ft in geo["features"]:
    p = ft["properties"]
    locs.append({
        "name": p.get("name", ""),
        "district": p.get("district", ""),
        "lon": ft["geometry"]["coordinates"][0],
        "lat": ft["geometry"]["coordinates"][1],
        "installed": p.get("Date_installed"),
    })
loc_df = pd.DataFrame(locs)
loc_df["installed"] = pd.to_datetime(loc_df["installed"])
loc_df.to_csv('data/sensor_locations.csv', index=False)
print(f"Sensor metadata: {len(loc_df)} sensors")
print(f"Install dates: {loc_df['installed'].min().date()} -> {loc_df['installed'].max().date()}")

# ---- ERA5 -> one DataArray per variable, time dim named 'time' ----
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
print(f"Raw hourly: {df_hourly.shape}")

# ---- QC 1: mask everything before each sensor's install date ----
print("\nMasking pre-installation readings...")
default_install = loc_df["installed"].max()
masked_pre = 0

for col in temp_cols:
    sensor_name = str(col).split('_')[-1]

    match = None
    for _, row in loc_df.iterrows():
        rn = str(row['name']).lower()
        if rn and (sensor_name.lower() in rn or rn in sensor_name.lower()):
            match = row
            break

    install = match['installed'] if match is not None and pd.notna(match['installed']) else default_install
    cutoff = install + pd.Timedelta(days=INSTALL_BUFFER_DAYS)

    n_before = df_hourly[col].notna().sum()
    df_hourly.loc[df_hourly.index < cutoff, col] = np.nan
    masked_pre += n_before - df_hourly[col].notna().sum()

print(f"  masked {masked_pre} pre-installation readings")

# ---- QC 2: mask flat days (indoor / enclosed sensor) ----
print("\nFlagging flat days (daily range < "
      f"{FLAT_DAY_THRESHOLD} °C)...")
masked_flat = 0

for col in temp_cols:
    daily = df_hourly[col].groupby(df_hourly.index.normalize())
    day_range = daily.max() - daily.min()
    flat_days = day_range[day_range < FLAT_DAY_THRESHOLD].index

    if len(flat_days):
        mask = df_hourly.index.normalize().isin(flat_days)
        n_before = df_hourly[col].notna().sum()
        df_hourly.loc[mask, col] = np.nan
        removed = n_before - df_hourly[col].notna().sum()
        masked_flat += removed
        if removed:
            print(f"  {str(col).split('_')[-1]}: {len(flat_days)} flat day(s), "
                  f"{removed} readings removed")

if masked_flat == 0:
    print("  none found")

# ---- report ----
valid = df_hourly.notna().sum()
first_valid = df_hourly.apply(lambda s: s.first_valid_index())

print(f"\n=== Retained data per sensor ===")
for col in temp_cols:
    name = str(col).split('_')[-1]
    fv = first_valid[col]
    print(f"  {name:28s} {valid[col]:5d} obs   from {str(fv)[:16] if fv is not None else 'n/a'}")

df_hourly.to_pickle('data/fvh_hourly.pkl')
print(f"\nSaved hourly FVH: {df_hourly.shape}  "
      f"({int(valid.sum())} valid readings total)")