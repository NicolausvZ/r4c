import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("Loading ERA5 summer 2024...")
era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
temp_c = era5["t2m"] - 273.15

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

# Load sensor locations
with open("data/r4c_fvh_all_latest.geojson") as f:
    geo = json.load(f)

sensors = {}
for feature in geo["features"]:
    name = feature["properties"].get("name", "")
    district = feature["properties"].get("district", "")
    coords = feature["geometry"]["coordinates"]
    sensors[name] = {"lon": coords[0], "lat": coords[1], "district": district}

# Match all sensors
results = []
for col in temp_cols:
    sensor_name = str(col).split('_')[-1]
    district = "Koivukylä" if "Koivukyl" in col else "Laajasalo"

    # Find nearest sensor location
    match = None
    for name, loc in sensors.items():
        if sensor_name.lower() in name.lower() or name.lower() in sensor_name.lower():
            match = loc
            break

    # Fall back to district centroid if no match
    if match is None:
        if district == "Koivukylä":
            match = {"lon": 25.06, "lat": 60.32, "district": district}
        else:
            match = {"lon": 25.05, "lat": 60.18, "district": district}

    # Extract ERA5 at this location
    era5_point = temp_c.sel(
        latitude=match['lat'],
        longitude=match['lon'],
        method='nearest'
    )
    era5_df = era5_point.to_dataframe(name='era5_temp').reset_index()
    era5_df = era5_df.rename(columns={'valid_time': 'datetime'})
    era5_df = era5_df.set_index('datetime')['era5_temp']

    fvh_series = df_hourly[col].dropna()
    aligned = pd.DataFrame({'era5': era5_df, 'fvh': fvh_series}).dropna()

    if len(aligned) > 10:
        corr = aligned['era5'].corr(aligned['fvh'])
        bias = (aligned['fvh'] - aligned['era5']).mean()
        rmse = np.sqrt(((aligned['fvh'] - aligned['era5'])**2).mean())
        results.append({
            'sensor': sensor_name,
            'district': district,
            'correlation': corr,
            'bias': bias,
            'rmse': rmse,
            'n': len(aligned)
        })

results_df = pd.DataFrame(results)
print("\n=== Results for all sensors ===")
print(results_df.to_string(index=False))

# Plot bias by sensor
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = results_df['district'].map({'Koivukylä': 'blue', 'Laajasalo': 'red'})

# Bias plot
axes[0].barh(results_df['sensor'], results_df['bias'], color=colors, alpha=0.7)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('Bias (FVH - ERA5) °C')
axes[0].set_title('Temperature Bias per Sensor\n(positive = sensor warmer than ERA5)')
axes[0].tick_params(axis='y', labelsize=7)

# RMSE plot
axes[1].barh(results_df['sensor'], results_df['rmse'], color=colors, alpha=0.7)
axes[1].set_xlabel('RMSE (°C)')
axes[1].set_title('RMSE per Sensor')
axes[1].tick_params(axis='y', labelsize=7)

# Legend
from matplotlib.patches import Patch
legend = [Patch(color='blue', label='Koivukylä'), Patch(color='red', label='Laajasalo')]
axes[0].legend(handles=legend)

plt.tight_layout()
plt.savefig("data/era5_fvh_bias.png", dpi=150)
plt.show()
print("\nSaved to data/era5_fvh_bias.png")