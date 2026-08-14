import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
locations = pd.read_csv('data/sensor_locations.csv')

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15

temp_cols = [c for c in df_hourly.columns if 'temperature' in str(c).lower()]

# ---- build bias series for every sensor ----
bias_frames = {}
meta = []
for col in temp_cols:
    name = str(col).split('_')[-1]
    district = "Koivukyla" if "Koivukyl" in str(col) else "Laajasalo"

    match = None
    for _, row in locations.iterrows():
        if name.lower() in str(row['name']).lower() or str(row['name']).lower() in name.lower():
            match = row
            break
    lat, lon = (match['lat'], match['lon']) if match is not None else (
        (60.32, 25.06) if district == "Koivukyla" else (60.18, 25.05))

    era5_t = t2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()
    obs = df_hourly[col].dropna()
    pair = pd.DataFrame({'era5': era5_t, 'obs': obs}).dropna()
    if len(pair) < 100:
        continue

    bias = pair['obs'] - pair['era5']
    bias_frames[name] = bias
    meta.append({
        'sensor': name, 'district': district,
        'mean_bias': bias.mean(), 'max_bias': bias.max(), 'min_bias': bias.min(),
        'p99': bias.quantile(0.99),
    })

meta_df = pd.DataFrame(meta).sort_values('max_bias', ascending=False)
print("=== Bias summary per sensor (sorted by max) ===")
print(meta_df.to_string(index=False))

# ---- how extreme is the tail, and where does it occur? ----
all_bias = pd.concat(bias_frames.values())
print(f"\nBias >8C occurs in {(all_bias > 8).sum()} of {len(all_bias)} obs "
      f"({(all_bias > 8).mean()*100:.2f}%)")
extreme_hours = all_bias[all_bias > 8].index.hour
if len(extreme_hours):
    print("Hour-of-day distribution of bias >8C:")
    print(pd.Series(extreme_hours).value_counts().sort_index().to_string())

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) mean diurnal bias cycle, one line per sensor
ax = axes[0]
for name, bias in bias_frames.items():
    cycle = bias.groupby(bias.index.hour).mean()
    colour = 'tab:blue' if 'Koivukyl' in name or name in meta_df[
        meta_df.district == 'Koivukyla'].sensor.values else 'tab:red'
    ax.plot(cycle.index, cycle.values, linewidth=1, alpha=0.75, color=colour)
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('Hour of day (UTC)')
ax.set_ylabel('Mean bias, sensor - ERA5 (°C)')
ax.set_title('(a) Diurnal bias cycle\nmidday hump = radiation error,'
             ' night hump = urban heat island')
ax.set_xticks(range(0, 24, 3))
ax.grid(alpha=0.3)

# (b) spread of bias by hour, pooled
ax = axes[1]
pooled = pd.DataFrame({'bias': all_bias.values, 'hour': all_bias.index.hour})
pooled.boxplot(column='bias', by='hour', ax=ax, grid=False,
               flierprops=dict(marker='.', markersize=2, alpha=0.3))
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlabel('Hour of day (UTC)')
ax.set_ylabel('Bias (°C)')
ax.set_title('(b) Bias distribution by hour (all sensors)')
plt.suptitle('')

# (c) per-sensor max bias
ax = axes[2]
colours = meta_df['district'].map({'Koivukyla': 'tab:blue', 'Laajasalo': 'tab:red'})
ax.barh(meta_df['sensor'], meta_df['max_bias'], color=colours, alpha=0.8)
ax.set_xlabel('Maximum bias (°C)')
ax.set_title('(c) Peak bias per sensor\nisolated spikes = siting problem')
ax.tick_params(axis='y', labelsize=7)
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('data/bias_diagnosis.png', dpi=150)
plt.show()
print("\nSaved: data/bias_diagnosis.png")