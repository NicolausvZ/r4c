import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading FVH data from June 2024...")
df = pd.read_excel('data/weather_data.xlsx', 
                   sheet_name='fvh_vtt_10_min_laajasalo',
                   skiprows=range(1, 77000),
                   nrows=10000)

df = df.replace(-999.0, np.nan)
df['datetime'] = pd.to_datetime(df['Date time'])

temp_cols = [c for c in df.columns if 'temperature' in str(c).lower()]

# Basic stats
print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"\nTemperature statistics across all sensors:")
all_temps = df[temp_cols]
print(f"  Min: {all_temps.min().min():.1f}°C")
print(f"  Max: {all_temps.max().max():.1f}°C")
print(f"  Mean: {all_temps.mean().mean():.1f}°C")

# Plot all sensors
fig, ax = plt.subplots(figsize=(14, 5))
for col in temp_cols:
    name = str(col).split('_')[-1]
    ax.plot(df['datetime'], df[col], linewidth=0.5, alpha=0.7, label=name)

ax.set_title("FVH Sensor Network - All 20 Stations Temperature\nJune 2024")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=6, ncol=4, loc='upper right')
plt.tight_layout()
plt.savefig("data/fvh_temperature_timeseries.png", dpi=150)
plt.show()
print("\nSaved to data/fvh_temperature_timeseries.png")