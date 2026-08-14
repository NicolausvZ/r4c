import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
locations = pd.read_csv('data/sensor_locations.csv')

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15

col = [c for c in df_hourly.columns if 'Koivutaival' in str(c)][0]
lat, lon = 60.32798, 25.03485

era5_t = t2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()
obs = df_hourly[col].dropna()

print("=== RMSE vs applied shift to observation timestamps ===")
print("(negative = obs treated as ahead of UTC, i.e. local time)\n")

best = None
for shift in range(-6, 7):
    shifted = obs.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=shift)
    pair = pd.DataFrame({'era5': era5_t, 'obs': shifted}).dropna()
    if len(pair) < 100:
        continue
    bias = pair['obs'] - pair['era5']
    rmse = np.sqrt((bias ** 2).mean())
    corr = pair['era5'].corr(pair['obs'])
    flag = ""
    if best is None or rmse < best[1]:
        best = (shift, rmse)
    print(f"  shift {shift:+d} h :  RMSE {rmse:6.3f} °C   r {corr:.4f}   "
          f"mean {bias.mean():+.2f}   max {bias.max():5.2f}")

print(f"\nBest alignment: {best[0]:+d} h  (RMSE {best[1]:.3f} °C)")
if best[0] == 0:
    print("-> timestamps already aligned; the bias is physical, not a phase error")
else:
    print(f"-> observations appear offset by {best[0]:+d} h relative to ERA5 (UTC)")
    print("   Finland is UTC+3 in summer (EEST)")

# where do the extreme events sit in time?
pair0 = pd.DataFrame({'era5': era5_t, 'obs': obs}).dropna()
bias0 = pair0['obs'] - pair0['era5']
extreme_days = bias0[bias0 > 8].index.normalize().value_counts().sort_index()
print("\n=== Days containing bias >8 °C (unshifted) ===")
print(extreme_days.to_string())

# visual check on the worst day
if len(extreme_days):
    worst_day = extreme_days.idxmax()
    win = pair0.loc[str(worst_day.date())]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(win.index, win['era5'], 'o-', label='ERA5 (UTC)', linewidth=2)
    ax.plot(win.index, win['obs'], 's-', label='FVH sensor', linewidth=2)
    for sh, style in [(3, '--'), (-3, ':')]:
        s = obs.copy()
        s.index = s.index + pd.Timedelta(hours=sh)
        s = s.loc[str(worst_day.date())]
        ax.plot(s.index, s.values, style, alpha=0.8, label=f'FVH shifted {sh:+d} h')
    ax.set_title(f'Worst-bias day: {worst_day.date()}\n'
                 'if a shifted line overlays ERA5, the offset is confirmed')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/time_alignment_check.png', dpi=150)
    plt.show()
    print("\nSaved: data/time_alignment_check.png")