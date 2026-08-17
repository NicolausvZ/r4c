import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os, sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata, match_sensor

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
meta = load_sensor_metadata()
lc = pd.read_csv('data/sensor_landcover.csv')

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15

frames = []
for name in df_hourly.columns:
    m = match_sensor(name, meta)
    lc_row = lc[lc['name'] == name]
    if m is None or lc_row.empty:
        continue
    era5_t = t2m_grid.sel(latitude=m['lat'], longitude=m['lon'], method='nearest').to_series()
    d = pd.DataFrame({'era5': era5_t, 'obs': df_hourly[name]}).dropna()
    if len(d) < 200:
        continue
    d['bias'] = d['obs'] - d['era5']
    d['sensor'] = name
    d['built'] = lc_row.iloc[0]['built_100m']
    d['water'] = lc_row.iloc[0]['water_5km']
    frames.append(d)

pool = pd.concat(frames)
pool['hour'] = pool.index.hour          # UTC; Helsinki summer = UTC+3
pool['local_hour'] = (pool['hour'] + 3) % 24

# ---------------------------------------------------------------
# 1. Is the UHI-landcover relationship stronger at night?
# ---------------------------------------------------------------
print("=== Correlation (built_100m vs mean bias) by local hour ===")
print(" hour    r      p       n_sensors   mean_bias_spread")
by_hour = []
for h in range(24):
    sub = pool[pool['local_hour'] == h]
    agg = sub.groupby('sensor').agg(bias=('bias', 'mean'), built=('built', 'first'))
    if len(agg) < 10:
        continue
    r, p = stats.pearsonr(agg['built'], agg['bias'])
    spread = agg['bias'].max() - agg['bias'].min()
    by_hour.append({'hour': h, 'r': r, 'p': p, 'spread': spread,
                    'mean_bias': agg['bias'].mean()})
    print(f"  {h:2d}   {r:+.3f}  {p:.4f}     {len(agg):2d}        {spread:.2f} °C")

hr = pd.DataFrame(by_hour)
best_hour = hr.loc[hr['r'].idxmax()]
print(f"\nStrongest at local hour {int(best_hour['hour'])}: "
      f"r = {best_hour['r']:+.3f}, spread {best_hour['spread']:.2f} °C")

# ---------------------------------------------------------------
# 2. Nocturnal UHI intensity per sensor (22:00-04:00 local)
# ---------------------------------------------------------------
night = pool[(pool['local_hour'] >= 22) | (pool['local_hour'] <= 4)]
day = pool[(pool['local_hour'] >= 10) & (pool['local_hour'] <= 16)]

uhi = pd.DataFrame({
    'night_bias': night.groupby('sensor')['bias'].mean(),
    'day_bias': day.groupby('sensor')['bias'].mean(),
    'all_bias': pool.groupby('sensor')['bias'].mean(),
    'built': pool.groupby('sensor')['built'].first(),
    'water': pool.groupby('sensor')['water'].first(),
})
uhi['night_minus_day'] = uhi['night_bias'] - uhi['day_bias']
uhi = uhi.sort_values('night_bias', ascending=False)

print("\n=== UHI intensity per sensor ===")
print(uhi.to_string(float_format=lambda v: f"{v:7.3f}"))

for label, col in [('all hours', 'all_bias'), ('night 22-04', 'night_bias'),
                   ('day 10-16', 'day_bias')]:
    r, p = stats.pearsonr(uhi['built'], uhi[col])
    print(f"\n  built vs {label:12s}: r = {r:+.3f}  (p = {p:.4f})")

# ---------------------------------------------------------------
# 3. Can we predict a sensor's UHI intensity from land cover alone?
#    Leave-one-out over 20 sensors — the site-level question.
# ---------------------------------------------------------------
print("\n=== Leave-one-sensor-out prediction of NOCTURNAL UHI intensity ===")
for feats in [['built'], ['built', 'water']]:
    X = uhi[feats].values
    y = uhi['night_bias'].values
    preds = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        preds[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    rmse = np.sqrt(((y - preds) ** 2).mean())
    null_rmse = np.sqrt(((y - y.mean()) ** 2).mean())
    r_pred = np.corrcoef(y, preds)[0, 1]
    rho = stats.spearmanr(y, preds).correlation
    print(f"  {str(feats):22s} RMSE {rmse:.3f} °C  (predict-the-mean {null_rmse:.3f})"
          f"  r {r_pred:+.3f}  rank-rho {rho:+.3f}")

# ---------------------------------------------------------------
# 4. Hotspot rank stability — the planning-relevant question
# ---------------------------------------------------------------
print("\n=== Hotspot rank-order stability ===")
halves = np.array_split(np.sort(night.index.unique()), 2)
r1 = night[night.index.isin(halves[0])].groupby('sensor')['bias'].mean()
r2 = night[night.index.isin(halves[1])].groupby('sensor')['bias'].mean()
rho_split = stats.spearmanr(r1, r2).correlation
print(f"  rank correlation between first/second half of record: rho = {rho_split:+.3f}")
top5_1, top5_2 = set(r1.nlargest(5).index), set(r2.nlargest(5).index)
print(f"  top-5 hottest sensors agree on {len(top5_1 & top5_2)}/5 between halves")

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(hr['hour'], hr['r'], 'o-', color='tab:red')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Local hour'); ax.set_ylabel('r (built vs bias)')
ax.set_title('(a) Land-cover control on bias\nby time of day')
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(uhi['built'], uhi['night_bias'], s=60, label='night 22-04', color='navy')
ax.scatter(uhi['built'], uhi['day_bias'], s=60, label='day 10-16',
           color='orange', marker='^')
for lbl, row in uhi.iterrows():
    ax.annotate(str(lbl)[:14], (row['built'], row['night_bias']),
                fontsize=6, xytext=(3, 3), textcoords='offset points')
b = np.linspace(0, uhi['built'].max(), 50)
fit = np.polyfit(uhi['built'], uhi['night_bias'], 1)
ax.plot(b, np.polyval(fit, b), '--', color='navy', alpha=0.7)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Built fraction within 100 m'); ax.set_ylabel('Mean bias (°C)')
ax.set_title(f'(b) UHI vs built fraction\nnight slope {fit[0]:+.2f} °C per unit built')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
diurnal = pool.groupby(['sensor', 'local_hour'])['bias'].mean().unstack()
order = uhi.index
cmap = plt.cm.RdYlBu_r
for i, s in enumerate(order):
    if s in diurnal.index:
        ax.plot(diurnal.columns, diurnal.loc[s],
                color=cmap(1 - i / len(order)), lw=1.2)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Local hour'); ax.set_ylabel('Mean bias (°C)')
ax.set_title('(c) Diurnal bias cycle\nred = most built, blue = least')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/uhi_analysis.png', dpi=150)
plt.show()

uhi.to_csv('data/uhi_intensity.csv')
print("\nSaved: data/uhi_analysis.png, data/uhi_intensity.csv")