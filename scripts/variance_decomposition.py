import numpy as np
import pandas as pd
import xarray as xr
import os, sys
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata, match_sensor

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
meta = load_sensor_metadata()
lc = pd.read_csv('data/sensor_landcover.csv')

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15
d2m_grid = era5['d2m'] - 273.15

frames = []
for name in df_hourly.columns:
    m = match_sensor(name, meta)
    lc_row = lc[lc['name'] == name]
    if m is None or lc_row.empty:
        continue
    era5_t = t2m_grid.sel(latitude=m['lat'], longitude=m['lon'], method='nearest').to_series()
    era5_d = d2m_grid.sel(latitude=m['lat'], longitude=m['lon'], method='nearest').to_series()
    d = pd.DataFrame({'era5_t2m': era5_t, 'era5_d2m': era5_d,
                      'obs': df_hourly[name]}).dropna()
    if len(d) < 200:
        continue
    d['bias'] = d['obs'] - d['era5_t2m']
    h = d.index.hour
    d['hour_sin'] = np.sin(2*np.pi*h/24); d['hour_cos'] = np.cos(2*np.pi*h/24)
    d['built'] = lc_row.iloc[0]['built_100m']
    d['water'] = lc_row.iloc[0]['water_5km']
    d['sensor'] = name
    frames.append(d)

pool = pd.concat(frames)
bias = pool['bias']

# --- how much of bias variance is site-constant vs time-varying? ---
site_mean = pool.groupby('sensor')['bias'].transform('mean')
time_mean = pool.groupby(pool.index)['bias'].transform('mean')
grand = bias.mean()

var_total = bias.var()
var_site = site_mean.var()
var_time = time_mean.var()
resid = bias - site_mean - time_mean + grand

print("=== Variance decomposition of ERA5 bias ===")
print(f"  total variance        : {var_total:.4f}  (sd {np.sqrt(var_total):.3f} °C)")
print(f"  between-sensor (site) : {var_site:.4f}  ({var_site/var_total*100:5.1f}%)")
print(f"  between-hour  (time)  : {var_time:.4f}  ({var_time/var_total*100:5.1f}%)")
print(f"  residual              : {resid.var():.4f}  ({resid.var()/var_total*100:5.1f}%)")

print("\n  -> A model using only static site covariates can address at most")
print(f"     the site component: {var_site/var_total*100:.1f}% of bias variance.")
best_possible = np.sqrt(mean_squared_error(bias, site_mean))
print(f"  -> Perfect site-offset model RMSE: {best_possible:.3f} °C "
      f"vs baseline {np.sqrt((bias**2).mean()):.3f} °C")

# --- permutation test: is the observed skill distinguishable from chance? ---
FEATURES = ['era5_t2m','era5_d2m','hour_sin','hour_cos','built','water']
midpoint = pool.index.min() + (pool.index.max() - pool.index.min()) / 2

def strict_gain(data):
    gains = []
    for held in data['sensor'].unique():
        tr = data[(data['sensor'] != held) & (data.index < midpoint)]
        te = data[(data['sensor'] == held) & (data.index >= midpoint)]
        if len(te) < 50:
            continue
        mod = Ridge(alpha=1.0).fit(tr[FEATURES], tr['bias'])
        pred = mod.predict(te[FEATURES])
        gains.append(np.sqrt((te['bias']**2).mean()) -
                     np.sqrt(mean_squared_error(te['bias'], pred)))
    return float(np.mean(gains))

observed = strict_gain(pool)
print(f"\n=== Permutation test (100 shuffles of site covariates) ===")
print(f"  observed gain: {observed:+.4f} °C")

smap = pool.groupby('sensor')[['built','water']].first()
null = []
for i in range(100):
    sh = smap.sample(frac=1, random_state=i).set_index(smap.index)
    p = pool.copy()
    p['built'] = p['sensor'].map(sh['built'])
    p['water'] = p['sensor'].map(sh['water'])
    null.append(strict_gain(p))

null = np.array(null)
pval = (null >= observed).mean()
print(f"  null mean    : {null.mean():+.4f} °C  (sd {null.std():.4f})")
print(f"  null range   : {null.min():+.4f} to {null.max():+.4f}")
print(f"  p-value      : {pval:.3f}")
print("  -> land cover adds real skill" if pval < 0.05
      else "  -> land cover skill NOT distinguishable from chance")