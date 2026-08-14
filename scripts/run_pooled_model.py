import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata, match_sensor

LANDCOVER_VAR = 'built_100m'   # strongest single predictor, r = +0.80

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
meta = load_sensor_metadata()
lc = pd.read_csv('data/sensor_landcover.csv')

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15
d2m_grid = era5['d2m'] - 273.15

# ---- build one long table: every sensor-hour is a row ----
frames = []
for name in df_hourly.columns:
    m = match_sensor(name, meta)
    lc_row = lc[lc['name'] == name]
    if m is None or lc_row.empty:
        print(f"  skipping {name}: missing metadata or land cover")
        continue

    lat, lon = m['lat'], m['lon']
    era5_t = t2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()
    era5_d = d2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()

    d = pd.DataFrame({'era5_t2m': era5_t, 'era5_d2m': era5_d,
                      'obs': df_hourly[name]}).dropna()
    if len(d) < 200:
        continue

    d['bias'] = d['obs'] - d['era5_t2m']
    h = d.index.hour
    d['hour_sin'] = np.sin(2 * np.pi * h / 24)
    d['hour_cos'] = np.cos(2 * np.pi * h / 24)
    d['built'] = lc_row.iloc[0][LANDCOVER_VAR]
    d['sensor'] = name
    d['district'] = m['district']
    frames.append(d)

pool = pd.concat(frames)
print(f"Pooled dataset: {len(pool)} rows, {pool['sensor'].nunique()} sensors")
print(f"built_100m range: {pool['built'].min():.3f} to {pool['built'].max():.3f}")

FEATURES = ['era5_t2m', 'era5_d2m', 'hour_sin', 'hour_cos', 'built']
X = pool[FEATURES]
y = pool['bias']
groups = pool['sensor']

baseline_rmse = np.sqrt((y ** 2).mean())
print(f"\nBASELINE raw ERA5 RMSE: {baseline_rmse:.3f} °C")

# ---- leave-one-sensor-out: can we predict where we have NO sensor? ----
# GroupKFold with n = n_sensors holds out one entire sensor per fold.
n_sensors = pool['sensor'].nunique()
cv = GroupKFold(n_splits=n_sensors)

models = {
    'Ridge': Ridge(alpha=1.0),
    'RF': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

print("\n=== Leave-one-sensor-out (predicting an unseen location) ===")
preds = {}
for mname, model in models.items():
    yhat = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
    preds[mname] = yhat
    rmse = np.sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    print(f"  {mname:6s} RMSE {rmse:.3f} °C   R2 {r2:+.3f}   "
          f"gain {baseline_rmse - rmse:+.3f} °C")

# ---- per-sensor breakdown for the better model ----
best_name = min(preds, key=lambda k: np.sqrt(mean_squared_error(y, preds[k])))
pool['pred'] = preds[best_name]

print(f"\n=== Per-sensor, held out entirely ({best_name}) ===")
rows = []
for name, g in pool.groupby('sensor'):
    base = np.sqrt((g['bias'] ** 2).mean())
    rmse = np.sqrt(mean_squared_error(g['bias'], g['pred']))
    rows.append({'sensor': name, 'district': g['district'].iloc[0],
                 'built': g['built'].iloc[0], 'mean_bias': g['bias'].mean(),
                 'baseline_rmse': base, 'rmse': rmse, 'gain': base - rmse})

bysensor = pd.DataFrame(rows).sort_values('gain', ascending=False)
print(bysensor.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

n_better = (bysensor['gain'] > 0).sum()
print(f"\nBeats raw ERA5 at {n_better}/{len(bysensor)} unseen sensors")

# ---- what is the model using? ----
rf = models['RF'].fit(X, y)
print("\n=== RF feature importance (pooled) ===")
for f, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda kv: -kv[1]):
    print(f"  {f:12s} {imp:.3f}")

ridge = models['Ridge'].fit(X, y)
print("\n=== Ridge coefficients ===")
for f, c in zip(FEATURES, ridge.coef_):
    print(f"  {f:12s} {c:+.4f}")

bysensor.to_csv('data/pooled_model_results.csv', index=False)
print("\nSaved: data/pooled_model_results.csv")