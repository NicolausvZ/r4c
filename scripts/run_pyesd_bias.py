import numpy as np
import pandas as pd
import xarray as xr
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata, match_sensor

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
meta = load_sensor_metadata()

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15
d2m_grid = era5['d2m'] - 273.15

# columns are now clean sensor names, one per sensor
sensor_cols = list(df_hourly.columns)
print(f"Sensors: {len(sensor_cols)}")

cv = KFold(n_splits=5, shuffle=False)
scoring = ['neg_root_mean_squared_error', 'r2']

rows = []
for name in sensor_cols:
    m = match_sensor(name, meta)
    if m is None:
        print(f"  skipping {name}: no metadata match")
        continue
    lat, lon, district = m['lat'], m['lon'], m['district']

    era5_t = t2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()
    era5_d = d2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()

    d = pd.DataFrame({'era5_t2m': era5_t, 'era5_d2m': era5_d,
                      'obs': df_hourly[name]}).dropna()
    if len(d) < 200:
        print(f"  skipping {name}: only {len(d)} obs")
        continue

    d['bias'] = d['obs'] - d['era5_t2m']
    h = d.index.hour
    d['hour_sin'] = np.sin(2 * np.pi * h / 24)
    d['hour_cos'] = np.cos(2 * np.pi * h / 24)

    X = d[['era5_t2m', 'era5_d2m', 'hour_sin', 'hour_cos']]
    y = d['bias']
    baseline = np.sqrt((y ** 2).mean())

    res = {'sensor': name, 'district': district, 'n': len(d),
           'mean_bias': y.mean(), 'baseline_rmse': baseline}

    for mname, model in [('RF', RandomForestRegressor(n_estimators=200,
                                                      random_state=42, n_jobs=-1)),
                         ('Ridge', Ridge(alpha=1.0))]:
        cvr = cross_validate(model, X, y, cv=cv, scoring=scoring,
                             return_train_score=True)
        rmse = -cvr['test_neg_root_mean_squared_error']
        res[f'{mname}_rmse'] = rmse.mean()
        res[f'{mname}_gain'] = baseline - rmse.mean()
        res[f'{mname}_train_r2'] = cvr['train_r2'].mean()

    rows.append(res)

if not rows:
    raise SystemExit("No sensors produced results — check data/fvh_hourly.pkl")

out = pd.DataFrame(rows).sort_values('RF_gain', ascending=False)
pd.set_option('display.width', 200)
print()
print(out.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

print("\n=== Summary (positive gain = beats raw ERA5) ===")
print(f"  sensors            : {len(out)}")
print(f"  mean baseline RMSE : {out['baseline_rmse'].mean():.3f} °C")
for mname in ['RF', 'Ridge']:
    print(f"  {mname:6s} mean RMSE   : {out[f'{mname}_rmse'].mean():.3f} °C   "
          f"gain {out[f'{mname}_gain'].mean():+.3f} °C   "
          f"beats baseline at {(out[f'{mname}_gain'] > 0).sum()}/{len(out)} sensors")

out.to_csv('data/model_results_by_sensor.csv', index=False)
print("\nSaved: data/model_results_by_sensor.csv")