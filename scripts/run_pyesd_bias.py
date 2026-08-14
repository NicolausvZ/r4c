import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
locations = pd.read_csv('data/sensor_locations.csv')

era5 = xr.open_dataset("data/data_stream-oper_stepType-instant.nc", engine="netcdf4")
era5 = era5.rename({'valid_time': 'time'})
t2m_grid = era5['t2m'] - 273.15
d2m_grid = era5['d2m'] - 273.15

temp_cols = [c for c in df_hourly.columns if 'temperature' in str(c).lower()]
cv = KFold(n_splits=5, shuffle=False)
scoring = ['neg_root_mean_squared_error', 'r2']

rows = []
for col in temp_cols:
    name = str(col).split('_')[-1]
    district = "Koivukyla" if "Koivukyl" in str(col) else "Laajasalo"

    match = None
    for _, r in locations.iterrows():
        rn = str(r['name']).lower()
        if rn and (name.lower() in rn or rn in name.lower()):
            match = r
            break
    lat, lon = (match['lat'], match['lon']) if match is not None else (
        (60.32, 25.06) if district == "Koivukyla" else (60.18, 25.05))

    era5_t = t2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()
    era5_d = d2m_grid.sel(latitude=lat, longitude=lon, method='nearest').to_series()

    d = pd.DataFrame({'era5_t2m': era5_t, 'era5_d2m': era5_d,
                      'obs': df_hourly[col]}).dropna()
    if len(d) < 200:
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

out = pd.DataFrame(rows).sort_values('RF_gain', ascending=False)
pd.set_option('display.width', 200)
print(out.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

print(f"\n=== Summary (positive gain = beats raw ERA5) ===")
print(f"  mean baseline RMSE : {out['baseline_rmse'].mean():.3f} °C")
for m in ['RF', 'Ridge']:
    print(f"  {m:6s} mean RMSE   : {out[f'{m}_rmse'].mean():.3f} °C   "
          f"gain {out[f'{m}_gain'].mean():+.3f} °C   "
          f"beats baseline at {(out[f'{m}_gain'] > 0).sum()}/{len(out)} sensors")

out.to_csv('data/model_results_by_sensor.csv', index=False)
print("\nSaved: data/model_results_by_sensor.csv")