import numpy as np
import pandas as pd
import xarray as xr
import os, sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

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
FEATURES = ['era5_t2m','era5_d2m','hour_sin','hour_cos','built','water']
baseline = np.sqrt((pool['bias']**2).mean())
print(f"Pooled: {len(pool)} rows, {pool['sensor'].nunique()} sensors")
print(f"BASELINE RMSE: {baseline:.3f} °C\n")

# how many sensors share an ERA5 cell?
cells = pool.groupby('sensor')[['era5_t2m']].first().round(4)
print(f"Distinct ERA5 t2m values at first timestep across sensors: "
      f"{cells['era5_t2m'].nunique()} of {len(cells)}")

# split time in half: train on first half, test on second
midpoint = pool.index.min() + (pool.index.max() - pool.index.min()) / 2
print(f"Time split at {midpoint}\n")

models = {'Ridge': Ridge(alpha=1.0),
          'RF': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)}

print("=== Leave-one-sensor-out AND hold out time (no shared hours) ===")
results = {k: [] for k in models}
for held in pool['sensor'].unique():
    train = pool[(pool['sensor'] != held) & (pool.index < midpoint)]
    test  = pool[(pool['sensor'] == held) & (pool.index >= midpoint)]
    if len(test) < 50:
        continue
    for mname, model in models.items():
        model.fit(train[FEATURES], train['bias'])
        pred = model.predict(test[FEATURES])
        results[mname].append({
            'sensor': held,
            'base': np.sqrt((test['bias']**2).mean()),
            'rmse': np.sqrt(mean_squared_error(test['bias'], pred)),
        })

for mname, rows in results.items():
    r = pd.DataFrame(rows)
    r['gain'] = r['base'] - r['rmse']
    print(f"\n{mname}:  mean RMSE {r['rmse'].mean():.3f} °C  "
          f"(baseline {r['base'].mean():.3f})  "
          f"gain {r['gain'].mean():+.3f}  "
          f"beats at {(r['gain']>0).sum()}/{len(r)}")

# control: shuffle sensor-level predictors across sensors.
# If skill survives this, the model isn't using land cover at all.
print("\n=== CONTROL: land cover randomly reassigned between sensors ===")
rng = np.random.default_rng(42)
smap = pool.groupby('sensor')[['built','water']].first()
shuffled = smap.sample(frac=1, random_state=42).set_index(smap.index)
pool_s = pool.copy()
pool_s['built'] = pool_s['sensor'].map(shuffled['built'])
pool_s['water'] = pool_s['sensor'].map(shuffled['water'])

for mname, model in models.items():
    rows = []
    for held in pool_s['sensor'].unique():
        train = pool_s[(pool_s['sensor'] != held) & (pool_s.index < midpoint)]
        test  = pool_s[(pool_s['sensor'] == held) & (pool_s.index >= midpoint)]
        if len(test) < 50:
            continue
        model.fit(train[FEATURES], train['bias'])
        pred = model.predict(test[FEATURES])
        rows.append({'base': np.sqrt((test['bias']**2).mean()),
                     'rmse': np.sqrt(mean_squared_error(test['bias'], pred))})
    r = pd.DataFrame(rows)
    r['gain'] = r['base'] - r['rmse']
    print(f"  {mname:6s} gain {r['gain'].mean():+.3f} °C  "
          f"beats at {(r['gain']>0).sum()}/{len(r)}")