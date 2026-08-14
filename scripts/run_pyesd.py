# --- compatibility shims for pyESD's older numpy/sklearn/pandas expectations ---
# Must precede pyESD imports.
import numpy as np
for _alias, _real in [('int', int), ('float', float), ('bool', bool)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

import pandas as pd
import xarray as xr

import pyESD.metrics as _pyesd_metrics
from sklearn.metrics import mean_squared_error as _sk_mse, r2_score as _sk_r2

# sklearn >=1.6 removed mean_squared_error(squared=...)
def _mse_compat(y_true, y_pred, squared=True, **kwargs):
    value = _sk_mse(y_true, y_pred, **kwargs)
    return value if squared else np.sqrt(value)

_pyesd_metrics.mean_squared_error = _mse_compat

# pyESD's adjusted_r2 indexes y_true[1] instead of the predictor count.
N_PREDICTORS = 2

def _adjusted_r2(self):
    n = len(self.y_true)
    p = N_PREDICTORS
    r2 = _sk_r2(self.y_true, self.y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

_pyesd_metrics.Evaluate.adjusted_r2 = _adjusted_r2

from pyESD.StationOperator import StationOperator
from pyESD.ESD_utils import Dataset
from pyESD.standardizer import NoStandardizer

variable = "Temperature"
cachedir = 'data/pyesd_cache'

# cross_validate_and_predict requires all three of these keys
SCORING = ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"]

ERA5Data = Dataset('ERA5', {
    't2m': 'data/pyesd_era5/t2m.nc',
    'd2m': 'data/pyesd_era5/d2m.nc',
}, 'NH')

df_hourly = pd.read_pickle('data/fvh_hourly.pkl')
locations = pd.read_csv('data/sensor_locations.csv')

col = [c for c in df_hourly.columns if 'Koivutaival' in str(c)][0]
sensor_name = str(col).split('_')[-1]

match = None
for _, row in locations.iterrows():
    if sensor_name.lower() in str(row['name']).lower():
        match = row
        break
lat, lon = (match['lat'], match['lon']) if match is not None else (60.32, 25.06)

series = df_hourly[col].dropna()
data = pd.DataFrame({variable: series.values}, index=series.index)

era5_time = pd.DatetimeIndex(xr.open_dataarray('data/pyesd_era5/t2m.nc').time.values)
daterange = data.index.intersection(era5_time)

print(f"=== {sensor_name}  lat={lat:.4f} lon={lon:.4f} ===")
print(f"daterange: {daterange[0]} -> {daterange[-1]}  (n={len(daterange)})")

SO = StationOperator(data=data, name=sensor_name, lat=lat, lon=lon, elevation=0)

SO.set_predictors(variable, ["t2m", "d2m"], cachedir, radius=100,
                  standardizer=NoStandardizer())
SO.set_standardizer(variable, standardizer=NoStandardizer())

SO.set_model(variable, method="RandomForest", cv=5,
             daterange=daterange, predictor_dataset=ERA5Data,
             scoring=SCORING)

SO.fit(variable, daterange, ERA5Data,
       fit_predictors=True, predictor_selector=False)

print("\n=== In-sample evaluation ===")
print(SO.evaluate(variable, daterange, ERA5Data))

print("\n=== Cross-validated (honest) ===")
cv_score, y_pred = SO.cross_validate_and_predict(variable, daterange, ERA5Data)
print(cv_score)

y_pred.to_csv(f'data/pyesd_pred_{sensor_name.replace(" ", "_")}.csv')
print(f"\nPredictions saved: data/pyesd_pred_{sensor_name.replace(' ', '_')}.csv")