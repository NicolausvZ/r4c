import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata

RADII_M = [100, 250, 500, 1000]
CLASSES = {10: 'tree', 20: 'shrub', 30: 'grass', 40: 'crop',
           50: 'built', 60: 'bare', 80: 'water', 90: 'wetland'}

meta = load_sensor_metadata()

src = rasterio.open('data/worldcover_helsinki.tif')
print(f"WorldCover: {src.shape}, res {src.res[0]:.6f}°, CRS {src.crs}")

rows = []
for _, s in meta.iterrows():
    lat, lon = s['lat'], s['lon']
    rec = {'name': s['name'], 'sensor_id': s['sensor_id'],
           'district': s['district'], 'lat': lat, 'lon': lon}

    for r in RADII_M:
        dlat = r / 111_320.0
        dlon = r / (111_320.0 * np.cos(np.radians(lat)))
        win = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform)
        patch = src.read(1, window=win)

        if patch.size == 0:
            for c in CLASSES.values():
                rec[f'{c}_{r}m'] = np.nan
            continue

        for code, cname in CLASSES.items():
            rec[f'{cname}_{r}m'] = (patch == code).sum() / patch.size

    rows.append(rec)

src.close()

lc = pd.DataFrame(rows)
lc.to_csv('data/sensor_landcover.csv', index=False)

cols500 = [f'{c}_500m' for c in ['built', 'tree', 'water', 'grass']]
print("\n=== Land cover fractions within 500 m ===")
print(lc[['name', 'district'] + cols500]
      .sort_values('built_500m', ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

# ---- relate to measured bias ----
# model_results_by_sensor.csv keys the sensor column as 'sensor', not 'name'
try:
    res = pd.read_csv('data/model_results_by_sensor.csv')
    merged = res.merge(lc, left_on='sensor', right_on='name', how='inner')
    print(f"\nMatched {len(merged)}/{len(lc)} sensors to model results")
    missing = set(lc['name']) - set(merged['sensor'])
    if missing:
        print(f"  unmatched: {sorted(missing)}")

    if len(merged) > 5:
        print("\n=== Correlation of mean bias with land cover ===")
        best = None
        for r in RADII_M:
            for c in ['built', 'tree', 'water']:
                col = f'{c}_{r}m'
                if col in merged and merged[col].notna().sum() > 5:
                    rho = merged['mean_bias'].corr(merged[col])
                    print(f"  mean_bias vs {col:14s} r = {rho:+.3f}")
                    if best is None or abs(rho) > abs(best[1]):
                        best = (col, rho)
        print(f"\nStrongest predictor: {best[0]}  (r = {best[1]:+.3f})")
        print("\nNote: built and tree fractions are collinear "
              f"(r = {merged['built_100m'].corr(merged['tree_100m']):+.3f}); "
              "use one, not both.")
except FileNotFoundError:
    print("\n(run run_pyesd_bias.py first to see bias correlations)")

print("\nSaved: data/sensor_landcover.csv")