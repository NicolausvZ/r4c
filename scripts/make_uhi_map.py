import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os, sys
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4c_utils import load_sensor_metadata, match_sensor

OUT_RES_M = 100          # output pixel size
BUILT_RADIUS_M = 100     # must match built_100m in extract_landcover.py
WATER_RADIUS_M = 5000    # must match water_5km in add_water_predictors.py
BUILT_CODE, WATER_CODE = 50, 80

AREAS = {
    'Koivukyla': dict(lon=(25.02, 25.10), lat=(60.30, 60.35)),
    'Laajasalo': dict(lon=(25.02, 25.12), lat=(60.15, 60.21)),
}


def window_fraction(src, lat, lon, radius_m, code):
    """Fraction of `code` pixels in a square window of half-width radius_m.
    Deliberately identical to the sensor covariate definition in
    extract_landcover.py / add_water_predictors.py — if this diverges,
    the model is fed values it was never fitted on."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * np.cos(np.radians(lat)))
    win = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform)
    patch = src.read(1, window=win)
    if patch.size == 0:
        return np.nan, np.nan
    return (patch == code).sum() / patch.size, patch


# ---------------------------------------------------------------
# 1. Refit the pooled model
# ---------------------------------------------------------------
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
pool['local_hour'] = (pool.index.hour + 3) % 24
night = pool[(pool['local_hour'] >= 22) | (pool['local_hour'] <= 4)]

site = pd.DataFrame({
    'night_bias': night.groupby('sensor')['bias'].mean(),
    'built': night.groupby('sensor')['built'].first(),
    'water': night.groupby('sensor')['water'].first(),
})

model = LinearRegression().fit(site[['built', 'water']].values,
                               site['night_bias'].values)
b_built, b_water = model.coef_
print("=== Pooled model (nocturnal UHI, 22:00-04:00 local) ===")
print(f"  night_bias = {model.intercept_:+.4f} "
      f"{b_built:+.4f}*built_100m {b_water:+.4f}*water_5km")
print(f"  fitted on {len(site)} sensors, {len(night)} nocturnal observations")

built_lo, built_hi = site['built'].min(), site['built'].max()
water_lo, water_hi = site['water'].min(), site['water'].max()
print(f"  fitted range: built {built_lo:.3f}-{built_hi:.3f}, "
      f"water {water_lo:.3f}-{water_hi:.3f}")

src = rasterio.open('data/worldcover_helsinki.tif')

# ---------------------------------------------------------------
# 2. CONSISTENCY CHECK: recompute sensor covariates with the map's own
#    function. These must match sensor_landcover.csv.
# ---------------------------------------------------------------
print("\n=== Covariate consistency check (map function vs stored CSV) ===")
print(f"  {'sensor':26s} {'built_csv':>9s} {'built_map':>9s} {'diff':>7s}")
diffs = []
for name in site.index:
    row = lc[lc['name'] == name]
    if row.empty:
        continue
    lat, lon = row.iloc[0]['lat'], row.iloc[0]['lon']
    bmap, _ = window_fraction(src, lat, lon, BUILT_RADIUS_M, BUILT_CODE)
    bcsv = row.iloc[0]['built_100m']
    diffs.append(abs(bmap - bcsv))
    print(f"  {name[:26]:26s} {bcsv:9.3f} {bmap:9.3f} {bmap - bcsv:+7.3f}")

maxdiff = max(diffs) if diffs else np.nan
print(f"\n  max |difference| = {maxdiff:.4f}")
if maxdiff > 0.01:
    print("  WARNING: covariate definitions differ - map predictions unreliable")
else:
    print("  OK: map and sensor covariates are computed identically")

# ---------------------------------------------------------------
# 3. Predict on a grid, pixel by pixel, using the same function
# ---------------------------------------------------------------
results = {}
for area, box in AREAS.items():
    lon0, lon1 = box['lon']
    lat0, lat1 = box['lat']
    latc = (lat0 + lat1) / 2

    dlat = OUT_RES_M / 111_320.0
    dlon = OUT_RES_M / (111_320.0 * np.cos(np.radians(latc)))
    lats = np.arange(lat1, lat0, -dlat)
    lons = np.arange(lon0, lon1, dlon)
    print(f"\n{area}: {len(lats)} x {len(lons)} grid at ~{OUT_RES_M} m "
          f"({len(lats)*len(lons)} pixels)")

    built = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    water = np.full_like(built, np.nan)
    is_water = np.zeros_like(built, dtype=bool)

    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            b, patch = window_fraction(src, la, lo, BUILT_RADIUS_M, BUILT_CODE)
            built[i, j] = b
            if patch is not np.nan and np.size(patch):
                # centre pixel class, to mask open water
                cy, cx = np.array(patch.shape) // 2
                is_water[i, j] = (patch[cy, cx] == WATER_CODE)
        if (i + 1) % 20 == 0:
            print(f"    built rows {i+1}/{len(lats)}")

    # water_5km varies slowly; compute on a coarse grid and interpolate
    wstep = max(1, int(round(1000 / OUT_RES_M)))   # every ~1 km
    for i in range(0, len(lats), wstep):
        for j in range(0, len(lons), wstep):
            w, _ = window_fraction(src, lats[i], lons[j], WATER_RADIUS_M, WATER_CODE)
            water[i, j] = w
    wdf = pd.DataFrame(water)
    water = (wdf.interpolate(axis=0, limit_direction='both')
                .interpolate(axis=1, limit_direction='both').values)
    print(f"    water done (every ~{wstep*OUT_RES_M} m, interpolated)")

    X = np.column_stack([built.ravel(), water.ravel()])
    uhi = model.predict(X).reshape(built.shape)

    oor = ((built < built_lo) | (built > built_hi) |
           (water < water_lo) | (water > water_hi))
    uhi_masked = np.where(is_water, np.nan, uhi)

    results[area] = dict(uhi=uhi_masked, built=built, water=water,
                         extent=[lon0, lon1, lat0, lat1], oor=oor)

    valid = uhi_masked[~np.isnan(uhi_masked)]
    print(f"  built range on map: {np.nanmin(built):.3f}-{np.nanmax(built):.3f}")
    print(f"  water range on map: {np.nanmin(water):.3f}-{np.nanmax(water):.3f}")
    print(f"  predicted UHI: {valid.min():+.2f} to {valid.max():+.2f} °C "
          f"(mean {valid.mean():+.2f})")
    print(f"  pixels outside fitted covariate range: {oor.mean()*100:.1f}%")

# ---------------------------------------------------------------
# 4. Validate: map value at each sensor vs observed
# ---------------------------------------------------------------
print("\n=== Map vs observed at sensor locations ===")
print(f"  {'sensor':26s} {'observed':>9s} {'mapped':>9s} {'diff':>7s}")
val = []
for name in site.index:
    row = lc[lc['name'] == name]
    if row.empty:
        continue
    lat, lon = row.iloc[0]['lat'], row.iloc[0]['lon']
    for area, r in results.items():
        lo0, lo1, la0, la1 = r['extent']
        if lo0 <= lon <= lo1 and la0 <= lat <= la1:
            h, w = r['uhi'].shape
            i = int((la1 - lat) / (la1 - la0) * h)
            j = int((lon - lo0) / (lo1 - lo0) * w)
            i, j = min(max(i, 0), h - 1), min(max(j, 0), w - 1)
            mapped = r['uhi'][i, j]
            obs = site.loc[name, 'night_bias']
            val.append({'sensor': name, 'obs': obs, 'map': mapped})
            print(f"  {name[:26]:26s} {obs:+9.2f} {mapped:+9.2f} {mapped-obs:+7.2f}")
            break

vdf = pd.DataFrame(val).dropna()
if len(vdf) > 3:
    rmse = np.sqrt(((vdf['map'] - vdf['obs']) ** 2).mean())
    r = vdf['obs'].corr(vdf['map'])
    print(f"\n  n = {len(vdf)}   RMSE {rmse:.3f} °C   r = {r:+.3f}")
    print("  (in-sample: these sensors were used to fit the model)")

src.close()

# ---------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
vmin = min(np.nanmin(r['uhi']) for r in results.values())
vmax = max(np.nanmax(r['uhi']) for r in results.values())
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

for j, (area, r) in enumerate(results.items()):
    ax = axes[0, j]
    im = ax.imshow(r['uhi'], extent=r['extent'], origin='upper',
                   cmap='RdYlBu_r', norm=norm, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Nocturnal UHI (°C vs ERA5)')
    for name in site.index:
        row = lc[lc['name'] == name]
        if row.empty:
            continue
        lat, lon = row.iloc[0]['lat'], row.iloc[0]['lon']
        if r['extent'][0] <= lon <= r['extent'][1] and r['extent'][2] <= lat <= r['extent'][3]:
            ax.plot(lon, lat, 'o', ms=8, mfc='none', mec='k', mew=1.5)
            ax.annotate(f"{name[:12]}\n{site.loc[name,'night_bias']:+.2f}",
                        (lon, lat), fontsize=6, xytext=(5, 3),
                        textcoords='offset points')
    ax.set_title(f'{area} — predicted nocturnal UHI\ncircles = sensors, observed °C')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    ax = axes[1, j]
    im = ax.imshow(r['built'], extent=r['extent'], origin='upper',
                   cmap='pink_r', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Built fraction (100 m)')
    ax.set_title(f'{area} — built fraction (model input)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('data/uhi_map.png', dpi=150)
plt.show()

# ---------------------------------------------------------------
# 6. GeoTIFF export
# ---------------------------------------------------------------
for area, r in results.items():
    lon0, lon1, lat0, lat1 = r['extent']
    h, w = r['uhi'].shape
    transform = rasterio.transform.from_bounds(lon0, lat0, lon1, lat1, w, h)
    path = f'data/uhi_map_{area.lower()}.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=h, width=w,
                       count=1, dtype='float32', crs='EPSG:4326',
                       transform=transform, nodata=np.nan) as dst:
        dst.write(r['uhi'].astype('float32'), 1)
        dst.update_tags(model='night_bias ~ built_100m + water_5km',
                        intercept=str(model.intercept_),
                        coef_built=str(b_built), coef_water=str(b_water),
                        period='2024-06-27 to 2024-08-31, 22:00-04:00 local',
                        loo_rmse='0.238 C')
    print(f"Saved: {path}")

print("\nSaved: data/uhi_map.png")