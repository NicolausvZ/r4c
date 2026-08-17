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

N_PERM = 1000
MIN_TRAIN_STD = 1e-3   # a predictor with no spread in training cannot be fitted
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------
# Rebuild the pooled table
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
    d['district'] = m['district']
    d['built'] = lc_row.iloc[0]['built_100m']
    d['water'] = lc_row.iloc[0]['water_5km']
    frames.append(d)

pool = pd.concat(frames)
pool['local_hour'] = (pool.index.hour + 3) % 24
night = pool[(pool['local_hour'] >= 22) | (pool['local_hour'] <= 4)].copy()

site = pd.DataFrame({
    'night_bias': night.groupby('sensor')['bias'].mean(),
    'built': night.groupby('sensor')['built'].first(),
    'water': night.groupby('sensor')['water'].first(),
    'district': night.groupby('sensor')['district'].first(),
})
print(f"Sensors: {len(site)}   nocturnal obs: {len(night)}")

# ---------------------------------------------------------------
# 1. LOO skill + permutation test
# ---------------------------------------------------------------
def loo_rmse(X, y):
    pred = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return np.sqrt(((y - pred) ** 2).mean()), pred

y = site['night_bias'].values
null_rmse = np.sqrt(((y - y.mean()) ** 2).mean())

print(f"\n=== Permutation test ({N_PERM} shuffles) ===")
print(f"predict-the-mean RMSE: {null_rmse:.3f} °C\n")

perm_results = {}
for feats in [['built'], ['water'], ['built', 'water']]:
    X = site[feats].values
    obs_rmse, pred = loo_rmse(X, y)

    null = np.empty(N_PERM)
    for i in range(N_PERM):
        Xs = X[RNG.permutation(len(X))]
        null[i], _ = loo_rmse(Xs, y)

    p = (null <= obs_rmse).mean()
    perm_results[tuple(feats)] = (obs_rmse, null, p, pred)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {str(feats):20s} RMSE {obs_rmse:.3f}  "
          f"null {null.mean():.3f} (sd {null.std():.3f})  p = {p:.4f}  {stars}")

# ---------------------------------------------------------------
# 2. Is `water` just a district label?
# ---------------------------------------------------------------
print("\n=== Within-district check (does built work inside one district?) ===")
for dist, g in site.groupby('district'):
    if len(g) < 6:
        continue
    r, p = stats.pearsonr(g['built'], g['night_bias'])
    rmse, _ = loo_rmse(g[['built']].values, g['night_bias'].values)
    nullr = np.sqrt(((g['night_bias'].values - g['night_bias'].mean()) ** 2).mean())
    print(f"  {dist:12s} n={len(g):2d}  built vs night_bias r={r:+.3f} (p={p:.4f})  "
          f"LOO RMSE {rmse:.3f} vs mean {nullr:.3f}")

print("\n=== Leave-one-district-out ===")
for dist in site['district'].unique():
    tr = site[site['district'] != dist]
    te = site[site['district'] == dist]
    for feats in [['built'], ['built', 'water']]:
        # A predictor with no spread in the training district cannot be fitted:
        # the coefficient is unconstrained and explodes on extrapolation.
        # (water is ~0.001-0.002 across all of Koivukyla, 0.44-0.67 in Laajasalo.)
        dropped = [f for f in feats if tr[f].std() <= MIN_TRAIN_STD]
        if dropped:
            print(f"  hold out {dist:12s} {str(feats):20s} SKIPPED - "
                  f"no variance in training data: {dropped}")
            continue
        mod = LinearRegression().fit(tr[feats].values, tr['night_bias'].values)
        pred = mod.predict(te[feats].values)
        rmse = np.sqrt(((te['night_bias'].values - pred) ** 2).mean())
        base = np.sqrt(((te['night_bias'].values - tr['night_bias'].mean()) ** 2).mean())
        verdict = "beats" if rmse < base else "WORSE than"
        print(f"  hold out {dist:12s} {str(feats):20s} RMSE {rmse:.3f}  "
              f"({verdict} train-mean baseline {base:.3f})")

# ---------------------------------------------------------------
# 3. Rank stability vs aggregation window
# ---------------------------------------------------------------
print("\n=== Hotspot rank stability vs averaging window ===")
print("  window      split-half rho    top5 overlap   mean |bias| spread")

for window_days in [1, 3, 7, 14, 30]:
    night['_blk'] = (night.index - night.index.min()).days // window_days
    odd = night[night['_blk'] % 2 == 1]
    even = night[night['_blk'] % 2 == 0]
    if odd.empty or even.empty:
        continue
    r1 = odd.groupby('sensor')['bias'].mean()
    r2 = even.groupby('sensor')['bias'].mean()
    common = r1.index.intersection(r2.index)
    rho = stats.spearmanr(r1[common], r2[common]).correlation
    ov = len(set(r1[common].nlargest(5).index) & set(r2[common].nlargest(5).index))
    spread = (r1[common] - r2[common]).abs().mean()
    print(f"  {window_days:3d} day    rho = {rho:+.3f}       {ov}/5           {spread:.3f} °C")

night['_blk'] = (night.index - night.index.min()).days // 7
r1 = night[night['_blk'] % 2 == 1].groupby('sensor')['bias'].mean()
r2 = night[night['_blk'] % 2 == 0].groupby('sensor')['bias'].mean()
common = r1.index.intersection(r2.index)
hot1 = set(r1[common].nlargest(10).index)
hot2 = set(r2[common].nlargest(10).index)
print(f"\n  hot-half vs cool-half agreement (7-day blocks): {len(hot1 & hot2)}/10 sensors")

# ---------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
obs_rmse, null, p, pred = perm_results[('built', 'water')]
ax.hist(null, bins=40, color='lightgrey', edgecolor='grey')
ax.axvline(obs_rmse, color='red', lw=2, label=f'observed {obs_rmse:.3f} (p={p:.3f})')
ax.axvline(null_rmse, color='k', ls='--', lw=1.5, label=f'predict-mean {null_rmse:.3f}')
ax.set_xlabel('LOO RMSE (°C)'); ax.set_ylabel('count')
ax.set_title('(a) Permutation null: built + water')
ax.legend(fontsize=8)

ax = axes[1]
colours = site['district'].map({'Koivukylä': 'tab:blue', 'Laajasalo': 'tab:red'})
ax.scatter(y, pred, c=colours, s=60)
lims = [min(y.min(), pred.min()) - 0.2, max(y.max(), pred.max()) + 0.2]
ax.plot(lims, lims, 'k--', lw=1)
for i, s in enumerate(site.index):
    ax.annotate(str(s)[:12], (y[i], pred[i]), fontsize=6,
                xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Observed nocturnal UHI (°C)')
ax.set_ylabel('Predicted, sensor held out (°C)')
ax.set_title(f'(b) LOO prediction  r = {np.corrcoef(y, pred)[0,1]:+.3f}')
ax.grid(alpha=0.3)

ax = axes[2]
windows, rhos = [], []
for wd in [1, 3, 7, 14, 30]:
    night['_blk'] = (night.index - night.index.min()).days // wd
    a = night[night['_blk'] % 2 == 1].groupby('sensor')['bias'].mean()
    b = night[night['_blk'] % 2 == 0].groupby('sensor')['bias'].mean()
    c = a.index.intersection(b.index)
    if len(c) < 5:
        continue
    windows.append(wd)
    rhos.append(stats.spearmanr(a[c], b[c]).correlation)
ax.plot(windows, rhos, 'o-', color='tab:green')
ax.axhline(0.8, color='k', ls=':', lw=1, label='rho = 0.8')
ax.set_xlabel('Averaging window (days)'); ax.set_ylabel('split-half rank rho')
ax.set_title('(c) Hotspot rank stability\nvs averaging window')
ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/uhi_validation.png', dpi=150)
plt.show()
print("\nSaved: data/uhi_validation.png")