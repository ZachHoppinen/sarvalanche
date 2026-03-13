"""Compare baseline (2-ch) vs magnitude (3-ch) CNN models on 2024-2025 Sawtooth."""
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.transform import from_bounds
import rasterio.features

obs = pd.read_csv('local/issw/snfac_obs_2021_2025.csv')
obs['date'] = pd.to_datetime(obs['date'])
mask = (obs['date'] >= '2024-11-01') & (obs['date'] <= '2025-04-30')
saw_mask = obs['zone_name'] == 'Sawtooth & Western Smoky Mtns'
obs_2425 = obs[mask & saw_mask].copy().reset_index(drop=True)
print(f'Total 2024-2025 Sawtooth obs: {len(obs_2425)}')

paths = gpd.read_file('local/issw/observations/all_flowpy_paths.gpkg')
paths['date'] = pd.to_datetime(paths['date'])

onset_old = xr.open_dataset('local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/v2_season_inference_2024-2025/temporal_onset.nc')
onset_new = xr.open_dataset('local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/v2_season_inference_2024-2025_mag_v2/temporal_onset.nc')
prob_old = xr.open_dataset('local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/v2_season_inference_2024-2025/season_v2_debris_probabilities.nc')
prob_new = xr.open_dataset('local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns/v2_season_inference_2024-2025_mag_v2/season_v2_debris_probabilities.nc')
crs = prob_old.rio.crs or "EPSG:4326"

x = prob_old.x.values
y = prob_old.y.values
H, W = len(y), len(x)
dx = abs(float(x[1] - x[0]))
dy = abs(float(y[1] - y[0]))
transform = from_bounds(
    float(x.min()) - dx/2, float(y.min()) - dy/2,
    float(x.max()) + dx/2, float(y.max()) + dy/2, W, H
)

def check(row, prob_ds, onset_ds):
    obs_date = row['date']
    obs_id = row['id']

    date_paths = paths[(paths['date'].dt.date == obs_date.date()) & (paths['obs_id'] == obs_id)]
    if len(date_paths) == 0:
        return None

    path_gdf = date_paths.to_crs(crs)
    geom = path_gdf.geometry.values[0]
    if geom is None or geom.is_empty:
        return None

    try:
        path_mask = ~rasterio.features.geometry_mask(
            [geom], out_shape=(H, W), transform=transform, all_touched=True
        )
    except Exception:
        return None

    if path_mask.sum() == 0:
        return None

    cnn_times = pd.DatetimeIndex(prob_ds.time.values)
    time_diffs = np.abs(cnn_times - obs_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 6:
        return None

    prob_map = prob_ds['debris_probability'].isel(time=ci).values
    path_probs = prob_map[path_mask]
    bg_mask = ~path_mask & (prob_map > 0)
    bg_probs = prob_map[bg_mask] if bg_mask.any() else np.array([0.0])

    p95_path = float(np.percentile(path_probs, 95))
    p95_bg = float(np.percentile(bg_probs, 95))
    n_above = int((path_probs > 0.5).sum())

    spatial = (p95_path > p95_bg) or (n_above >= 10)

    temporal = False
    if spatial and 'onset_date' in onset_ds:
        onsets = onset_ds['onset_date'].values[path_mask]
        valid = onsets[~np.isnat(onsets)]
        if len(valid) > 0:
            med = pd.Timestamp(np.sort(valid)[len(valid)//2])
            temporal = abs((med - obs_date).days) <= 6

    return {'spatial': spatial, 'detected': spatial and temporal,
            'p95_path': p95_path, 'n_above': n_above}

res_old = []
res_new = []
d_sizes = []

for _, row in obs_2425.iterrows():
    r_old = check(row, prob_old, onset_old)
    r_new = check(row, prob_new, onset_new)
    if r_old is not None and r_new is not None:
        res_old.append(r_old)
        res_new.append(r_new)
        d_sizes.append(row.get('d_size', float('nan')))

n = len(res_old)
print(f'\nMatched observations: {n} / {len(obs_2425)}')

s_old = sum(r['spatial'] for r in res_old)
s_new = sum(r['spatial'] for r in res_new)
d_old = sum(r['detected'] for r in res_old)
d_new = sum(r['detected'] for r in res_new)

print(f'\n{"Metric":<20} {"BASELINE (2-ch)":>18} {"MAGNITUDE (3-ch)":>18}')
print(f'{"Spatial detected":<20} {s_old}/{n} ({s_old/n*100:5.1f}%)     {s_new}/{n} ({s_new/n*100:5.1f}%)')
print(f'{"Full detected":<20} {d_old}/{n} ({d_old/n*100:5.1f}%)     {d_new}/{n} ({d_new/n*100:5.1f}%)')

print(f'\n--- Per D-size ---')
print(f'{"D-size":<8} {"n":>4} {"OLD spat":>12} {"NEW spat":>12} {"OLD full":>12} {"NEW full":>12}')
for d in sorted(set(ds for ds in d_sizes if ds == ds)):
    idx = [i for i, ds in enumerate(d_sizes) if ds == d]
    nd = len(idx)
    os = sum(res_old[i]['spatial'] for i in idx)
    ns = sum(res_new[i]['spatial'] for i in idx)
    od = sum(res_old[i]['detected'] for i in idx)
    ndd = sum(res_new[i]['detected'] for i in idx)
    print(f'D{d:<7} {nd:>4} {os:>3}/{nd} ({os/nd*100:4.0f}%)  {ns:>3}/{nd} ({ns/nd*100:4.0f}%)  {od:>3}/{nd} ({od/nd*100:4.0f}%)  {ndd:>3}/{nd} ({ndd/nd*100:4.0f}%)')

gained_s = sum(1 for o, nn in zip(res_old, res_new) if not o['spatial'] and nn['spatial'])
lost_s = sum(1 for o, nn in zip(res_old, res_new) if o['spatial'] and not nn['spatial'])
gained_f = sum(1 for o, nn in zip(res_old, res_new) if not o['detected'] and nn['detected'])
lost_f = sum(1 for o, nn in zip(res_old, res_new) if o['detected'] and not nn['detected'])
print(f'\nChanges:')
print(f'  Spatial: +{gained_s} gained, -{lost_s} lost (net {gained_s-lost_s:+d})')
print(f'  Full:    +{gained_f} gained, -{lost_f} lost (net {gained_f-lost_f:+d})')

# Monthly breakdown
print(f'\n--- Per month ---')
months = [pd.to_datetime(row['date']).month for row in [obs_2425.iloc[i] for i in range(len(obs_2425)) if check(obs_2425.iloc[i], prob_old, onset_old) is not None and check(obs_2425.iloc[i], prob_new, onset_new) is not None]]
# Actually just get months from matched obs
obs_dates = []
for _, row in obs_2425.iterrows():
    r_old = check(row, prob_old, onset_old)
    r_new = check(row, prob_new, onset_new)
    if r_old is not None and r_new is not None:
        obs_dates.append(row['date'].month)

month_names = {11: 'Nov', 12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr'}
print(f'{"Month":<8} {"n":>4} {"OLD spat":>12} {"NEW spat":>12} {"OLD full":>12} {"NEW full":>12}')
for m in [11, 12, 1, 2, 3, 4]:
    idx = [i for i, md in enumerate(obs_dates) if md == m]
    if not idx:
        continue
    nd = len(idx)
    os = sum(res_old[i]['spatial'] for i in idx)
    ns = sum(res_new[i]['spatial'] for i in idx)
    od = sum(res_old[i]['detected'] for i in idx)
    ndd = sum(res_new[i]['detected'] for i in idx)
    print(f'{month_names[m]:<8} {nd:>4} {os:>3}/{nd} ({os/nd*100:4.0f}%)  {ns:>3}/{nd} ({ns/nd*100:4.0f}%)  {od:>3}/{nd} ({od/nd*100:4.0f}%)  {ndd:>3}/{nd} ({ndd/nd*100:4.0f}%)')
