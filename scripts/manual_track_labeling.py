import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import json
from pathlib import Path
import numpy as np

ds = xr.open_dataset('/Users/zmhoppinen/Documents/sarvalanche/local/data/2020-03-31.nc').astype(float).rio.write_crs('EPSG:4326')
gdf = gpd.read_file('/Users/zmhoppinen/Documents/sarvalanche/local/data/2020-03-31.gpkg')#.to_crs('EPSG:4326')
ds = ds.rio.reproject(gdf.crs)
# pip install geopandas fiona
observed = None
observed = gpd.read_file("/Users/zmhoppinen/Documents/sarvalanche/local/observed/200331Crowns_BV200511.kmz", driver="KML").to_crs(gdf.crs)
output_path = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data/2020-03-31_labels.json')

print(ds.x.values[:5])
print(ds.y.values[:5])


layers = {
    'distance_mahalanobis': {'cmap': 'plasma',   'label': 'Mahalanobis Distance', 'vmin': 0.4, 'vmax': 1},
    'p_empirical':          {'cmap': 'RdYlGn_r', 'label': 'Empirical p-value', 'vmin': 0.4, 'vmax': 1},
    'slope':                {'cmap': 'bone',  'label': 'Slope (rad)', 'vmin': np.deg2rad(15), 'vmax': np.deg2rad(45)},
    'cell_counts':          {'cmap': 'Blues',    'label': 'Cell Counts', 'vmin': 0, 'vmax': 1000},
}

if output_path.exists():
    with open(output_path) as f:
        labels = json.load(f)
else:
    labels = {}

unlabeled = gdf[~gdf.index.astype(str).isin(labels.keys())]
print(f"{len(labels)} already labeled, {len(unlabeled)} remaining")

result = {'label': None}

def on_key(event):
    if event.key in ['0', '1']:
        result['label'] = int(event.key)
        plt.close()
    elif event.key == 'n':
        result['label'] = -1
        plt.close()
    elif event.key == 'q':
        result['label'] = 'quit'
        plt.close()

from sarvalanche.utils.projections import resolution_to_meters
# buffer = resolution_to_meters(100, gdf.crs)[0]  # add a buffer around the path for better context
# buffer = 0.0005  # roughly 200m in degrees at this latitude


for idx, path in unlabeled.iterrows():
    bounds = path.geometry.bounds
    # print(f"Path {idx} bounds: {bounds}")
    # print(f"Path {idx} width: {bounds[2]-bounds[0]:.6f} height: {bounds[3]-bounds[1]:.6f}")
    # path_width = bounds[2] - bounds[0]
    # path_height = bounds[3] - bounds[1]
    # buffer = max(path_width, path_height) * 0.1
    buffer = 500
    path_gdf = gpd.GeoDataFrame([path], geometry='geometry', crs=gdf.crs)

    clip = ds.sel(
        x=slice(bounds[0] - buffer, bounds[2] + buffer),
        y=slice(bounds[3] + buffer, bounds[1] - buffer)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Path {idx} | 1=avalanche  0=no avalanche  s=skip  q=quit", fontsize=12)
    fig.canvas.mpl_connect('key_press_event', on_key)

    for ax, (var, opts) in zip(axes.flat, layers.items()):
        ds[var].sel(
            x=slice(bounds[0] - buffer, bounds[2] + buffer),
            y=slice(bounds[3] + buffer, bounds[1] - buffer)
        ).plot(ax=ax, cmap=opts['cmap'], robust=True, add_colorbar=True, vmin=opts['vmin'], vmax=opts['vmax'])
        path_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)
        if observed is not None:
            observed.plot(ax = ax, color = 'red', linewidth = 1)
        ax.set_title(opts['label'])
        ax.set_aspect('equal')

    plt.tight_layout()
    result['label'] = None
    plt.show()

    if result['label'] == 'quit':
        break
    if result['label'] is not None:
        labels[str(idx)] = result['label']
        with open(output_path, 'w') as f:
            json.dump(labels, f)
        print(f"Path {idx} → {result['label']} ({len(labels)} total)")

print(f"Done. {len(labels)} paths labeled, saved to {output_path}")