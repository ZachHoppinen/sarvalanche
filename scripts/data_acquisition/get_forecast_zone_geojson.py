import requests
import json
from shapely.geometry import shape, box
from pathlib import Path

def fetch_avalanche_zones(center_id: str = None, save_path: str = None):
    """
    Fetch forecast zone polygons from the avalanche.org public API.

    Parameters
    ----------
    center_id : str, optional
        Filter to a specific center (e.g. 'SNFAC', 'GNFAC', 'CAIC', 'UAC').
        If None, fetches all US centers.
    save_path : str, optional
        If provided, saves raw GeoJSON to disk for inspection/caching.

    Returns
    -------
    dict
        {zone_name: shapely_polygon} — use .bounds for bbox or pass directly to assemble_dataset
    """
    if center_id:
        url = f"https://api.avalanche.org/v2/public/products/map-layer/{center_id}"
    else:
        url = "https://api.avalanche.org/v2/public/products/map-layer"

    headers = {'User-Agent': 'sarvalanche-research/1.0 (your@email.com)'}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    geojson = response.json()

    if save_path:
        Path(save_path).write_text(json.dumps(geojson, indent=2))
        print(f"Saved GeoJSON to {save_path}")

    zones = {}
    for feature in geojson['features']:
        props = feature['properties']
        geom  = feature['geometry']

        zone_name   = props['name']
        center_name = props.get('center', 'unknown')
        center_code = props.get('center_id', 'unknown')

        try:
            polygon = shape(geom)
            zones[zone_name] = {
                'geometry': polygon,
                'bbox': polygon.bounds,          # (minx, miny, maxx, maxy)
                'shapely_box': box(*polygon.bounds),
                'center': center_name,
                'center_id': center_code,
            }
        except Exception as e:
            print(f"  Skipping {zone_name}: {e}")

    print(f"Fetched {len(zones)} zones" + (f" from {center_id}" if center_id else " across all centers"))
    return zones


def zones_to_aoi_dict(zones: dict, center_ids: list = None):
    """
    Convert fetched zones to the ZONES dict format expected by your training loop.

    Parameters
    ----------
    center_ids : list, optional
        Filter to specific centers e.g. ['SNFAC', 'GNFAC', 'CAIC']
    """
    aoi_dict = {}
    for zone_name, info in zones.items():
        if center_ids and info['center_id'] not in center_ids:
            continue
        # Use zone_name + center as key to avoid collisions
        key = f"{info['center_id']}_{zone_name.replace(' ', '_').replace('/', '-')}"
        aoi_dict[key] = info['shapely_box']
    return aoi_dict


# --- USAGE ---
if __name__ == '__main__':

    # Fetch all zones and cache locally
    all_zones = fetch_avalanche_zones(save_path='forecast_zones.geojson')

    # Print a summary by center
    from collections import defaultdict
    by_center = defaultdict(list)
    for name, info in all_zones.items():
        by_center[info['center_id']].append(name)

    for center_id, zone_names in sorted(by_center.items()):
        print(f"\n{center_id} ({len(zone_names)} zones):")
        for z in zone_names:
            bbox = all_zones[z]['bbox']
            print(f"  {z:40s}  bbox: {[round(c,3) for c in bbox]}")

    # Filter to specific centers for training
    TARGET_CENTERS = ['SNFAC', 'GNFAC', 'CAIC', 'UAC', 'ESAC']

    ZONES = zones_to_aoi_dict(all_zones, center_ids=TARGET_CENTERS)
    print(f"\nSelected {len(ZONES)} zones for training:")
    for k, v in ZONES.items():
        print(f"  {k}")