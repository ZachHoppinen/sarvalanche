from pathlib import Path
from shapely.geometry import box
import xarray as xr

if __name__ == '__main__':
    aoi = box(-115.23172, 43.84563, -114.87403, 44.25274)

    from sarvalanche.utils.projections import resolution_to_degrees
    from sarvalanche.utils.validation import validate_crs
    resolution = 20 # meters
    crs= 'EPSG:4326'
    resolution = resolution_to_degrees(resolution, validate_crs(crs))

    start_date = "2020-03-01"
    stop_date = "2020-05-01"
    avalanche_date = '2020-03-31'

    from sarvalanche.detection_pipeline import run_detection

    cache_dir = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data')
    ds = run_detection(aoi, crs, resolution, start_date, stop_date, avalanche_date, cache_dir=cache_dir, overwrite=False)