from pathlib import Path
from shapely.geometry import box
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
# aoi = box(-115.0, 43.58, -114.5, 44.13)
# aoi = box(-115.02068191820455, 44.14529196659243,  -114.99119907695828, 44.16379687146455)
# aoi = box(*total_bounds)
if __name__ == '__main__':
    aoi = box(-110.772, 43.734, -110.745, 43.756)
    crs= 'EPSG:4326'

    # convert 30 meters to degrees
    from sarvalanche.utils.projections import resolution_to_degrees
    from sarvalanche.utils.validation import validate_crs, validate_resolution
    resolution = resolution_to_degrees(20, validate_crs(crs))

    start_date = "2019-12-01"
    stop_date = "2020-02-01"
    avalanche_date = '2020-01-11'
    # from sarvalanche.detection.debris import detect_avalanche_debris
    from sarvalanche.detection_pipeline import run_detection
    from sarvalanche.io.dataset import assemble_dataset
    cache_dir = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data')
    # ds = assemble_dataset(aoi, start_date, stop_date, crs, resolution, cache_dir)
    resolution = resolution_to_degrees(20, validate_crs(crs))
    # ds = run_detection(aoi, crs = crs, resolution = resolution, start_date=start_date, stop_date=stop_date,cache_dir=cache_dir, avalanche_date='2020-01-11', overwrite=False, static_fp='/Users/zmhoppinen/Documents/sarvalanche/local/data/2020-01-11.nc', debug=True)
    ds = run_detection(aoi, avalanche_date='2020-01-11', overwrite=True, cache_dir=cache_dir) #static_fp='/Users/zmhoppinen/Documents/sarvalanche/local/data/2020-01-11.nc'
    ds['detections'].plot()
    plt.show()