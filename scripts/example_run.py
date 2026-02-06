from pathlib import Path
from shapely.geometry import box
import xarray as xr

aoi = box(-115.23172, 43.84563, -114.87403, 44.25274)
crs= 'EPSG:4326'

from sarvalanche.utils.projections import resolution_to_degrees
from sarvalanche.utils.validation import validate_crs
resolution = 20 # meters...

start_date = "2020-01-01"
stop_date = "2020-04-30"
avalanche_date = '2020-03-31'

from sarvalanche.detection.debris import detect_avalanche_debris

cache_dir = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data')
# ds = assemble_dataset(aoi, start_date, stop_date, crs, resolution, cache_dir)
ds = detect_avalanche_debris(aoi, crs, 20, start_date, stop_date, avalanche_date, cache_dir=cache_dir, overwrite=False)