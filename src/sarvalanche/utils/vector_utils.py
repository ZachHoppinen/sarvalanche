from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

def vectorize(raster, reference_da):
    """Vectorize a binary raster using rasterio.features.shapes."""
    shapes_list = list(shapes(raster.astype(int), transform=reference_da.rio.transform()))
    geoms = [shape(s[0]) for s in shapes_list if s[1] == 1]
    return gpd.GeoDataFrame(geometry=geoms, crs=reference_da.rio.crs)
