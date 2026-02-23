from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from shapely.ops import unary_union
from sarvalanche.utils.constants import eps

def vectorize(raster, reference_da, min_area = None):
    """
    from sarvalanche.utils.vector_utils import vectorize
    vectorize(ds['cell_counts'] > 0, ds, min_area=0.000001).iloc[:].plot()
    from sarvalanche.utils.projections import resolution_to_degrees
    min_length = resolution_to_degrees(500, ds.rio.crs)
    vectorize(ds['cell_counts'] > 0, ds, min_area=min_length[0]**2)
    """
    shapes_list = list(shapes(raster.astype(int), transform=reference_da.rio.transform()))
    geoms = [shape(s[0]) for s in shapes_list if s[1] == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=reference_da.rio.crs)

    # Small buffer to connect touching pixels, then unbuffer
    merged = unary_union(gdf.geometry.buffer(eps))

    # Split multipolygon back into individual blobs
    if merged.geom_type == 'MultiPolygon':
        blobs = list(merged.geoms)
    else:
        blobs = [merged]

    gdf_blobs = gpd.GeoDataFrame(
        geometry=[g.buffer(-eps) for g in blobs],
        crs=reference_da.rio.crs
    )
    if min_area is not None:
        gdf_blobs = gdf_blobs[gdf_blobs.geometry.area > min_area]

    return gdf_blobs.reset_index(drop=True)