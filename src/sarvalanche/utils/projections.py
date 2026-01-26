from pyproj import Transformer

def get_aoi_utm_bounds(aoi, da_crs, aoi_crs = 'EPSG:4326'):
    minx, miny, maxx, maxy = aoi.bounds
    # Transformer
    T = Transformer.from_crs(aoi_crs, da_crs, always_xy=True)

    # Transform all corners at once
    x0, y0 = T.transform(minx, miny)
    x1, y1 = T.transform(maxx, maxy)

    utm_bounds = [x0, y0, x1, y1]
    return utm_bounds
