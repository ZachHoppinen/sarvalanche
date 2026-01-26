from pyproj import Transformer

def get_aoi_utm_bounds(aoi, da_crs, aoi_crs="EPSG:4326"):
    minx, miny, maxx, maxy = aoi.bounds

    T = Transformer.from_crs(aoi_crs, da_crs, always_xy=True)

    xs, ys = T.transform(
        [minx, minx, maxx, maxx],
        [miny, maxy, miny, maxy],
    )

    return [
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ]