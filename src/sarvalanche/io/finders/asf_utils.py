import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon

def get_opera_urls_from_asf_search(asf_results_df):
    urls = []

    for _, row in asf_results_df.iterrows():
        main = row.get("properties.url")
        if main:
            urls.append(main)

        extras = row.get("properties.additionalUrls") or []
        urls.extend(extras)

    return urls

def subset_asf_search_results(
    results_df,
    aoi=None,
    path_numbers=None,
    direction=None,
    polarization=None,
    start_time=None,
    stop_time=None,
    scene_name=None
):
    """
    Optional subset ASF search results with filters and AOI intersection.

    Args:
        results_df (pd.DataFrame): ASF search results.
        aoi (list/tuple/shapely.geometry): AOI as [xmin, ymin, xmax, ymax] or shapely geometry.
        path_numbers (list[int], optional): Filter by multiple path numbers.
        direction (str, optional): Filter by flightDirection.
        polarization (str, optional): Filter by polarization.
        start_time (str/pd.Timestamp, optional): Filter results after this time.
        stop_time (str/pd.Timestamp, optional): Filter results before this time.
        scene_name (str, optional): Filter by sceneName.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df = results_df.copy()

    # -- Convert AOI to shapely box if provided as list --
    # --- AOI intersection  ---
    if aoi is not None:
        def intersects_aoi(coords):
            geom = Polygon(coords[0])
            return geom.intersects(aoi)

        if aoi is not None and _looks_projected(aoi):
            pass
            # ("AOI must be in EPSG:4326 (lon/lat)")

        mask = df["geometry.coordinates"].apply(intersects_aoi)
        df = df[mask]

    # -- Path number filtering (support multiple) --
    if path_numbers is not None:
        df = df[df['properties.pathNumber'].isin(path_numbers)]

    # -- Other optional filters --
    if direction is not None:
        df = df[df['properties.flightDirection'] == direction]
    if polarization is not None:
        df = df[df['properties.polarization'] == polarization]
    if start_time is not None:
        start_time = pd.to_datetime(start_time)
        df = df[pd.to_datetime(df['properties.startTime']) >= start_time]
    if stop_time is not None:
        stop_time = pd.to_datetime(stop_time)
        df = df[pd.to_datetime(df['properties.stopTime']) <= stop_time]
    if scene_name is not None:
        df = df[df['properties.sceneName'] == scene_name]

    return df

def _looks_projected(geom):
    xmin, ymin, xmax, ymax = geom.bounds
    return abs(xmin) > 180 or abs(ymin) > 90
