import gc
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import geopandas as gpd
import xarray as xr

from sarvalanche.io.dataset import load_netcdf_to_dataset

log = logging.getLogger(__name__)

def iter_track_pol_combinations(
    ds: xr.Dataset,
    polarizations: Sequence[str] = ("VV", "VH"),
    track_var: str = "track",
    skip_missing: bool = True,
) -> Iterator[tuple[str, str, xr.DataArray]]:
    """
    Generator yielding (track, polarization, data, local_incidence_angle) tuples.

    Iterates over all combinations of tracks and polarizations in a SAR dataset,
    selecting the appropriate data and local incidence angle for each combination.

    Parameters
    ----------
    ds : xr.Dataset
        SAR dataset containing polarization variables and track information.
    polarizations : Sequence[str], default=("VV", "VH")
        Polarization channels to iterate over.
    track_var : str, default="track"
        Name of the track variable/coordinate in the dataset.
    skip_missing : bool, default=True
        If True, skip polarizations not present in dataset.
        If False, raise KeyError for missing polarizations.

    Yields
    ------
    track : Any
        Track identifier (orbit/pass).
    pol : str
        Polarization string (e.g., "VV", "VH").
    da : xr.DataArray
        Data for this track/polarization combination, dims=(time, y, x).

    Examples
    --------
    >>> for track, pol, da in iter_track_pol_combinations(ds):
    ...     print(f"Processing {pol} on track {track}")
    ...     da_db = linear_to_dB(da)


    >>> # Only process VV polarization
    >>> for track, pol, da in iter_track_pol_combinations(ds, polarizations=["VV"]):
    ...     ...
    """
    if track_var not in ds.coords and track_var not in ds.data_vars:
        raise ValueError(f"Track variable '{track_var}' not found in dataset")

    tracks = np.unique(ds[track_var].values)
    log.debug(f"Iterating over {len(tracks)} tracks and {len(polarizations)} polarizations")

    for track in tracks:
        # Get LIA for this track if requested

        for pol in polarizations:
            # Check if polarization exists
            if pol not in ds:
                if skip_missing:
                    log.debug(f"Skipping missing polarization: {pol}")
                    continue
                else:
                    raise KeyError(f"Polarization '{pol}' not found in dataset")

            # Select data for this track
            da = ds[pol].sel(time=ds[track_var] == track)

            if da.sizes.get('time', 0) == 0:
                log.warning(f"No data for track {track}, polarization {pol}")
                if not skip_missing:
                    raise ValueError(f"Empty data for track {track}, pol {pol}")
                continue

            yield track, pol, da

def iter_run_tracks(
    gpkg_paths: list[Path],
    nc_paths: list[Path],
    var_whitelist: list[str] | None = None,
) -> Iterator[tuple[int, gpd.GeoSeries, xr.Dataset]]:
    """Yield (idx, row, ds) for every track across all paired run files.

    Loads each (gpkg, nc) pair once, yields all tracks within it, then closes
    before moving to the next file. Suitable for unlabeled use cases such as
    building segmentation training sets across all available data.

    Parameters
    ----------
    gpkg_paths : list[Path]
        Ordered list of .gpkg paths, paired with nc_paths.
    nc_paths : list[Path]
        Ordered list of .nc paths, paired with gpkg_paths.
    var_whitelist : list[str] or None
        Passed to load_ds; limits which variables are loaded.

    Yields
    ------
    idx : int
        Track index within the GeoDataFrame.
    row : gpd.GeoSeries
        Track row.
    ds : xr.Dataset
        Loaded and reprojected dataset for this file pair. Valid only for the
        current iteration — do not store references across yields.
    crs: crs of track raw
    """
    for gpkg_path, nc_path in zip(gpkg_paths, nc_paths):
        gdf = gpd.read_file(gpkg_path)
        ds = load_netcdf_to_dataset(nc_path)
        log.info('iter_run_tracks: %d tracks — %s', len(gdf), nc_path.name)

        try:
            for idx in gdf.index:
                yield idx, gdf.loc[idx], ds, gdf.crs
        finally:
            ds.close()
            del ds
            gc.collect()


def iter_labeled_run_tracks(
    gpkg_paths: list[Path],
    nc_paths: list[Path],
    labels: dict,
    var_whitelist: list[str] | None = None,
) -> Iterator[tuple[str, dict, gpd.GeoSeries, xr.Dataset]]:
    """Yield (key, meta, row, ds) for every labeled track across all paired run files.

    Groups labels by (gpkg, nc) pair so each file is loaded only once.
    Tracks whose index is not found in the GeoDataFrame are skipped with a warning.

    Parameters
    ----------
    gpkg_paths : list[Path]
        Ordered list of .gpkg paths, paired with nc_paths.
    nc_paths : list[Path]
        Ordered list of .nc paths, paired with gpkg_paths.
    labels : dict
        Contents of ``track_labels.json``. Keys are ``zone|date|track_idx``;
        values have ``zone``, ``date``, ``track_idx``, ``label`` fields.
    var_whitelist : list[str] or None
        Passed to load_ds; limits which variables are loaded.

    Yields
    ------
    key : str
        Label key (``zone|date|track_idx``).
    meta : dict
        Full label metadata entry.
    row : gpd.GeoSeries
        Track row.
    ds : xr.Dataset
        Loaded and reprojected dataset. Valid only for the current iteration.
    """
    # Group label keys by their (gpkg, nc) pair using the same path pairing
    from collections import defaultdict

    # Build a lookup: (zone, date) stem → list of (key, meta)
    by_stem: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, meta in labels.items():
        stem = f"{meta['zone']}_{meta['date']}"
        by_stem[stem].append((key, meta))

    for gpkg_path, nc_path in zip(gpkg_paths, nc_paths):
        stem = nc_path.stem
        entries = by_stem.get(stem)
        if not entries:
            log.debug('iter_labeled_run_tracks: no labels for %s, skipping', stem)
            continue

        gdf = gpd.read_file(gpkg_path)
        ds = load_netcdf_to_dataset(nc_path)
        log.info('iter_labeled_run_tracks: %d labeled tracks — %s', len(entries), nc_path.name)

        try:
            for key, meta in entries:
                idx = meta['track_idx']
                if idx not in gdf.index:
                    log.warning('iter_labeled_run_tracks: track %d not in %s, skipping', idx, stem)
                    continue
                yield key, meta, gdf.loc[idx], ds, gdf.crs
        finally:
            ds.close()
            del ds
            gc.collect()