import numpy as np
import xarray as xr
from typing import Iterator, Sequence, Any
import logging

log = logging.getLogger(__name__)

def iter_track_pol_combinations(
    ds: xr.Dataset,
    polarizations: Sequence[str] = ("VV", "VH"),
    track_var: str = "track",
    lia_var: str = "lia",
    lia_dim: str = "static_track",
    include_lia: bool = True,
    skip_missing: bool = True,
) -> Iterator[tuple[Any, str, xr.DataArray, xr.DataArray | None]]:
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
    lia_var : str, default="lia"
        Name of the local incidence angle variable.
    lia_dim : str, default="static_track"
        Dimension name used to index the LIA by track.
    include_lia : bool, default=True
        Whether to include local incidence angle in output.
        If False, yields None for LIA.
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
    lia : xr.DataArray or None
        Local incidence angle for this track, dims=(y, x).
        None if include_lia=False or if lia_var not in dataset.

    Examples
    --------
    >>> for track, pol, da, lia in iter_track_pol_combinations(ds):
    ...     print(f"Processing {pol} on track {track}")
    ...     da_db = linear_to_dB(da)

    >>> # Skip LIA if not needed
    >>> for track, pol, da, _ in iter_track_pol_combinations(ds, include_lia=False):
    ...     process(da)

    >>> # Only process VV polarization
    >>> for track, pol, da, lia in iter_track_pol_combinations(ds, polarizations=["VV"]):
    ...     ...
    """
    if track_var not in ds.coords and track_var not in ds.data_vars:
        raise ValueError(f"Track variable '{track_var}' not found in dataset")

    tracks = np.unique(ds[track_var].values)
    log.debug(f"Iterating over {len(tracks)} tracks and {len(polarizations)} polarizations")

    for track in tracks:
        # Get LIA for this track if requested
        lia = None
        if include_lia:
            if lia_var in ds:
                try:
                    lia = ds[lia_var].sel({lia_dim: track})
                except (KeyError, ValueError) as e:
                    log.warning(f"Could not select LIA for track {track}: {e}")
            else:
                log.debug(f"LIA variable '{lia_var}' not found in dataset")

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

            yield track, pol, da, lia