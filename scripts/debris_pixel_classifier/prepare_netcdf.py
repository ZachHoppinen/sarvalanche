"""Prepare season NetCDFs for pairwise debris detector training.

Adds precomputed static channels (aspect_northing, aspect_easting, tpi)
to existing season NetCDFs so the training dataset can load them lazily
without recomputing per run.

Skips if the variables already exist (idempotent).

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/prepare_netcdf.py \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_*.nc"

    # All SNFAC:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/prepare_netcdf.py \
        --nc local/issw/snfac/netcdfs/*/season_*.nc

    # From config:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/prepare_netcdf.py \
        --config scripts/debris_pixel_classifier/train_config_combined.yaml
"""

import argparse
import logging
import time as _time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.pairwise_debris_classifier.channels import (
    STATIC_CHANNELS,
    normalize_static_channel,
)
from sarvalanche.ml.pairwise_debris_classifier.static_stack import _estimate_utm_crs
from sarvalanche.preprocessing.pipelines import preprocess_rtc
from sarvalanche.utils.validation import check_rad_degrees

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Variables that must be derived and added to the netcdf.
# This is the subset of STATIC_CHANNELS that aren't in the raw netcdf.
# Raw netcdf has: slope, cell_counts. Missing: aspect_northing, aspect_easting, tpi.
_RAW_NC_VARS = {'slope', 'cell_counts'}
DERIVED_VARS = [ch for ch in STATIC_CHANNELS if ch not in _RAW_NC_VARS]


def prepare_one(nc_path: Path, idx: int = 0, total: int = 1):
    """Add derived static variables to a season NetCDF."""
    t_start = _time.time()
    log.info("[%d/%d] Processing %s", idx + 1, total, nc_path)

    # Quick header check — no data loaded
    with xr.open_dataset(nc_path) as ds_header:
        existing = [v for v in DERIVED_VARS if v in ds_header.data_vars]
    if len(existing) == len(DERIVED_VARS):
        log.info("  Already prepared (has %s), skipping", existing)
        return

    # Check if TV denoising is needed — determines what we load
    with xr.open_dataset(nc_path) as ds_check:
        already_preprocessed = ds_check.attrs.get('preprocessed') == 'rtc_tv'
        # Also check if data is already in dB (e.g. from assemble_dataset)
        if not already_preprocessed and 'VV' in ds_check:
            from sarvalanche.utils.validation import check_db_linear as _check_db
            vv_scale = _check_db(ds_check['VV'])
            already_in_db = vv_scale == 'dB'
        else:
            already_in_db = False

    t0 = _time.time()
    if already_preprocessed:
        # Already preprocessed — open lazily, only load static vars we need
        log.info("  Already preprocessed, opening lazily...")
        ds = load_netcdf_to_dataset(nc_path)
        for v in ['slope', 'aspect', 'dem', 'cell_counts']:
            if v in ds:
                ds[v].load()
        log.info("  Loaded static layers in %.0fs (%dx%d)",
                 _time.time() - t0, ds.sizes['y'], ds.sizes['x'])
    elif already_in_db:
        # Data is already in dB but not TV-denoised.
        # Mark as preprocessed — skip TV denoise for OPERA RTC dB data.
        log.info("  Data already in dB scale, marking as preprocessed (skipping TV denoise)")
        ds = load_netcdf_to_dataset(nc_path)
        ds.attrs['preprocessed'] = 'rtc_tv'
    else:
        # Need full dataset for TV denoising (linear scale data)
        log.info("  Loading full dataset (needs preprocess_rtc)...")
        ds = load_netcdf_to_dataset(nc_path)
        if any(var.chunks is not None for var in ds.variables.values()):
            ds = ds.load()
        log.info("  Loaded in %.0fs (%d times, %dx%d)",
                 _time.time() - t0, ds.sizes.get('time', 0), ds.sizes['y'], ds.sizes['x'])
        log.info("  Running preprocess_rtc (this may take a few minutes)...")
        t0 = _time.time()
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs['preprocessed'] = 'rtc_tv'
        log.info("  Preprocessed in %.0fs", _time.time() - t0)

    H, W = ds.sizes['y'], ds.sizes['x']
    dims_yx = ('y', 'x')

    # Aspect decomposition
    if 'aspect' in ds.data_vars and 'aspect_northing' not in ds.data_vars:
        aspect_da = ds['aspect']
        unit = check_rad_degrees(aspect_da)
        aspect_arr = np.nan_to_num(aspect_da.values.astype(np.float32), nan=0.0)
        if unit == 'degrees':
            log.info("  Aspect is in degrees, converting to radians")
            aspect_arr = np.deg2rad(aspect_arr)

        ds['aspect_northing'] = xr.DataArray(
            np.cos(aspect_arr).astype(np.float32), dims=dims_yx,
            attrs={'units': 'unitless', 'source': 'derived', 'product': 'cos(aspect)'})
        ds['aspect_easting'] = xr.DataArray(
            np.sin(aspect_arr).astype(np.float32), dims=dims_yx,
            attrs={'units': 'unitless', 'source': 'derived', 'product': 'sin(aspect)'})
        log.info("  Added aspect_northing, aspect_easting")

    # TPI from DEM
    if 'dem' in ds.data_vars and 'tpi' not in ds.data_vars:
        from sarvalanche.utils.terrain import compute_tpi

        dem_da = ds['dem']
        try:
            if ds.rio.crs and ds.rio.crs.is_geographic:
                utm_crs = _estimate_utm_crs(ds)
                log.info("  Reprojecting DEM to %s for TPI", utm_crs)
                dem_proj = dem_da.rio.reproject(utm_crs)
                tpi_da = compute_tpi(dem_proj, radius_m=300.0)
                tpi_da = tpi_da.rio.reproject_match(dem_da)
            else:
                tpi_da = compute_tpi(dem_da, radius_m=300.0)

            tpi_arr = np.nan_to_num(tpi_da.values.astype(np.float32), nan=0.0)
            if tpi_arr.shape != (H, W):
                raise ValueError(f"TPI shape {tpi_arr.shape} != ({H}, {W})")

            ds['tpi'] = xr.DataArray(
                tpi_arr, dims=dims_yx,
                attrs={'units': 'meters', 'source': 'derived', 'product': 'TPI_300m'})
            log.info("  Added tpi")
        except Exception:
            log.warning("  Could not compute TPI", exc_info=True)

    # Delete stale npy files before saving new netcdf
    for suffix in ['_VV.npy', '_VH.npy', '_static.npy', '_anf.npy.npz']:
        stale = nc_path.parent / (nc_path.stem + suffix)
        if stale.exists():
            stale.unlink()
            log.info("  Removed stale %s", stale.name)

    # Save atomically: write to temp file, then rename
    from sarvalanche.io.export import export_netcdf
    import shutil
    tmp_path = nc_path.parent / (nc_path.stem + '_tmp.nc')
    log.info("  Saving to temp file...")
    t0 = _time.time()
    export_netcdf(ds, tmp_path, overwrite=True)
    ds.close()
    shutil.move(str(tmp_path), str(nc_path))
    log.info("  Saved in %.0fs. Total: %.0fs", _time.time() - t0, _time.time() - t_start)


def export_npy(nc_path: Path, idx: int = 0, total: int = 1):
    """Export VV and VH as contiguous float32 .npy files for fast memmap access.

    Saves VV.npy and VH.npy in the same directory as the netcdf.
    Skips if files already exist and are newer than the netcdf.
    """
    vv_path = nc_path.parent / (nc_path.stem + '_VV.npy')
    vh_path = nc_path.parent / (nc_path.stem + '_VH.npy')
    static_path = nc_path.parent / (nc_path.stem + '_static.npy')
    anf_path = nc_path.parent / (nc_path.stem + '_anf.npy')

    # Skip if all npy files exist and are newer than the netcdf
    nc_mtime = nc_path.stat().st_mtime
    all_exist = all(p.exists() and p.stat().st_mtime >= nc_mtime
                    for p in [vv_path, vh_path, static_path])
    if all_exist:
        log.info("[%d/%d] NPY files up to date for %s, skipping", idx + 1, total, nc_path.stem)
        return

    log.info("[%d/%d] Exporting NPY for %s", idx + 1, total, nc_path.stem)
    t0 = _time.time()
    ds = xr.open_dataset(nc_path)

    # Check for required layers
    required_sar = ['VV', 'VH']
    required_static = ['slope', 'aspect_northing', 'aspect_easting', 'cell_counts', 'tpi']
    required_other = ['anf', 'track']

    missing_sar = [v for v in required_sar if v not in ds]
    missing_static = [v for v in required_static if v not in ds]
    missing_other = [v for v in required_other if v not in ds]

    if missing_sar:
        log.error("  MISSING SAR layers %s — cannot export NPY", missing_sar)
        ds.close()
        return
    if missing_static:
        log.warning("  Missing static layers %s — will be zeros in training", missing_static)
    if missing_other:
        log.warning("  Missing layers %s", missing_other)

    if ds.attrs.get('preprocessed') != 'rtc_tv':
        log.warning("  Not preprocessed (preprocessed=%s) — run prepare_netcdf first",
                     ds.attrs.get('preprocessed', 'none'))

    # VV: (T, H, W) float32
    vv = ds['VV'].values.astype(np.float32)
    np.save(vv_path, vv)
    log.info("  VV: %s → %.1f GB", vv.shape, vv.nbytes / 1e9)

    # VH: (T, H, W) float32
    if 'VH' in ds:
        vh = ds['VH'].values.astype(np.float32)
        np.save(vh_path, vh)
        log.info("  VH: %s → %.1f GB", vh.shape, vh.nbytes / 1e9)

    # Static channels: (N_STATIC, H, W) float32, pre-normalized
    from sarvalanche.ml.pairwise_debris_classifier.channels import (
        STATIC_CHANNELS, normalize_static_channel,
    )
    H, W = ds.sizes['y'], ds.sizes['x']
    _NC_VAR = {'slope': 'slope', 'aspect_northing': 'aspect_northing',
               'aspect_easting': 'aspect_easting', 'cell_counts': 'cell_counts', 'tpi': 'tpi'}
    static = np.zeros((len(STATIC_CHANNELS), H, W), dtype=np.float32)
    for ch, var in enumerate(STATIC_CHANNELS):
        nc_var = _NC_VAR.get(var)
        if nc_var and nc_var in ds:
            arr = np.nan_to_num(ds[nc_var].values.astype(np.float32), nan=0.0)
            static[ch] = normalize_static_channel(arr, var)
    np.save(static_path, static)
    log.info("  Static: %s → %.1f MB", static.shape, static.nbytes / 1e6)

    # ANF: (N_tracks, H, W) float32 with track IDs
    if 'anf' in ds:
        anf_data = ds['anf'].values.astype(np.float32)  # (static_track, H, W)
        anf_tracks = ds['anf'].static_track.values
        np.savez(anf_path, data=anf_data, tracks=anf_tracks)
        log.info("  ANF: %s, %d tracks", anf_data.shape, len(anf_tracks))

    ds.close()
    log.info("  Done in %.0fs", _time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, nargs="*", default=[],
                        help="NetCDF files to prepare")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config — extracts nc paths from all zones")
    args = parser.parse_args()

    nc_paths = list(args.nc)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for zone in config['zones']:
            p = Path(zone['nc'])
            if p not in nc_paths:
                nc_paths.append(p)

    if not nc_paths:
        parser.error("No netcdf files specified. Use --nc or --config.")

    t_total = _time.time()
    valid_paths = [p for p in nc_paths if p.exists()]
    missing = [p for p in nc_paths if not p.exists()]
    for p in missing:
        log.warning("Not found: %s", p)

    log.info("Processing %d NetCDF files", len(valid_paths))
    for i, nc_path in enumerate(tqdm(valid_paths, desc="Preparing NetCDFs")):
        prepare_one(nc_path, idx=i, total=len(valid_paths))

    log.info("Exporting NPY files for fast training access (2 workers)...")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    max_npy_workers = 2  # ~9 GB peak for two Galena zones
    with ProcessPoolExecutor(max_workers=max_npy_workers) as pool:
        futures = {pool.submit(export_npy, nc_path, i, len(valid_paths)): nc_path
                   for i, nc_path in enumerate(valid_paths)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Exporting NPY"):
            nc_path = futures[future]
            try:
                future.result()
            except Exception:
                log.error("NPY export failed for %s", nc_path, exc_info=True)

    log.info("All done. %d files in %.0fs.", len(valid_paths), _time.time() - t_total)


if __name__ == "__main__":
    main()
