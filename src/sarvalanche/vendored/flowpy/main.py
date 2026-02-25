#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:23:00 2018

    Copyright (C) <2020>  <Michael Neuhauser>
    Michael.Neuhauser@bfw.gv.at

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# import standard libraries
import os
import sys
import signal
import psutil
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
import multiprocessing as mp
import concurrent.futures
import logging
from xml.etree import ElementTree as ET
from tqdm.auto import tqdm
# Flow-Py Libraries
# import raster_io as io
from .Simulation import Simulation as Sim
# from . import flow_core as fc
from . import flow_core_fast as fc
from sarvalanche.vendored.flowpy.flow_math_numba import warmup_numba

log = logging.getLogger(__name__)

def _cleanup_executor(executor):
    """Force-kill all worker processes."""
    try:
        for pid, process in executor._processes.items():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker process state
# ---------------------------------------------------------------------------

# Each worker process populates this dict once via _worker_init.
# The DEM and header are sent to each worker exactly once (in the initializer),
# rather than being re-pickled into every task's IPC message.
_SHARED = {}


def _worker_init(dem, header):
    """Initializer called once per worker process.

    Stores the (large) DEM and header in process-local globals so that task
    args only need to carry the tiny per-zone release pixel indices.
    Also warms up Numba JIT inside each worker — required on macOS/Windows
    where the 'spawn' start method means workers don't inherit the parent's
    compiled functions.
    """
    _SHARED['dem'] = dem
    _SHARED['header'] = header
    from sarvalanche.vendored.flowpy.flow_math_numba import warmup_numba
    warmup_numba()

def _run_calculation(args):
    """Worker task.

    `args` is now (pixel_indices, alpha, exp, flux_threshold, max_z) where
    pixel_indices is a small (n_pixels, 2) int32 array of (row, col) pairs.
    The full DEM and header come from _SHARED, populated by _worker_init.

    This replaces the old pattern of pickling the full DEM array into every
    single task message — for 200 release zones on an 8 MB DEM that was
    200 × 2 × 8 MB = 3.2 GB sitting in the IPC queue simultaneously.
    """
    pixel_indices, alpha, exp, flux_threshold, max_z = args
    dem    = _SHARED['dem']
    header = _SHARED['header']
    release = np.zeros_like(dem, dtype=np.float64)
    release[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0
    return fc.calculation_effect((dem, header, release, alpha, exp, flux_threshold, max_z))

def read_header(ds):
    #Reads in the header of the raster file, input: filepath

    header = {}
    header['ncols'] = ds.x.size
    header['nrows'] = ds.y.size
    header['xllcorner'] = (ds.rio.transform() * (0, 0))[0]
    # header['yllcorner'] = (ds.rio.transform() * (0, ds.y.size))[1]
    # assumes north up convention
    header['yllcorner'] = (ds.rio.transform() * (0, ds.y.size - 1))[1]  # lower-left corner
    # assumes square pixels
    header['cellsize'] = ds.rio.transform()[0]
    header['noDataValue'] = ds.rio.nodata
    return header

def generate_path_vector(path_arr, release):
    from sarvalanche.utils.vector_utils import vectorize
    return vectorize(path_arr, release)

def run_flowpy(
    dem,
    release,
    *,
    alpha=20,
    exp=8,
    flux_threshold=3e-4,
    max_z=270,
    max_workers=12
):
    ref_da = dem.copy()

    log.info("Starting...")

    start = datetime.now().replace(microsecond=0)

    # Start of Calculation
    log.info('Start Calculation')
    log.info('Alpha Angle: {}'.format(alpha))
    log.info('Exponent: {}'.format(exp))
    log.info('Flux Threshold: {}'.format(flux_threshold))
    log.info('Max Z_delta: {}'.format(max_z))
    # Read in raster files
    header = read_header(dem)
    release_header = read_header(release)

    # Check if Layers have same size!!!
    if header['ncols'] == release_header['ncols'] and header['nrows'] == release_header['nrows']:
        log.info("DEM and Release Layer ok!")
    else:
        raise ValueError("Error: Release Layer doesn't match DEM!")

    if not isinstance(dem, np.ndarray): dem = dem.compute().values
    if not isinstance(release, np.ndarray): release = release.compute().values

    # Check actual available memory
    avaiable_memory = psutil.virtual_memory().available
    needed_memory = dem.nbytes

    # Per-worker memory is much more than DEM size alone:
    #   - DEM duplicated in each spawned worker process (~1×)
    #   - flowpy creates ~15 intermediate arrays of DEM size (~15×)
    #   - 7 result arrays returned per task (~7×)
    #   - Python process overhead + IPC buffers
    # 25× is conservative; floor of 0.5 GB for small DEMs so we don't
    # underestimate on tiny test grids.
    per_worker_estimate = max(needed_memory * 25, 0.5e9)

    log.info(f"Available memory: {avaiable_memory / 1e9:.2f} GB")
    log.info(f"DEM size: {needed_memory / 1e9:.3f} GB")
    log.info(f"Estimated memory per worker: {per_worker_estimate / 1e9:.2f} GB")

    # Sanity check — use 60 % of available as safe ceiling
    total_needed = per_worker_estimate * max_workers
    if total_needed > avaiable_memory * 0.6:
        log.info(f"May not have enough memory!")
        log.info(f"Needed: {total_needed / 1e9:.2f} GB, Available × 60%%: {avaiable_memory * 0.6 / 1e9:.2f} GB")
        # Reduce workers: target ≤ 50 % of available memory
        max_workers = max(1, int(avaiable_memory * 0.5 / per_worker_estimate))
        log.info(f"Reducing to {max_workers} workers")

    log.info('Files read in')

    z_delta = np.zeros_like(dem)
    flux = np.zeros_like(dem)
    cell_counts = np.zeros_like(dem)
    z_delta_sum = np.zeros_like(dem)
    backcalc = np.zeros_like(dem)
    fp_ta = np.zeros_like(dem)
    sl_ta = np.zeros_like(dem)

    log.info(
        "There are %.2f GBytes of Memory available and %.2f GBytes estimated per worker. Max. Nr. of Processes = %d",
        avaiable_memory / 1e9, per_worker_estimate / 1e9, max_workers)

    # Calculation
    log.info('Multiprocessing starts, available cores: %d', cpu_count())

    # split so we have 10 tasks per processes
    # release_list = fc.split_release(release, release_header, max_workers * 10)
    # use a shuffled version to reduce processing itme.
    # release_list = fc.split_release_by_points_shuffled(release, release_header, max_workers * 10)
    # use a by each labeled release zone split
    # Build one sparse (row, col) index array per connected release zone WITHOUT
    # ever materialising N full DEM-sized arrays simultaneously.
    #
    # The old approach was:
    #   release_list = split_release_by_label(...)   # N × DEM_size in RAM!
    #   sparse_list  = [np.argwhere(r > 0) for r in release_list]
    #   del release_list
    #
    # For 3 452 zones on a large DEM (e.g. 8 MB each) that peaks at ~27 GB
    # before a single worker starts.  Instead we label once (one int32 DEM-
    # sized array) and extract coordinates zone-by-zone:
    from scipy.ndimage import label as _nd_label

    # Apply the same nodata / clipping that split_release_by_label does
    nodata = release_header.get("noDataValue")
    if nodata:
        release[release == nodata] = 0
    else:
        release[release < 0] = 0
    release[release > 1] = 1

    labeled, n_zones = _nd_label(release > 0)   # single int32 DEM-sized array
    log.info("Found %d release zones", n_zones)

    sparse_list = []
    for zone_id in range(1, n_zones + 1):
        coords = np.argwhere(labeled == zone_id).astype(np.int32)
        if len(coords):
            sparse_list.append(coords)

    del labeled  # free the single int32 label array — done with it

    log.info("{} Processes started.".format(min(max_workers, len(sparse_list))))
    # --- prepare arguments (small per-task tuples only) ---
    args = [
        (pixels, alpha, exp, flux_threshold, max_z)
        for pixels in sparse_list
    ]
    # run biggest zones first.
    args.sort(key=lambda a: len(a[0]))
    # Log the distribution of zone sizes
    sizes = [len(a[0]) for a in args]
    log.info("Zone size stats: min=%d, max=%d, mean=%.1f, p95=%d, p99=%d",
            min(sizes), max(sizes),
            np.mean(sizes),
            int(np.percentile(sizes, 95)),
            int(np.percentile(sizes, 99)))

    del sparse_list

    warmup_numba()  # JIT compile in parent (inherited by fork workers on Linux)
    log.info("Numba warmup complete.")

    n_tasks = len(args)
    n_workers = min(max_workers, n_tasks)

    log.info("Starting flow calculation (%d tasks, %d workers)", n_tasks, n_workers)

    path_list = []

    executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(dem, header),
        )
    _active_executor = executor  # module-level reference

    def _signal_handler(signum, frame):
        _cleanup_executor(_active_executor)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        futures = {executor.submit(_run_calculation, arg): i for i, arg in enumerate(args)}
        del args

        for future in tqdm(concurrent.futures.as_completed(futures), total=n_tasks,
                        desc="FlowPy calculation", unit="task"):
            res = future.result()
            z_delta     = np.maximum(z_delta,     res[0])
            flux        = np.maximum(flux,         res[1])
            cell_counts += res[2]
            z_delta_sum += res[3]
            backcalc    = np.maximum(backcalc,     res[4])
            fp_ta       = np.maximum(fp_ta,        res[5])
            sl_ta       = np.maximum(sl_ta,        res[6])

            path = (res[2] > 0).astype(float)
            path_list.append(generate_path_vector(path, ref_da))
            futures.pop(future)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)  # restore default
        signal.signal(signal.SIGINT, signal.SIG_DFL)


    end = datetime.now().replace(microsecond=0)
    log.info('Calculation finished in %s', end - start)

    # return z_delta, flux, cell_counts, z_delta_sum, backcalc, fp_ta, sl_ta
    # only return cell_counts (number of start cells that converge to a pixel)
    # fp_ta the flow path travel angle. small = long shallow runout, big = steep direct hit
    # and path list which is a list of geopandas dataframes for each paths
    return cell_counts, fp_ta, path_list

if __name__ == "__main__":
    run_flowpy()