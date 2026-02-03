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
import psutil
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
import multiprocessing as mp
import concurrent.futures
import logging
from xml.etree import ElementTree as ET
from tqdm import tqdm

# Flow-Py Libraries
# import raster_io as io
from .Simulation import Simulation as Sim
from . import flow_core as fc

log = logging.getLogger(__name__)

def _run_calculation(args):
    """Wrapper for multiprocessing worker."""
    return fc.calculation_effect(args)

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

def run_flowpy(
    dem,
    release,
    *,
    alpha=25,
    exp=8,
    flux_threshold=3e-4,
    max_z=270,
):

    log.info("Starting...")
    log.info("...")

    start = datetime.now().replace(microsecond=0)
    calc_bool = False
    # Create result directory
    time_string = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    infra = np.zeros_like(dem)
    if not isinstance(dem, np.ndarray): dem = dem.data
    if not isinstance(release, np.ndarray): release = release.data

    log.info('Files read in')

    z_delta = np.zeros_like(dem)
    flux = np.zeros_like(dem)
    cell_counts = np.zeros_like(dem)
    z_delta_sum = np.zeros_like(dem)
    backcalc = np.zeros_like(dem)
    fp_ta = np.zeros_like(dem)
    sl_ta = np.zeros_like(dem)

    avaiable_memory = psutil.virtual_memory()[1]
    needed_memory = sys.getsizeof(dem)

    max_number_procces = int(avaiable_memory / (needed_memory * 10))


    log.info(
        "There are {} Bytes of Memory avaiable and {} Bytes needed per process. \nMax. Nr. of Processes = {}".format(
            avaiable_memory, needed_memory*10, max_number_procces))

    # Calculation
    log.info('Multiprocessing starts, used cores: {}'.format(cpu_count()))

    release_list = fc.split_release(release, release_header, min(mp.cpu_count() * 5, max_number_procces))

    log.info("{} Processes started.".format(len(release_list)))

    # --- prepare arguments ---
    args = [
        (
            dem,
            header,
            release_pixel,
            alpha,
            exp,
            flux_threshold,
            max_z,
        )
        for release_pixel in release_list
    ]

    n_tasks = len(args)
    n_workers = min(mp.cpu_count(), max_number_procces, n_tasks)
    log.info("Starting flow calculation (%d tasks, %d workers)", n_tasks, n_workers)

    results = [None] * n_tasks  # placeholder

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # submit all tasks
        future_to_idx = {executor.submit(_run_calculation, arg): i for i, arg in enumerate(args)}

        # update progress bar as tasks complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=n_tasks,
                        desc="FlowPy calculation", unit="task"):
            idx = future_to_idx[future]
            results[idx] = future.result()  # store result in original order

    z_delta_list = []
    flux_list = []
    cc_list = []
    z_delta_sum_list = []
    backcalc_list = []
    fp_ta_list = []
    sl_ta_list = []
    for i in range(len(results)):
        res = results[i]
        res = list(res)
        z_delta_list.append(res[0])
        flux_list.append(res[1])
        cc_list.append(res[2])
        z_delta_sum_list.append(res[3])
        backcalc_list.append(res[4])
        fp_ta_list.append(res[5])
        sl_ta_list.append(res[6])

    log.info('Calculation finished, getting results.')
    for i in range(len(z_delta_list)):
        z_delta = np.maximum(z_delta, z_delta_list[i])
        flux = np.maximum(flux, flux_list[i])
        cell_counts += cc_list[i]
        z_delta_sum += z_delta_sum_list[i]
        backcalc = np.maximum(backcalc, backcalc_list[i])
        fp_ta = np.maximum(fp_ta, fp_ta_list[i])
        sl_ta = np.maximum(sl_ta, sl_ta_list[i])



    log.info("Calculation finished")
    log.info("...")
    end = datetime.now().replace(microsecond=0)
    log.info('Calculation needed: ' + str(end - start) + ' seconds')

    # return z_delta, flux, cell_counts, z_delta_sum, backcalc, fp_ta, sl_ta
    # only return cell_counts (number of start cells that converge to a pixel)
    # fp_ta the flow path travel angle. small = long shallow runout, big = steep direct hit
    # backcalc is
    return cell_counts, fp_ta, backcalc

if __name__ == "__main__":
    run_flowpy()