"""
Numba-JIT inner math functions for FlowPy.

Drop these into flow_core_fast.py and call them from CellFast methods.
First call will trigger compilation (~2-5s). All subsequent calls run at C speed.

Usage in CellFast:
    from sarvalanche.vendored.flowpy.flow_math_numba import (
        numba_calc_z_delta,
        numba_calc_tanbeta,
        numba_calc_persistence,
        numba_calc_distribution,
    )
"""

import logging
import numpy as np
import math
from numba import njit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants as plain numpy arrays (Numba can't access module-level xr/class state)
# ---------------------------------------------------------------------------

_DS = np.array([[math.sqrt(2), 1., math.sqrt(2)],
                [1.,           1., 1.           ],
                [math.sqrt(2), 1., math.sqrt(2)]], dtype=np.float64)

# Persistence LUT: shape (3, 3, 3, 3) — [dy+1, dx+1] -> 3x3 weight matrix
# Mirrors _PERSISTENCE_LUT in flow_core_fast.py exactly
def _build_persistence_lut():
    W = 0.707
    lut = np.zeros((3, 3, 3, 3), dtype=np.float64)
    # dx=-1, dy=-1
    lut[0, 0] = np.array([[0, 0, 0], [0, 0, W], [0, W, 1.0]])
    # dx=0,  dy=-1
    lut[0, 1] = np.array([[0, 0, 0], [0, 0, 0], [W, 1.0, W]])
    # dx=1,  dy=-1
    lut[0, 2] = np.array([[0, 0, 0], [W, 0, 0], [1.0, W, 0]])
    # dx=-1, dy=0
    lut[1, 0] = np.array([[0, 0, W], [0, 0, 1.0], [0, 0, W]])
    # dx=0,  dy=0  (centre — should never be used)
    lut[1, 1] = np.zeros((3, 3))
    # dx=1,  dy=0
    lut[1, 2] = np.array([[W, 0, 0], [1.0, 0, 0], [W, 0, 0]])
    # dx=-1, dy=1
    lut[2, 0] = np.array([[0, W, 1.0], [0, 0, W], [0, 0, 0]])
    # dx=0,  dy=1
    lut[2, 1] = np.array([[W, 1.0, W], [0, 0, 0], [0, 0, 0]])
    # dx=1,  dy=1
    lut[2, 2] = np.array([[1.0, W, 0], [W, 0, 0], [0, 0, 0]])
    return lut

_PERSISTENCE_LUT_NP = _build_persistence_lut()


# ---------------------------------------------------------------------------
# JIT functions
# All operate on plain float64 arrays — no Python objects, no numpy dispatch.
# ---------------------------------------------------------------------------

@njit(cache=True)
def numba_calc_z_delta(altitude, dem_ng, z_delta, ds_alpha, max_z_delta):
    """
    Replaces CellFast.calc_z_delta().

    Parameters
    ----------
    altitude : float
    dem_ng   : (3,3) float64 — neighbourhood DEM
    z_delta  : float — current cell z_delta
    ds_alpha : (3,3) float64 — cached _DS * cellsize * tan(alpha)
    max_z_delta : float

    Returns
    -------
    z_delta_neighbour : (3,3) float64
    """
    z_delta_neighbour = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            v = z_delta + (altitude - dem_ng[i, j]) - ds_alpha[i, j]
            if v < 0.0:
                v = 0.0
            if v > max_z_delta:
                v = max_z_delta
            z_delta_neighbour[i, j] = v
    return z_delta_neighbour


@njit(cache=True)
def numba_calc_tanbeta(altitude, dem_ng, cellsize, exp, z_delta_neighbour, persistence):
    """
    Replaces CellFast.calc_tanbeta().

    Uses math.atan / math.tan per element — avoids numpy ufunc dispatch
    overhead on tiny (3,3) arrays, which dominates at 4M calls.

    Returns
    -------
    tan_beta : (3,3) float64
    r_t      : (3,3) float64
    """
    SQRT2 = math.sqrt(2.0)
    PI    = math.pi
    half_pi = PI / 2.0

    # Distance multipliers — same as _DS
    ds = np.array([[SQRT2, 1.0, SQRT2],
                   [1.0,   1.0, 1.0  ],
                   [SQRT2, 1.0, SQRT2]])

    tan_beta = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            if z_delta_neighbour[i, j] <= 0.0:
                continue
            if persistence[i, j] <= 0.0:
                continue
            dist = ds[i, j] * cellsize
            beta = math.atan((altitude - dem_ng[i, j]) / dist) + half_pi
            tan_beta[i, j] = math.tan(beta / 2.0)

    # r_t = tan_beta^exp / sum(tan_beta^exp)
    r_t = np.zeros((3, 3), dtype=np.float64)
    total = 0.0
    for i in range(3):
        for j in range(3):
            v = tan_beta[i, j] ** exp
            r_t[i, j] = v
            total += v

    if total > 0.0:
        for i in range(3):
            for j in range(3):
                r_t[i, j] /= total

    return tan_beta, r_t


@njit(cache=True)
def numba_calc_persistence(is_start, parent_is_start,
                            n_parents,
                            parent_dx, parent_dy, parent_z_delta,
                            lut):
    """
    Replaces CellFast.calc_persistence().

    Parameters
    ----------
    is_start       : bool
    parent_is_start: bool  — True if first parent is a start cell
    n_parents      : int   — number of parents
    parent_dx      : (n,) int array — colindex delta per parent
    parent_dy      : (n,) int array — rowindex delta per parent
    parent_z_delta : (n,) float64  — z_delta per parent
    lut            : (3,3,3,3) float64 — persistence weight LUT

    Returns
    -------
    persistence : (3,3) float64
    no_flow     : (3,3) float64  — 1 everywhere except parent directions (0)
    """
    persistence = np.zeros((3, 3), dtype=np.float64)
    no_flow     = np.ones((3, 3),  dtype=np.float64)

    if is_start or parent_is_start:
        for i in range(3):
            for j in range(3):
                persistence[i, j] = 1.0
        return persistence, no_flow

    for p in range(n_parents):
        dx = parent_dx[p]
        dy = parent_dy[p]
        no_flow[dy + 1, dx + 1] = 0.0
        w = parent_z_delta[p]
        for i in range(3):
            for j in range(3):
                persistence[i, j] += w * lut[dy + 1, dx + 1, i, j]

    return persistence, no_flow


@njit(cache=True)
def numba_calc_distribution(altitude, dem_ng, z_delta, ds_alpha, max_z_delta,
                             cellsize, exp, flux_threshold, flux,
                             is_start, parent_is_start,
                             n_parents, parent_dx, parent_dy, parent_z_delta,
                             lut,
                             rowindex, colindex):
    """
    Single entry point replacing calc_z_delta + calc_persistence +
    calc_tanbeta + calc_distribution in one JIT-compiled function.

    Returns
    -------
    out_rows     : (k,) int64
    out_cols     : (k,) int64
    out_flux     : (k,) float64
    out_z_delta  : (k,) float64
    max_gamma    : float  (fp travel angle, degrees; 0 if is_start)
    sl_gamma     : float  (straight-line travel angle, degrees; 0 if is_start)
    new_z_delta_neighbour : (3,3) float64  — stored back onto cell
    """
    # --- z_delta ---
    z_delta_neighbour = numba_calc_z_delta(altitude, dem_ng, z_delta, ds_alpha, max_z_delta)

    # --- persistence ---
    persistence, no_flow = numba_calc_persistence(
        is_start, parent_is_start,
        n_parents, parent_dx, parent_dy, parent_z_delta,
        lut
    )
    # apply no_flow mask
    for i in range(3):
        for j in range(3):
            persistence[i, j] *= no_flow[i, j]

    # --- tan_beta / r_t ---
    tan_beta, r_t = numba_calc_tanbeta(
        altitude, dem_ng, cellsize, exp, z_delta_neighbour, persistence
    )

    # --- distribution ---
    dist = np.zeros((3, 3), dtype=np.float64)
    r_t_sum = 0.0
    for i in range(3):
        for j in range(3):
            r_t_sum += r_t[i, j]

    if r_t_sum > 0.0:
        pers_rt_sum = 0.0
        for i in range(3):
            for j in range(3):
                pers_rt_sum += persistence[i, j] * r_t[i, j]
        if pers_rt_sum > 0.0:
            for i in range(3):
                for j in range(3):
                    dist[i, j] = (persistence[i, j] * r_t[i, j]) / pers_rt_sum * flux

    # Redistribute sub-threshold flux
    threshold = flux_threshold
    mass_below = 0.0
    count_above = 0
    for i in range(3):
        for j in range(3):
            v = dist[i, j]
            if 0.0 < v < threshold:
                mass_below += v
                dist[i, j] = 0.0
            elif v >= threshold:
                count_above += 1

    if mass_below > 0.0 and count_above > 0:
        add = mass_below / count_above
        for i in range(3):
            for j in range(3):
                if dist[i, j] >= threshold:
                    dist[i, j] += add

    # residual
    if count_above > 0:
        dist_sum = 0.0
        for i in range(3):
            for j in range(3):
                dist_sum += dist[i, j]
        residual = flux - dist_sum
        if residual > 0.0:
            add = residual / count_above
            for i in range(3):
                for j in range(3):
                    if dist[i, j] >= threshold:
                        dist[i, j] += add

    # Collect output cells
    n_out = 0
    for i in range(3):
        for j in range(3):
            if dist[i, j] > threshold:
                n_out += 1

    out_rows   = np.empty(n_out, dtype=np.int64)
    out_cols   = np.empty(n_out, dtype=np.int64)
    out_flux   = np.empty(n_out, dtype=np.float64)
    out_zdelta = np.empty(n_out, dtype=np.float64)

    k = 0
    for i in range(3):
        for j in range(3):
            if dist[i, j] > threshold:
                out_rows[k]   = rowindex - 1 + i
                out_cols[k]   = colindex - 1 + j
                out_flux[k]   = dist[i, j]
                out_zdelta[k] = z_delta_neighbour[i, j]
                k += 1

    return out_rows, out_cols, out_flux, out_zdelta, z_delta_neighbour


def warmup_numba(cellsize=10.0, alpha=20.0, exp=8, flux_threshold=3e-4, max_z=270.0):
    """
    Call once at startup to trigger Numba JIT compilation before the
    real calculation begins. Takes 2-5s on first run, then cached.
    """
    dem_ng   = np.ones((3, 3), dtype=np.float64) * 100.0
    dem_ng[1,1] = 110.0
    ds_alpha = _DS * cellsize * math.tan(math.radians(alpha))
    lut      = _PERSISTENCE_LUT_NP

    # warmup each function
    numba_calc_z_delta(110.0, dem_ng, 5.0, ds_alpha, max_z)
    numba_calc_tanbeta(110.0, dem_ng, cellsize, exp,
                       np.ones((3,3)), np.ones((3,3)))
    numba_calc_persistence(False, False, 1,
                           np.array([0], dtype=np.int64),
                           np.array([-1], dtype=np.int64),
                           np.array([5.0]),
                           lut)
    numba_calc_distribution(
        110.0, dem_ng, 5.0, ds_alpha, max_z,
        cellsize, exp, flux_threshold, 1.0,
        False, False,
        1,
        np.array([0], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        np.array([5.0]),
        lut,
        1, 1
    )
    log.debug("Numba warmup complete.")