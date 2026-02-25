import numpy as np
import math
from datetime import datetime
import logging

from sarvalanche.vendored.flowpy.flow_core import (
    split_release_by_label,
    split_release_by_points_shuffled,
    split_release,
    get_start_idx,
)
from sarvalanche.vendored.flowpy.flow_math_numba import (
    numba_calc_distribution,
    warmup_numba,
    _PERSISTENCE_LUT_NP,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DS = np.array([[np.sqrt(2), 1., np.sqrt(2)],
                [1.,         1., 1.        ],
                [np.sqrt(2), 1., np.sqrt(2)]])

_DS_TANBETA = np.array([[np.sqrt(2), 1., np.sqrt(2)],
                         [1.,         1., 1.        ],
                         [np.sqrt(2), 1., np.sqrt(2)]])

def _build_persistence_lut():
    W = 0.707
    lut = np.zeros((3, 3, 3, 3), dtype=np.float64)
    lut[0, 0] = np.array([[0, 0, 0], [0, 0, W], [0, W, 1]])
    lut[0, 1] = np.array([[0, 0, 0], [0, 0, 0], [W, 1, W]])
    lut[0, 2] = np.array([[0, 0, 0], [W, 0, 0], [1, W, 0]])
    lut[1, 0] = np.array([[0, 0, W], [0, 0, 1], [0, 0, W]])
    lut[1, 1] = np.zeros((3, 3))
    lut[1, 2] = np.array([[W, 0, 0], [1, 0, 0], [W, 0, 0]])
    lut[2, 0] = np.array([[0, W, 1], [0, 0, W], [0, 0, 0]])
    lut[2, 1] = np.array([[W, 1, W], [0, 0, 0], [0, 0, 0]])
    lut[2, 2] = np.array([[1, W, 0], [W, 0, 0], [0, 0, 0]])
    return lut

_PERSISTENCE_LUT = _build_persistence_lut()


# ---------------------------------------------------------------------------
# CellFast
# ---------------------------------------------------------------------------

class CellFast:

    __slots__ = (
        'rowindex', 'colindex', 'altitude', 'dem_ng', 'cellsize',
        'tan_beta', 'dist', 'persistence', 'r_t', 'no_flow',
        'flux', 'z_delta', 'alpha', 'exp', 'max_z_delta', 'flux_threshold',
        'min_distance', 'max_distance', 'min_gamma', 'max_gamma', 'sl_gamma',
        'is_start', 'startcell', 'parent',
        '_ds_alpha',
        'z_delta_neighbour',
    )

    def __init__(self, rowindex, colindex, dem_ng, cellsize, flux, z_delta,
                 parent, alpha, exp, flux_threshold, max_z_delta, startcell):
        self.rowindex = rowindex
        self.colindex = colindex
        self.altitude = dem_ng[1, 1]
        self.dem_ng = dem_ng
        self.cellsize = cellsize
        self.tan_beta = np.zeros((3, 3))
        self.dist = np.zeros((3, 3))
        self.persistence = np.zeros((3, 3))
        self.r_t = np.zeros((3, 3))
        self.no_flow = np.ones((3, 3))
        self.flux = flux
        self.z_delta = z_delta
        self.alpha = float(alpha)
        self.exp = int(exp)
        self.max_z_delta = float(max_z_delta)
        self.flux_threshold = float(flux_threshold)
        self.min_distance = 0
        self.max_distance = 0
        self.min_gamma = 0
        self.max_gamma = 0
        self.sl_gamma = 0
        self.z_delta_neighbour = np.zeros((3, 3))
        self._ds_alpha = _DS * cellsize * math.tan(math.radians(alpha))

        if type(startcell) == bool:
            self.is_start = True
        else:
            self.startcell = startcell
            self.is_start = False

        self.parent = []
        if type(parent) == CellFast:
            self.parent.append(parent)

    def add_os(self, flux):
        self.flux += flux

    def add_parent(self, parent):
        self.parent.append(parent)

    def calc_fp_travelangle(self):
        dh = self.startcell.altitude - self.altitude
        dist_min = [
            math.sqrt((p.colindex - self.colindex) ** 2 +
                      (p.rowindex - self.rowindex) ** 2) * self.cellsize + p.min_distance
            for p in self.parent
        ]
        self.min_distance = min(dist_min)
        self.max_gamma = math.degrees(math.atan(dh / self.min_distance))

    def calc_sl_travelangle(self):
        dx = abs(self.startcell.colindex - self.colindex)
        dy = abs(self.startcell.rowindex - self.rowindex)
        dh = self.startcell.altitude - self.altitude
        ds = math.sqrt(dx ** 2 + dy ** 2) * self.cellsize
        self.sl_gamma = math.degrees(math.atan(dh / ds))

    # kept as fallback — not called in hot path
    def calc_z_delta(self):
        z_gamma = self.altitude - self.dem_ng
        self.z_delta_neighbour = self.z_delta + z_gamma - self._ds_alpha
        self.z_delta_neighbour[self.z_delta_neighbour < 0] = 0
        self.z_delta_neighbour[self.z_delta_neighbour > self.max_z_delta] = self.max_z_delta

    # kept as fallback — not called in hot path
    def calc_tanbeta(self):
        distance = _DS_TANBETA * self.cellsize
        beta = np.arctan((self.altitude - self.dem_ng) / distance) + (np.pi / 2)
        self.tan_beta = np.tan(beta / 2)
        self.tan_beta[self.z_delta_neighbour <= 0] = 0
        self.tan_beta[self.persistence <= 0] = 0
        self.tan_beta[1, 1] = 0
        total = np.sum(self.tan_beta ** self.exp)
        if abs(total) > 0:
            self.r_t = self.tan_beta ** self.exp / total

    # kept as fallback — not called in hot path
    def calc_persistence(self):
        self.persistence = np.zeros((3, 3))
        if self.is_start or self.parent[0].is_start:
            self.persistence += 1.0
            return
        for parent in self.parent:
            dx = parent.colindex - self.colindex
            dy = parent.rowindex - self.rowindex
            self.no_flow[dy + 1, dx + 1] = 0
            self.persistence += parent.z_delta * _PERSISTENCE_LUT[dy + 1, dx + 1]

    def calc_distribution(self):
        """Hot path — single call into Numba JIT."""
        n_parents = len(self.parent)
        if n_parents == 0:
            parent_dx       = np.zeros(1, dtype=np.int64)
            parent_dy       = np.zeros(1, dtype=np.int64)
            parent_z_delta  = np.zeros(1, dtype=np.float64)
            parent_is_start = False
        else:
            parent_dx       = np.array([p.colindex - self.colindex for p in self.parent], dtype=np.int64)
            parent_dy       = np.array([p.rowindex - self.rowindex for p in self.parent], dtype=np.int64)
            parent_z_delta  = np.array([p.z_delta                  for p in self.parent], dtype=np.float64)
            parent_is_start = self.parent[0].is_start

        rows, cols, fluxes, z_deltas, z_delta_neighbour = numba_calc_distribution(
            self.altitude,
            self.dem_ng.astype(np.float64),
            float(self.z_delta),
            self._ds_alpha,
            self.max_z_delta,
            self.cellsize,
            self.exp,
            self.flux_threshold,
            float(self.flux),
            self.is_start,
            parent_is_start,
            n_parents,
            parent_dx,
            parent_dy,
            parent_z_delta,
            _PERSISTENCE_LUT_NP,
            self.rowindex,
            self.colindex,
        )

        self.z_delta_neighbour = z_delta_neighbour

        if not self.is_start:
            self.calc_fp_travelangle()
            self.calc_sl_travelangle()

        if len(fluxes) > 1:
            order    = np.argsort(z_deltas)
            rows     = rows[order]
            cols     = cols[order]
            fluxes   = fluxes[order]
            z_deltas = z_deltas[order]

        return rows, cols, fluxes, z_deltas


# ---------------------------------------------------------------------------
# calculation_effect
# ---------------------------------------------------------------------------

def calculation_effect(args):
    dem            = args[0]
    header         = args[1]
    release        = args[2]
    alpha          = args[3]
    exp            = args[4]
    flux_threshold = args[5]
    max_z_delta    = args[6]

    z_delta_array        = np.zeros_like(dem)
    z_delta_sum          = np.zeros_like(dem)
    flux_array           = np.zeros_like(dem)
    count_array          = np.zeros_like(dem)
    backcalc             = np.zeros_like(dem)
    fp_travelangle_array = np.zeros_like(dem)
    sl_travelangle_array = np.zeros_like(dem)

    cellsize = header["cellsize"]
    nodata   = header["noDataValue"]

    start = datetime.now().replace(microsecond=0)
    row_list, col_list = get_start_idx(dem, release)

    startcell_idx = 0
    while startcell_idx < len(row_list):
        log.debug('Calculating Startcell: %d of %d = %.2f%%',
                  startcell_idx + 1, len(row_list),
                  (startcell_idx + 1) / len(row_list) * 100)

        row_idx = row_list[startcell_idx]
        col_idx = col_list[startcell_idx]
        dem_ng = dem[row_idx - 1:row_idx + 2, col_idx - 1:col_idx + 2]
        if (nodata in dem_ng) or np.size(dem_ng) < 9:
            startcell_idx += 1
            continue

        cell_list  = []
        cell_index = {}  # (row, col) -> index — O(1) lookup

        startcell = CellFast(row_idx, col_idx, dem_ng, cellsize, 1, 0, None,
                             alpha, exp, flux_threshold, max_z_delta, True)
        cell_list.append(startcell)
        cell_index[(row_idx, col_idx)] = 0

        for idx, cell in enumerate(cell_list):
            rows, cols, fluxes, z_deltas = cell.calc_distribution()

            if len(fluxes) == 0:
                continue

            is_new = np.ones(len(rows), dtype=bool)

            for k in range(len(rows)):
                key = (rows[k], cols[k])
                if key in cell_index:
                    existing = cell_list[cell_index[key]]
                    existing.add_os(fluxes[k])
                    existing.add_parent(cell)
                    if z_deltas[k] > existing.z_delta:
                        existing.z_delta = z_deltas[k]
                    is_new[k] = False

            for k in np.where(is_new)[0]:
                dem_ng = dem[rows[k] - 1:rows[k] + 2, cols[k] - 1:cols[k] + 2]
                if (nodata in dem_ng) or np.size(dem_ng) < 9:
                    continue
                key = (rows[k], cols[k])
                cell_index[key] = len(cell_list)
                cell_list.append(
                    CellFast(rows[k], cols[k], dem_ng, cellsize,
                             fluxes[k], z_deltas[k], cell,
                             alpha, exp, flux_threshold, max_z_delta, startcell)
                )

        for cell in cell_list:
            r, c = cell.rowindex, cell.colindex
            z_delta_array[r, c]        = max(z_delta_array[r, c],        cell.z_delta)
            flux_array[r, c]           = max(flux_array[r, c],           cell.flux)
            count_array[r, c]         += 1
            z_delta_sum[r, c]         += cell.z_delta
            fp_travelangle_array[r, c] = max(fp_travelangle_array[r, c], cell.max_gamma)
            sl_travelangle_array[r, c] = max(sl_travelangle_array[r, c], cell.sl_gamma)

        # Break reference cycles before dropping cell_list
        for cell in cell_list:
            cell.parent = []
        del cell_list
        del cell_index

        startcell_idx += 1

    end = datetime.now().replace(microsecond=0)
    log.debug('Time needed: %s', end - start)

    return (z_delta_array, flux_array, count_array,
            z_delta_sum, backcalc, fp_travelangle_array, sl_travelangle_array)