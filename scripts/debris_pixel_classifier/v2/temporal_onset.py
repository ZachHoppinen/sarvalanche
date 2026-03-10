"""Temporal onset localization via bump detection on CNN probability time series.

Stage 2 of the two-stage pipeline:
  Stage 1 (CNN) identifies *where* debris exists (per-date probability maps).
  Stage 2 (this script) identifies *when* the avalanche occurred.

Physical model:
  The CNN is run at each acquisition date with temporal weights (tau) that
  decay away from that reference date. Each date's inference uses mostly
  independent SAR pairs (at tau=6, pairs >12 days away have negligible
  weight). So each above-threshold time step represents a largely
  independent confirmation from different orbit passes.

  Real debris:  detected across MULTIPLE independent acquisition dates.
                The backscatter change persists because the deposit is
                physically present. Bump width at short tau directly
                measures how many independent passes confirm the signal.
  Noise:        single-pass spike. Only one acquisition date fires,
                because it was one weird SAR pair. Longer tau can smear
                this into a fake bump, but at short tau it's width=1.

Inputs:
  --cnn-nc     NetCDF from full_season_inference.py with debris_probability(time, y, x)
  --sar-nc     Original season_dataset.nc with VV, VH(time, y, x) and track coords

Outputs:
  NetCDF with (y, x) maps:
    onset_date            - estimated avalanche date (datetime64), at peak probability
    onset_step_idx        - index into time axis of peak
    peak_prob             - maximum CNN probability reached
    bump_width            - contiguous steps above threshold around peak (multi-pass confirmation)
    n_above_threshold     - total steps above threshold (not necessarily contiguous)
    mean_detection_prob   - mean probability across steps above threshold
    persistence_ratio     - fraction of first-to-last detection span with prob > lower threshold
    bump_smoothness       - temporal smoothness of the bump shape (vs jagged noise)
    step_height_vv        - VV backscatter step at the peak (dB)
    step_height_vh        - VH backscatter step at the peak (dB)
    confidence            - 0-1 composite confidence score
    spike_flag            - True if detection looks like transient noise (1-step)
    spatial_bump_amplitude - Gaussian-smoothed neighborhood prob at peak minus baseline
    spatial_peak_alignment - does neighborhood peak match pixel peak timing
    spatial_bump_symmetry  - symmetry of spatial probability rise/fall
    pre_existing          - debris present from first observation

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/temporal_onset.py \
        --cnn-nc local/issw/v2_season_inference/season_v2_debris_probabilities.nc \
        --sar-nc local/issw/dual_tau_output/zone/season_dataset.nc \
        --threshold 0.5
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from scipy import ndimage

from sarvalanche.io.dataset import load_netcdf_to_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak / bump detection
# ---------------------------------------------------------------------------

def find_peak_1d(series: np.ndarray, threshold: float) -> tuple[int, float]:
    """Find the time index of peak probability above threshold.

    Parameters
    ----------
    series : (T,) float array, may contain NaN

    Returns
    -------
    peak_idx : int, index of maximum probability (-1 if never above threshold)
    peak_val : float, the peak probability value
    """
    valid = ~np.isnan(series)
    if valid.sum() == 0:
        return -1, 0.0

    s = np.where(valid, series, -np.inf)
    peak_idx = int(np.argmax(s))
    peak_val = float(series[peak_idx])

    if peak_val < threshold:
        return -1, peak_val

    return peak_idx, peak_val


def find_all_peaks_1d(
    series: np.ndarray,
    threshold: float,
    min_separation: int = 3,
    max_peaks: int = 5,
) -> list[tuple[int, float]]:
    """Find ALL distinct peaks above threshold in a 1D time series.

    Multiple avalanches can hit the same path in one season. This function
    finds each distinct bump by iteratively masking already-found peaks.

    Parameters
    ----------
    series : (T,) float array, may contain NaN
    threshold : minimum probability to qualify as a peak
    min_separation : minimum time steps between distinct peaks
    max_peaks : maximum number of peaks to return

    Returns
    -------
    peaks : list of (peak_idx, peak_val), sorted by peak_val descending.
            Empty list if no peaks above threshold.
    """
    valid = ~np.isnan(series)
    if valid.sum() == 0:
        return []

    s = series.copy()
    peaks = []

    for _ in range(max_peaks):
        s_masked = np.where(valid & ~np.isnan(s), s, -np.inf)
        idx = int(np.argmax(s_masked))
        val = float(s_masked[idx])

        if val < threshold:
            break

        peaks.append((idx, val))

        # Mask out the bump around this peak so we find the next one
        lo = idx
        while lo > 0 and (s[lo - 1] >= threshold * 0.5 or abs(lo - 1 - idx) < min_separation):
            lo -= 1
        hi = idx
        T = len(series)
        while hi < T - 1 and (s[hi + 1] >= threshold * 0.5 or abs(hi + 1 - idx) < min_separation):
            hi += 1
        s[lo:hi + 1] = -np.inf

    return peaks


def measure_bump_1d(
    series: np.ndarray,
    peak_idx: int,
    threshold: float,
    persistence_threshold: float = 0.3,
) -> tuple[int, int, float, float, float]:
    """Measure multi-pass confirmation metrics around a peak.

    The key insight: each above-threshold time step at short tau represents
    a largely independent SAR pass confirming the detection. More passes =
    more likely real debris.

    Parameters
    ----------
    series : (T,) float
    peak_idx : index of peak
    threshold : level defining the bump extent
    persistence_threshold : lower threshold for persistence ratio

    Returns
    -------
    width : int, contiguous steps above threshold around peak
    n_above : int, total steps above threshold (anywhere in time series)
    mean_det_prob : float, mean probability across steps above threshold
    persistence : float in [0, 1], fraction of span from first to last
        detection that has prob > persistence_threshold
    smoothness : float in [0, 1], low second-derivative energy = smooth bump
    """
    T = len(series)
    if peak_idx < 0:
        return 0, 0, 0.0, 0.0, 0.0

    above = np.where(~np.isnan(series), series >= threshold, False)

    # Contiguous width around peak
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < T - 1 and above[right + 1]:
        right += 1
    width = right - left + 1

    # Total steps above threshold (multi-pass count)
    n_above = int(above.sum())

    # Mean probability across all above-threshold steps
    if n_above > 0:
        mean_det_prob = float(np.nanmean(series[above]))
    else:
        mean_det_prob = 0.0

    # Persistence: fraction of first-to-last detection span above lower threshold
    above_indices = np.where(above)[0]
    if len(above_indices) >= 2:
        first_det = above_indices[0]
        last_det = above_indices[-1]
        span = series[first_det:last_det + 1]
        span_valid = ~np.isnan(span)
        if span_valid.sum() > 0:
            above_lower = span[span_valid] >= persistence_threshold
            persistence = float(above_lower.sum() / span_valid.sum())
        else:
            persistence = 0.0
    elif n_above == 1:
        persistence = 0.0  # single step, no persistence
    else:
        persistence = 0.0

    # Smoothness: ratio of second-derivative energy to first-derivative energy
    bump_slice = series[left:right + 1].copy()
    nans = np.isnan(bump_slice)
    if nans.all() or len(bump_slice) < 3:
        return width, n_above, mean_det_prob, persistence, 1.0

    if nans.any() and (~nans).sum() >= 2:
        bump_slice[nans] = np.interp(
            np.where(nans)[0], np.where(~nans)[0], bump_slice[~nans]
        )

    d1 = np.diff(bump_slice)
    if len(d1) < 2:
        return width, n_above, mean_det_prob, persistence, 1.0

    d2 = np.diff(d1)
    energy_d1 = np.sum(d1 ** 2)
    energy_d2 = np.sum(d2 ** 2)

    if energy_d1 < 1e-10:
        smoothness = 1.0
    else:
        smoothness = float(np.clip(1.0 - energy_d2 / (energy_d1 + 1e-10), 0, 1))

    return width, n_above, mean_det_prob, persistence, smoothness


def find_peaks_batch(
    cube: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find peaks and measure multi-pass confirmation for all pixels.

    Parameters
    ----------
    cube : (T, N) float array

    Returns
    -------
    peak_idx : (N,) int
    peak_val : (N,) float
    bump_width : (N,) int, contiguous steps above threshold
    n_above : (N,) int, total steps above threshold
    mean_det_prob : (N,) float, mean prob across above-threshold steps
    persistence : (N,) float, fraction of detection span above lower threshold
    bump_smoothness : (N,) float
    """
    T, N = cube.shape
    peak_idx = np.full(N, -1, dtype=np.int32)
    peak_val = np.zeros(N, dtype=np.float32)
    bump_width = np.zeros(N, dtype=np.int32)
    n_above = np.zeros(N, dtype=np.int32)
    mean_det_prob = np.zeros(N, dtype=np.float32)
    persistence = np.zeros(N, dtype=np.float32)
    bump_smoothness = np.zeros(N, dtype=np.float32)

    for i in range(N):
        peak_idx[i], peak_val[i] = find_peak_1d(cube[:, i], threshold)
        bump_width[i], n_above[i], mean_det_prob[i], persistence[i], bump_smoothness[i] = measure_bump_1d(
            cube[:, i], peak_idx[i], threshold
        )

    return peak_idx, peak_val, bump_width, n_above, mean_det_prob, persistence, bump_smoothness


MAX_PEAKS = 5  # Maximum number of onset peaks stored per pixel


def find_all_peaks_batch(
    cube: np.ndarray,
    threshold: float,
    max_peaks: int = MAX_PEAKS,
    min_separation: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Find all distinct peaks for all pixels.

    Parameters
    ----------
    cube : (T, N) float array
    threshold : minimum probability for a peak
    max_peaks : maximum peaks per pixel
    min_separation : minimum time steps between peaks

    Returns
    -------
    all_peak_idx : (max_peaks, N) int, -1 for unused slots
    all_peak_val : (max_peaks, N) float, 0.0 for unused slots
    """
    T, N = cube.shape
    all_peak_idx = np.full((max_peaks, N), -1, dtype=np.int32)
    all_peak_val = np.zeros((max_peaks, N), dtype=np.float32)

    for i in range(N):
        peaks = find_all_peaks_1d(cube[:, i], threshold, min_separation, max_peaks)
        for k, (pidx, pval) in enumerate(peaks):
            all_peak_idx[k, i] = pidx
            all_peak_val[k, i] = pval

    return all_peak_idx, all_peak_val


# ---------------------------------------------------------------------------
# Backscatter step at peak
# ---------------------------------------------------------------------------

def fit_step_at_peak(
    sar_ts: np.ndarray,
    peak_idx: np.ndarray,
) -> np.ndarray:
    """Measure backscatter change at each pixel's peak time.

    Compares mean SAR before peak vs mean SAR after peak.

    Parameters
    ----------
    sar_ts : (T, N) backscatter time series (dB)
    peak_idx : (N,) peak time index

    Returns
    -------
    step_height : (N,) float, mean_after - mean_before (dB)
    """
    T, N = sar_ts.shape
    step_height = np.zeros(N, dtype=np.float32)

    for i in range(N):
        k = peak_idx[i]
        if k < 0 or k >= T:
            continue

        before = sar_ts[:k, i] if k > 0 else np.array([sar_ts[0, i]])
        after = sar_ts[k:, i]

        before_valid = before[~np.isnan(before)]
        after_valid = after[~np.isnan(after)]

        if len(before_valid) > 0 and len(after_valid) > 0:
            step_height[i] = np.mean(after_valid) - np.mean(before_valid)

    return step_height


# ---------------------------------------------------------------------------
# SAR time alignment
# ---------------------------------------------------------------------------

def average_sar_by_date(sar_ds: xr.Dataset, target_dates: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """For each target date, average all SAR acquisitions within 1 day.

    Returns
    -------
    vv_avg : (T, Y, X)
    vh_avg : (T, Y, X) or None
    """
    sar_times = pd.DatetimeIndex(sar_ds["time"].values)
    target_dates = pd.DatetimeIndex(target_dates)

    H = sar_ds.sizes["y"]
    W = sar_ds.sizes["x"]
    T = len(target_dates)

    vv_avg = np.full((T, H, W), np.nan, dtype=np.float32)
    has_vh = "VH" in sar_ds
    vh_avg = np.full((T, H, W), np.nan, dtype=np.float32) if has_vh else None

    for t, td in enumerate(target_dates):
        mask = np.abs(sar_times - td) <= pd.Timedelta(days=1)
        if mask.sum() == 0:
            continue
        vv_avg[t] = np.nanmean(sar_ds["VV"].values[mask], axis=0)
        if has_vh:
            vh_avg[t] = np.nanmean(sar_ds["VH"].values[mask], axis=0)

    return vv_avg, vh_avg


# ---------------------------------------------------------------------------
# Spike detection (noise vs real bump)
# ---------------------------------------------------------------------------

def detect_spikes(
    bump_width: np.ndarray,
    min_width: int = 2,
) -> np.ndarray:
    """Flag detections that look like transient noise rather than real bumps.

    A spike is a detection with width < min_width (too narrow for the
    overlapping pair structure). Real debris with tau=6 should produce
    bumps spanning at least 2-3 acquisition dates.

    Parameters
    ----------
    bump_width : (N,) int, width of bump above threshold
    min_width : int, minimum bump width to be considered real

    Returns
    -------
    is_spike : (N,) bool
    """
    return bump_width < min_width


# ---------------------------------------------------------------------------
# Gaussian-smoothed spatial context
# ---------------------------------------------------------------------------

def gaussian_smooth_prob_cube(
    cnn_probs: np.ndarray,
    sigma_px: float = 3.0,
) -> np.ndarray:
    """Apply Gaussian spatial smoothing to each time slice of the prob cube.

    NaNs are handled by smoothing a weights mask in parallel so the result
    is a proper weighted average (NaN-aware Gaussian blur).

    Parameters
    ----------
    cnn_probs : (T, H, W) probability cube, may contain NaN
    sigma_px : Gaussian kernel sigma in pixels

    Returns
    -------
    smoothed : (T, H, W) Gaussian-weighted local mean probability
    """
    T, H, W = cnn_probs.shape
    smoothed = np.zeros_like(cnn_probs)

    for t in range(T):
        frame = cnn_probs[t].copy()
        valid = ~np.isnan(frame)
        frame_filled = np.where(valid, frame, 0.0)

        # Gaussian blur of values and weights separately → weighted mean
        blurred_vals = ndimage.gaussian_filter(frame_filled, sigma=sigma_px, mode="constant", cval=0)
        blurred_weights = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma_px, mode="constant", cval=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            smoothed[t] = np.where(blurred_weights > 1e-10, blurred_vals / blurred_weights, 0.0)

    return smoothed.astype(np.float32)


def compute_local_prob_bump(
    smoothed_cube: np.ndarray,
    peak_idx_map: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Characterize the rise-then-fall of Gaussian-smoothed local probability.

    For each candidate pixel, extracts the smoothed probability time series
    (which represents the Gaussian-weighted neighborhood average) and
    measures:

    1. bump_amplitude: peak smoothed prob - baseline (mean of first/last 2 steps).
       Higher = larger area of high probability around this pixel at peak time.

    2. bump_symmetry: how symmetric the rise and fall are. Computed as
       correlation between the rising and (time-reversed) falling portions.
       Real debris should rise and fall similarly. Noise is one-sided.

    3. peak_alignment: does the smoothed local prob peak at the same time as
       this pixel's individual peak? Real debris: yes (the whole neighborhood
       responds together). Noise: the individual pixel peaks but the
       neighborhood doesn't care.

    Parameters
    ----------
    smoothed_cube : (T, H, W) Gaussian-smoothed probability
    peak_idx_map : (H, W) int, per-pixel peak time index
    candidate_mask : (H, W) bool

    Returns
    -------
    bump_amplitude : (H, W) float, peak smoothed prob minus baseline
    bump_symmetry : (H, W) float in [0, 1]
    peak_alignment : (H, W) float in [0, 1], 1.0 = smoothed peak matches pixel peak
    """
    T, H, W = smoothed_cube.shape
    bump_amplitude = np.zeros((H, W), dtype=np.float32)
    bump_symmetry = np.zeros((H, W), dtype=np.float32)
    peak_alignment = np.zeros((H, W), dtype=np.float32)

    cy, cx = np.where(candidate_mask)

    for y, x in zip(cy, cx):
        pk = peak_idx_map[y, x]
        if pk < 0:
            continue

        ts = smoothed_cube[:, y, x]

        # Baseline: mean of first and last 2 valid values
        n_edge = min(2, T // 3)
        baseline = (np.mean(ts[:max(n_edge, 1)]) + np.mean(ts[-max(n_edge, 1):])) / 2.0
        bump_amplitude[y, x] = ts[pk] - baseline

        # Peak alignment: how close is the smoothed peak to pixel peak?
        smoothed_peak = int(np.argmax(ts))
        dist = abs(smoothed_peak - pk)
        # Score: 1.0 if same step, decays with distance
        peak_alignment[y, x] = 1.0 / (1.0 + dist)

        # Symmetry: compare shape of rise vs fall
        rise = ts[:pk + 1]
        fall = ts[pk:]

        if len(rise) < 2 or len(fall) < 2:
            bump_symmetry[y, x] = 0.5
            continue

        # Normalize both to [0, 1] range for comparison
        rise_norm = rise - rise[0]
        fall_norm = fall - fall[-1]
        rise_range = rise_norm[-1] - rise_norm[0]
        fall_range = fall_norm[0] - fall_norm[-1]

        if abs(rise_range) < 1e-6 or abs(fall_range) < 1e-6:
            bump_symmetry[y, x] = 0.5
            continue

        rise_norm = rise_norm / rise_range  # 0 → 1
        fall_norm = fall_norm / fall_range  # 1 → 0

        # Resample shorter side to match longer for correlation
        shorter_len = min(len(rise_norm), len(fall_norm))
        if shorter_len < 2:
            bump_symmetry[y, x] = 0.5
            continue

        rise_resampled = np.interp(
            np.linspace(0, 1, shorter_len),
            np.linspace(0, 1, len(rise_norm)),
            rise_norm,
        )
        # Reverse the fall so both go 0→1 for correlation
        fall_reversed = fall_norm[::-1]
        fall_resampled = np.interp(
            np.linspace(0, 1, shorter_len),
            np.linspace(0, 1, len(fall_reversed)),
            fall_reversed,
        )

        # Pearson correlation — perfect symmetry = 1.0
        corr = np.corrcoef(rise_resampled, fall_resampled)[0, 1]
        bump_symmetry[y, x] = np.clip((corr + 1) / 2.0, 0, 1)  # map [-1,1] → [0,1]

    return bump_amplitude, bump_symmetry, peak_alignment


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(
    peak_val: np.ndarray,
    bump_width: np.ndarray,
    n_above: np.ndarray,
    mean_det_prob: np.ndarray,
    persistence: np.ndarray,
    bump_smoothness: np.ndarray,
    vv_step_height: np.ndarray,
    spatial_peak_alignment: np.ndarray,
    spatial_bump_amplitude: np.ndarray,
    max_width: int = 8,
    vh_step_height: np.ndarray | None = None,
) -> np.ndarray:
    """Combine multi-pass confirmation and spatial metrics into 0-1 confidence.

    Heavily weights multi-pass confirmation — the most reliable indicator
    that debris was independently detected by multiple SAR passes.

    Factors (in order of importance):
    - Bump width / n_above_threshold (multi-pass confirmation, dominant)
    - Persistence (signal visible across detection span)
    - Mean detection probability (consistently high across passes)
    - Spatial bump amplitude (neighborhood also sees it)
    - Spatial peak alignment (neighborhood peaks at same time)
    - VV backscatter step magnitude
    - Peak probability
    - Bump smoothness
    """
    peak_score = np.clip(peak_val, 0, 1)
    # Width: 3+ confirming passes is strong, 5+ is very strong
    width_score = np.clip(bump_width.astype(np.float32) / max_width, 0, 1)
    # n_above can exceed contiguous width (non-contiguous detections)
    n_above_score = np.clip(n_above.astype(np.float32) / max_width, 0, 1)
    # Mean detection prob: higher = more consistently detected
    mean_det_score = np.clip(mean_det_prob, 0, 1)
    # Persistence: what fraction of the detection span stays above 0.3
    persist_score = np.clip(persistence, 0, 1)

    vv_mag = np.clip(np.abs(vv_step_height) / 3.0, 0, 1)
    amp_score = np.clip(spatial_bump_amplitude / 0.5, 0, 1)

    if vh_step_height is not None:
        vh_mag = np.clip(np.abs(vh_step_height) / 3.0, 0, 1)
        conf = (
            0.20 * width_score
            + 0.10 * n_above_score
            + 0.10 * mean_det_score
            + 0.10 * persist_score
            + 0.05 * bump_smoothness
            + 0.05 * vv_mag
            + 0.05 * vh_mag
            + 0.05 * peak_score
            + 0.15 * spatial_peak_alignment
            + 0.15 * amp_score
        )
    else:
        conf = (
            0.20 * width_score
            + 0.10 * n_above_score
            + 0.10 * mean_det_score
            + 0.10 * persist_score
            + 0.05 * bump_smoothness
            + 0.08 * vv_mag
            + 0.07 * peak_score
            + 0.15 * spatial_peak_alignment
            + 0.15 * amp_score
        )

    return np.clip(conf, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_temporal_onset(
    cnn_ds: xr.Dataset,
    sar_ds: xr.Dataset,
    threshold: float = 0.5,
    pre_existing_max_idx: int = 1,
    min_bump_width: int = 2,
    spatial_radius_px: int = 3,
) -> xr.Dataset:
    """Run temporal onset detection via multi-pass confirmation.

    For each candidate pixel (ever above threshold), finds the probability
    peak and counts how many independent acquisition dates confirm the
    detection. Real debris is seen by multiple SAR passes across multiple
    days. Single-pass spikes are flagged as noise.

    Parameters
    ----------
    cnn_ds : xr.Dataset
        CNN output with debris_probability(time, y, x)
    sar_ds : xr.Dataset
        Original SAR dataset with VV, VH(time, y, x)
    threshold : float
        CNN probability threshold for candidate pixels
    pre_existing_max_idx : int
        Peak indices <= this are flagged as pre-existing debris
    min_bump_width : int
        Bumps narrower than this are flagged as spikes
    spatial_radius_px : int
        Radius for spatial coherence and growth symmetry

    Returns
    -------
    xr.Dataset with onset detection results
    """
    cnn_times = pd.DatetimeIndex(cnn_ds["time"].values)
    cnn_probs = cnn_ds["debris_probability"].values  # (T, Y, X)
    T, H, W = cnn_probs.shape

    log.info("CNN cube: %d times, %d x %d spatial", T, H, W)

    # Get SAR data averaged to CNN time steps
    vv_avg, vh_avg = average_sar_by_date(sar_ds, cnn_times.values)
    log.info("SAR data aligned to %d CNN time steps", T)

    # Convert SAR to dB if not already
    if np.nanmedian(vv_avg[np.isfinite(vv_avg)]) < 1.0:
        log.info("SAR appears to be in dB scale already")
    else:
        log.info("Converting SAR to dB scale")
        with np.errstate(divide="ignore", invalid="ignore"):
            vv_avg = 10.0 * np.log10(np.clip(vv_avg, 1e-10, None))
            if vh_avg is not None:
                vh_avg = 10.0 * np.log10(np.clip(vh_avg, 1e-10, None))

    # Find candidate pixels: ever above threshold
    max_prob = np.nanmax(cnn_probs, axis=0)
    candidate_mask = max_prob >= threshold
    n_candidates = candidate_mask.sum()
    log.info(
        "Candidate pixels (max prob >= %.2f): %d / %d (%.1f%%)",
        threshold, n_candidates, H * W, 100 * n_candidates / (H * W),
    )

    if n_candidates == 0:
        log.warning("No candidate pixels found. Lower --threshold?")
        return _empty_result(cnn_ds, cnn_times)

    # Flatten candidates
    cy, cx = np.where(candidate_mask)
    cnn_ts = cnn_probs[:, cy, cx]  # (T, N)
    vv_ts = vv_avg[:, cy, cx]
    vh_ts = vh_avg[:, cy, cx] if vh_avg is not None else None

    # ── Peak finding and multi-pass confirmation ────────────────────────
    log.info("Finding probability peaks and measuring multi-pass confirmation...")
    peak_idx, peak_val, bump_width, n_above, mean_det_prob, persistence, bump_smooth = find_peaks_batch(cnn_ts, threshold)

    # ── Multi-peak detection (multiple avalanches per season) ─────────
    log.info("Finding all distinct peaks per pixel (max %d)...", MAX_PEAKS)
    all_peak_idx, all_peak_val = find_all_peaks_batch(
        cnn_ts, threshold, max_peaks=MAX_PEAKS, min_separation=3
    )
    n_with_multi = (all_peak_idx[1, :] >= 0).sum()
    log.info("  %d / %d candidates have 2+ distinct peaks", n_with_multi, n_candidates)

    # ── Backscatter step at peak ──────────────────────────────────────
    log.info("Measuring VV backscatter step at peak...")
    vv_step = fit_step_at_peak(vv_ts, peak_idx)

    vh_step = None
    if vh_ts is not None:
        log.info("Measuring VH backscatter step at peak...")
        vh_step = fit_step_at_peak(vh_ts, peak_idx)

    # ── Spike detection ───────────────────────────────────────────────
    spike_flag = detect_spikes(bump_width, min_width=min_bump_width)
    n_spikes = spike_flag.sum()
    log.info("  %d / %d candidates flagged as spikes (bump width < %d)",
             n_spikes, n_candidates, min_bump_width)

    # ── Pre-existing flag ─────────────────────────────────────────────
    pre_existing = peak_idx <= pre_existing_max_idx

    # Map peak index to date
    onset_dates = np.array(
        [cnn_times[min(max(k, 0), T - 1)] for k in peak_idx],
        dtype="datetime64[ns]",
    )

    # ── Gaussian spatial context ────────────────────────────────────────
    log.info("Gaussian-smoothing probability cube (sigma=%d px)...", spatial_radius_px)
    smoothed = gaussian_smooth_prob_cube(cnn_probs, sigma_px=float(spatial_radius_px))

    log.info("Characterizing local probability bump shape...")
    peak_idx_map_full = np.full((H, W), -1, dtype=np.int32)
    peak_idx_map_full[cy, cx] = peak_idx

    spatial_amp, spatial_sym, spatial_align = compute_local_prob_bump(
        smoothed, peak_idx_map_full, candidate_mask,
    )

    # ── Confidence (includes spatial context) ─────────────────────────
    confidence = compute_confidence(
        peak_val, bump_width, n_above, mean_det_prob, persistence, bump_smooth,
        vv_step,
        spatial_align[cy, cx], spatial_amp[cy, cx],
        vh_step_height=vh_step,
    )
    confidence[spike_flag] *= 0.3

    # ── Assemble spatial maps ─────────────────────────────────────────
    onset_date_map = np.full((H, W), np.datetime64("NaT"), dtype="datetime64[ns]")
    peak_prob_map = np.full((H, W), np.nan, dtype=np.float32)
    width_map = np.full((H, W), 0, dtype=np.int32)
    n_above_map = np.full((H, W), 0, dtype=np.int32)
    mean_det_prob_map = np.full((H, W), np.nan, dtype=np.float32)
    persistence_map = np.full((H, W), np.nan, dtype=np.float32)
    smooth_map = np.full((H, W), np.nan, dtype=np.float32)
    step_vv_map = np.full((H, W), np.nan, dtype=np.float32)
    step_vh_map = np.full((H, W), np.nan, dtype=np.float32)
    conf_map = np.full((H, W), np.nan, dtype=np.float32)
    spike_map = np.zeros((H, W), dtype=bool)
    pre_existing_map = np.zeros((H, W), dtype=bool)

    onset_date_map[cy, cx] = onset_dates
    peak_idx_map_full[cy, cx] = peak_idx

    # Multi-peak onset date maps: (max_peaks, H, W)
    all_onset_date_maps = np.full((MAX_PEAKS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")
    all_peak_prob_maps = np.full((MAX_PEAKS, H, W), np.nan, dtype=np.float32)
    for k in range(MAX_PEAKS):
        k_dates = np.array(
            [cnn_times[min(max(idx, 0), T - 1)] if idx >= 0 else np.datetime64("NaT")
             for idx in all_peak_idx[k]],
            dtype="datetime64[ns]",
        )
        # Set NaT for unused slots
        k_dates[all_peak_idx[k] < 0] = np.datetime64("NaT")
        all_onset_date_maps[k, cy, cx] = k_dates
        all_peak_prob_maps[k, cy, cx] = all_peak_val[k]
    peak_prob_map[cy, cx] = peak_val
    width_map[cy, cx] = bump_width
    n_above_map[cy, cx] = n_above
    mean_det_prob_map[cy, cx] = mean_det_prob
    persistence_map[cy, cx] = persistence
    smooth_map[cy, cx] = bump_smooth
    step_vv_map[cy, cx] = vv_step
    if vh_step is not None:
        step_vh_map[cy, cx] = vh_step
    conf_map[cy, cx] = confidence
    spike_map[cy, cx] = spike_flag
    pre_existing_map[cy, cx] = pre_existing

    # ── Build output dataset ──────────────────────────────────────────
    coords = {"y": cnn_ds.y, "x": cnn_ds.x, "peak_rank": np.arange(MAX_PEAKS)}
    result = xr.Dataset(
        {
            # Primary (strongest) peak — backward compatible
            "onset_date": (["y", "x"], onset_date_map),
            "onset_step_idx": (["y", "x"], peak_idx_map_full),
            "peak_prob": (["y", "x"], peak_prob_map),
            "bump_width": (["y", "x"], width_map),
            "n_above_threshold": (["y", "x"], n_above_map),
            "mean_detection_prob": (["y", "x"], mean_det_prob_map),
            "persistence_ratio": (["y", "x"], persistence_map),
            "bump_smoothness": (["y", "x"], smooth_map),
            "step_height_vv": (["y", "x"], step_vv_map),
            "step_height_vh": (["y", "x"], step_vh_map),
            "confidence": (["y", "x"], conf_map),
            "spike_flag": (["y", "x"], spike_map),
            "spatial_bump_amplitude": (["y", "x"], spatial_amp),
            "spatial_bump_symmetry": (["y", "x"], spatial_sym),
            "spatial_peak_alignment": (["y", "x"], spatial_align),
            "pre_existing": (["y", "x"], pre_existing_map),
            "candidate_mask": (["y", "x"], candidate_mask),
            # Multi-peak onset dates (peak_rank, y, x)
            # peak_rank=0 is the strongest peak, 1 is second strongest, etc.
            "all_onset_dates": (["peak_rank", "y", "x"], all_onset_date_maps),
            "all_peak_probs": (["peak_rank", "y", "x"], all_peak_prob_maps),
        },
        coords=coords,
    )

    if cnn_ds.rio.crs is not None:
        result = result.rio.write_crs(cnn_ds.rio.crs)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("Onset detection complete:")
    log.info("  Candidates: %d pixels", n_candidates)
    log.info("  Spikes: %d (%.1f%%)", n_spikes, 100 * n_spikes / max(n_candidates, 1))
    log.info("  Pre-existing: %d (%.1f%%)", pre_existing.sum(), 100 * pre_existing.mean())
    log.info("  Mean confidence: %.3f", np.nanmean(confidence))
    log.info("  Mean bump width: %.1f steps (multi-pass confirmation)", np.mean(bump_width))
    log.info("  Mean n_above_threshold: %.1f steps", np.mean(n_above))
    log.info("  Mean detection prob: %.3f", np.nanmean(mean_det_prob))
    log.info("  Mean persistence: %.3f", np.nanmean(persistence))
    log.info("  Mean spatial bump amplitude: %.3f", np.nanmean(spatial_amp[candidate_mask]))
    log.info("  Mean spatial peak alignment: %.3f", np.nanmean(spatial_align[candidate_mask]))
    log.info("  Mean spatial bump symmetry: %.3f", np.nanmean(spatial_sym[candidate_mask]))

    # Per-date peak histogram
    for t in range(T):
        n_real = ((peak_idx == t) & ~spike_flag).sum()
        n_spike = ((peak_idx == t) & spike_flag).sum()
        if n_real > 0 or n_spike > 0:
            log.info("  %s: %d detections, %d spikes", cnn_times[t].date(), n_real, n_spike)

    return result


def _empty_result(cnn_ds, cnn_times):
    """Return empty result dataset when no candidates found."""
    H, W = cnn_ds.sizes["y"], cnn_ds.sizes["x"]
    coords = {"y": cnn_ds.y, "x": cnn_ds.x, "peak_rank": np.arange(MAX_PEAKS)}
    result = xr.Dataset(
        {
            "onset_date": (["y", "x"], np.full((H, W), np.datetime64("NaT"), dtype="datetime64[ns]")),
            "onset_step_idx": (["y", "x"], np.full((H, W), -1, dtype=np.int32)),
            "peak_prob": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "bump_width": (["y", "x"], np.zeros((H, W), dtype=np.int32)),
            "n_above_threshold": (["y", "x"], np.zeros((H, W), dtype=np.int32)),
            "mean_detection_prob": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "persistence_ratio": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "bump_smoothness": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "step_height_vv": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "step_height_vh": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "confidence": (["y", "x"], np.full((H, W), np.nan, dtype=np.float32)),
            "spike_flag": (["y", "x"], np.zeros((H, W), dtype=bool)),
            "spatial_bump_amplitude": (["y", "x"], np.zeros((H, W), dtype=np.float32)),
            "spatial_bump_symmetry": (["y", "x"], np.zeros((H, W), dtype=np.float32)),
            "spatial_peak_alignment": (["y", "x"], np.zeros((H, W), dtype=np.float32)),
            "pre_existing": (["y", "x"], np.zeros((H, W), dtype=bool)),
            "candidate_mask": (["y", "x"], np.zeros((H, W), dtype=bool)),
            "all_onset_dates": (["peak_rank", "y", "x"], np.full((MAX_PEAKS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")),
            "all_peak_probs": (["peak_rank", "y", "x"], np.full((MAX_PEAKS, H, W), np.nan, dtype=np.float32)),
        },
        coords=coords,
    )
    if cnn_ds.rio.crs is not None:
        result = result.rio.write_crs(cnn_ds.rio.crs)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Temporal onset localization from CNN debris probabilities + SAR backscatter",
    )
    parser.add_argument("--cnn-nc", type=Path, required=True,
                        help="NetCDF with debris_probability(time, y, x) from full_season_inference")
    parser.add_argument("--sar-nc", type=Path, required=True,
                        help="Original season_dataset.nc with VV, VH backscatter")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="CNN probability threshold for candidate pixels (default: 0.5)")
    parser.add_argument("--pre-existing-idx", type=int, default=1,
                        help="Peak indices <= this flagged as pre-existing (default: 1)")
    parser.add_argument("--min-bump-width", type=int, default=2,
                        help="Min bump width (steps) to not be a spike (default: 2)")
    parser.add_argument("--spatial-radius", type=int, default=3,
                        help="Radius in pixels for spatial coherence (default: 3)")
    parser.add_argument("--out", type=Path, default=None, help="Output NetCDF path")
    args = parser.parse_args()

    # Load CNN probabilities
    log.info("Loading CNN probabilities: %s", args.cnn_nc)
    cnn_ds = xr.open_dataset(args.cnn_nc)
    if not np.issubdtype(cnn_ds["time"].dtype, np.datetime64):
        cnn_ds["time"] = pd.DatetimeIndex(cnn_ds["time"].values)
    log.info("  %d time steps, %d x %d", len(cnn_ds.time), cnn_ds.sizes["y"], cnn_ds.sizes["x"])

    # Load SAR dataset
    log.info("Loading SAR dataset: %s", args.sar_nc)
    sar_ds = load_netcdf_to_dataset(args.sar_nc)
    if not np.issubdtype(sar_ds["time"].dtype, np.datetime64):
        sar_ds["time"] = pd.DatetimeIndex(sar_ds["time"].values)
    if any(var.chunks is not None for var in sar_ds.variables.values()):
        sar_ds = sar_ds.load()
    log.info("  %d time steps, %d x %d", len(sar_ds.time), sar_ds.sizes["y"], sar_ds.sizes["x"])

    # Run onset detection
    result = run_temporal_onset(
        cnn_ds, sar_ds,
        threshold=args.threshold,
        pre_existing_max_idx=args.pre_existing_idx,
        min_bump_width=args.min_bump_width,
        spatial_radius_px=args.spatial_radius,
    )

    # Save
    out_path = args.out or (args.cnn_nc.parent / "temporal_onset.nc")
    result.to_netcdf(out_path)
    log.info("Saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
