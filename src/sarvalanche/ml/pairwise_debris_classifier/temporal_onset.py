"""Pair-aware temporal onset detection for the pairwise debris classifier.

Multi-peak: a pixel can have multiple avalanche events per season.
Firing pairs are clustered into distinct events based on temporal gaps
in the [t_start, t_end] windows. Each event gets its own appearance
date (bracket) and confidence.

Key concepts
------------
- A pair **fires** at a pixel if ``prob >= threshold`` and has SAR coverage.
- Each firing pair provides a window ``[t_start, t_end]`` bracketing an event.
- Events are separated by temporal gaps: consecutive firing clusters whose
  ``t_end`` values are more than ``gap_days`` apart are treated as separate
  events.
- **appearance_date** = ``max(t_start)`` of firing pairs in an event.
  This is a *conservative* estimate — the latest possible onset date.
  A long-span pair starting early and a short-span pair starting late
  will produce a late appearance date even if debris appeared near the
  early start. ``min_t_start`` gives the earliest possible onset.
- **disappearance_date** = ``min(t_start)`` of non-firing pairs *from the
  same track(s)* after the event's appearance date. Track-aware: a non-fire
  on track A does not confirm disappearance for an event only on track B.
- **date_fires** (third return value) is indexed by ``unique_dates`` which
  are the *t_end* values of pairs (observation dates). A pair "fires" at
  its end/observation date.
- Confidence is per-event (``event_confidence``) with a backward-compatible
  per-pixel summary (``confidence`` = max across events).
- Per-pixel HRRR melt weighting (not scene-mean).

Magic numbers
-------------
- ``gap_days=18``: ~1.5× the Sentinel-1 repeat cycle (12 days). Ensures
  a single missed acquisition doesn't split one event into two.
- Bracket score constants 12 / 60: 12 = S1 repeat cycle (best achievable
  bracket), 60 = max useful bracket window. Score is 1.0 at 12 days,
  linearly decays to 0.2 at 60+ days.
- ``track_score=0.3`` for single-track detections: partial credit (valid
  but less independently confirmed).
- ``n_score`` clips at 6 clean dates: diminishing returns beyond 6.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr


log = logging.getLogger(__name__)

MAX_EVENTS = 5  # max events per pixel per season


def _compute_date_melt_weights(dates, hrrr_ds):
    """Compute per-date, per-pixel melt weights from HRRR.

    Returns dict mapping date -> (H, W) float32 array or scalar 1.0.
    0 = warm/melt, 1 = cold/clean.

    Supports both old HRRR format (``t2m_max``, ``pdd_24h``) and the
    new ``io.hrrr`` format (``t2m``).
    """
    from scipy.ndimage import gaussian_filter

    weights = {}
    if hrrr_ds is None:
        for d in dates:
            weights[d] = 1.0
        return weights

    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)

    # Determine which temperature variable is available
    t2m_key = None
    for key in ('t2m_max', 't2m'):
        if key in hrrr_ds:
            t2m_key = key
            break

    # Shape reference: use first available variable for ones_like
    shape_ref_key = t2m_key or ('pdd_24h' if 'pdd_24h' in hrrr_ds else None)

    for d in dates:
        diffs = np.abs(hrrr_times - d)
        ci = diffs.argmin()
        if diffs[ci].days > 2:
            weights[d] = 1.0
            continue

        if shape_ref_key is not None:
            w = np.ones_like(hrrr_ds[shape_ref_key].isel(time=0).values, dtype=np.float32)
        else:
            weights[d] = 1.0
            continue

        if 'pdd_24h' in hrrr_ds:
            pdd = hrrr_ds['pdd_24h'].isel(time=ci).values.astype(np.float32)
            pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
            w = np.minimum(w, np.clip(1.0 - pdd_smooth / 0.1, 0.0, 1.0))
        if t2m_key is not None:
            t2m = hrrr_ds[t2m_key].isel(time=ci).values.astype(np.float32)
            t2m_smooth = gaussian_filter(t2m, sigma=15, mode='nearest')
            w = np.minimum(w, np.clip((-t2m_smooth - 3.0) / 5.0, 0.0, 1.0))
        weights[d] = w

    return weights


def _split_into_events(firing_pairs_info, gap_days=18):
    """Split a pixel's firing pairs into distinct events.

    Pairs are sorted by ``t_end``. A new event starts when either:
    - The next pair's ``t_start`` is more than ``gap_days`` after the
      current cluster's last ``t_end`` (classic gap), OR
    - The next pair's ``t_end`` is more than ``gap_days`` after the
      current cluster's last ``t_end`` (prevents long-span pairs from
      bridging two otherwise-separate events).

    Parameters
    ----------
    firing_pairs_info : list of (t_start, t_end, prob, track)
    gap_days : int
        Temporal gap (days) separating distinct events.
        Default 18 ≈ 1.5× S1 12-day repeat cycle.

    Returns:
    -------
    events : list of lists, each sublist is firing pairs for one event
    """
    if not firing_pairs_info:
        return []

    sorted_pairs = sorted(firing_pairs_info, key=lambda x: x[1])

    events = [[sorted_pairs[0]]]
    for item in sorted_pairs[1:]:
        last_te = events[-1][-1][1]
        # Split if t_start gap OR t_end gap exceeds threshold
        gap_start = (item[0] - last_te).days > gap_days
        gap_end = (item[1] - last_te).days > gap_days
        if gap_start or gap_end:
            events.append([item])
        else:
            events[-1].append(item)

    return events[:MAX_EVENTS]


def run_pair_temporal_onset(
    pair_probs,
    pair_meta,
    threshold=0.2,
    hrrr_ds=None,
    melt_threshold=0.5,
    gap_days=18,
    coords=None,
    crs=None,
):
    """Run pair-aware temporal onset detection with multi-peak support.

    Candidate detection is per-track then combined:
      - Single track: 2+ clean firing dates
      - Cross-track: 2+ tracks each with 1+ clean firing date

    This handles layover correctly -- a pixel only visible to 1 track can
    still be detected if that track has enough temporal confirmation.

    Parameters
    ----------
    pair_probs : list of (H, W) np.ndarray
        Per-pair probability maps. NaN = no SAR coverage.
    pair_meta : list of dict
        Each dict has keys 'track', 't_start', 't_end', 'span_days'.
    threshold : float
        Probability threshold for a pair to "fire".
    hrrr_ds : xr.Dataset or None
        HRRR temperature data for melt weighting.
    melt_threshold : float
        Minimum melt weight for a date to be "clean".
    gap_days : int
        Temporal gap (days) separating distinct events.
    coords : dict or None
        y/x coordinate arrays for the output dataset.
    crs : str or None
        CRS string for georeferencing output variables.

    Returns:
    -------
    result : xr.Dataset
        Per-event arrays (event, y, x) and per-pixel summaries (y, x).
    unique_dates : list of pd.Timestamp
        Sorted unique t_end dates (observation dates).
    date_fires : np.ndarray, shape (T, H, W)
        Boolean array: whether any pair fired at each date/pixel.
        Indexed by ``unique_dates`` (t_end values).
    """
    H, W = pair_probs[0].shape
    N = len(pair_probs)

    # Parse timestamps
    pair_t_starts = [pd.Timestamp(str(m['t_start'])[:10]) for m in pair_meta]
    pair_t_ends = [pd.Timestamp(str(m['t_end'])[:10]) for m in pair_meta]

    # unique_dates from t_end: a pair "fires" at its observation/end date
    unique_dates = sorted(set(pair_t_ends))
    T = len(unique_dates)

    log.info('Pair temporal onset: %d pairs, %d distinct dates, %dx%d pixels', N, T, H, W)

    # Melt weights
    melt_weights = _compute_date_melt_weights(unique_dates, hrrr_ds)

    # Parse track IDs
    pair_tracks = [str(m.get('track', '')) for m in pair_meta]
    unique_tracks = sorted(set(pair_tracks))
    log.info('  Tracks: %s', unique_tracks)

    # ── Pass 1: find which pairs fire at each pixel ──────────────────
    # TODO(perf): pixel_firing_pairs is a Python dict-of-lists, O(n_pixels *
    # n_pairs). A vectorized accumulation into pre-allocated arrays would be
    # substantially faster for large scenes.
    date_fires = np.zeros((T, H, W), dtype=bool)
    pixel_firing_pairs = {}  # (y,x) -> list of (t_start, t_end, prob, track)
    n_covered = np.zeros((H, W), dtype=np.int32)  # total pairs with coverage

    # Per-track stats: clean firing dates and peak prob
    track_clean_fires = {t: np.zeros((H, W), dtype=np.int32) for t in unique_tracks}
    track_peak_prob = {t: np.zeros((H, W), dtype=np.float32) for t in unique_tracks}
    track_n_covered = {t: np.zeros((H, W), dtype=np.int32) for t in unique_tracks}

    for pi in range(N):
        prob = pair_probs[pi]
        has_data = ~np.isnan(prob)
        n_covered += has_data.astype(np.int32)
        track = pair_tracks[pi]
        track_n_covered[track] += has_data.astype(np.int32)

        fires = has_data & (prob >= threshold)
        if not fires.any():
            continue

        te_date = pair_t_ends[pi]
        ti = unique_dates.index(te_date)
        date_fires[ti] |= fires

        # Per-track: update peak prob
        better = fires & (prob > track_peak_prob[track])
        track_peak_prob[track][better] = prob[better]

        # Per-track: is this a clean firing date?
        mw = melt_weights[te_date]
        if np.isscalar(mw):
            is_clean = fires if mw >= melt_threshold else np.zeros_like(fires)
        else:
            is_clean = fires & (mw >= melt_threshold)
        track_clean_fires[track] += is_clean.astype(np.int32)

        fy, fx = np.where(fires)
        for i in range(len(fy)):
            key = (int(fy[i]), int(fx[i]))
            if key not in pixel_firing_pairs:
                pixel_firing_pairs[key] = []
            pixel_firing_pairs[key].append((pair_t_starts[pi], pair_t_ends[pi], float(prob[fy[i], fx[i]]), track))

    log.info('  Pixels with any firing pair: %d', len(pixel_firing_pairs))

    # ── Per-track candidate logic ────────────────────────────────────
    # Always require 2+ clean firing dates, but count per-track:
    #   (a) Single track with 2+ clean firing dates (including long pairs)
    #   (b) 2+ tracks each with 1+ clean firing date (cross-track = independent confirmation)
    # This ensures temporal confirmation while handling NaN/layover correctly.
    track_confirms = np.zeros((H, W), dtype=np.int32)  # tracks with 2+ clean dates

    for t in unique_tracks:
        has_coverage = track_n_covered[t] > 0
        multi_clean = track_clean_fires[t] >= 2
        confirms = has_coverage & multi_clean
        track_confirms += confirms.astype(np.int32)

    # Cross-track: 2+ tracks each saw it on at least 1 clean date
    tracks_with_clean = np.zeros((H, W), dtype=np.int32)
    for t in unique_tracks:
        tracks_with_clean += (track_clean_fires[t] >= 1).astype(np.int32)
    cross_track = tracks_with_clean >= 2

    candidate_mask = (track_confirms >= 1) | cross_track

    # Also compute overall clean dates for logging/output
    n_dates_clean = np.zeros((H, W), dtype=np.int32)
    n_dates_all = np.zeros((H, W), dtype=np.int32)
    for ti, d in enumerate(unique_dates):
        n_dates_all += date_fires[ti].astype(np.int32)
        mw = melt_weights[d]
        if np.isscalar(mw):
            if mw >= melt_threshold:
                n_dates_clean += date_fires[ti].astype(np.int32)
        else:
            n_dates_clean += (date_fires[ti] & (mw >= melt_threshold)).astype(np.int32)

    n_cand = candidate_mask.sum()
    n_single = int((track_confirms >= 1).sum())
    n_cross = int(cross_track.sum())
    log.info(
        '  Candidates: %d total — %d single-track (2+ clean dates), %d cross-track (2+ tracks × 1+ clean)',
        n_cand,
        n_single,
        n_cross,
    )

    # ── Pass 2: multi-peak event splitting for candidates ────────────
    # TODO(perf): Pass 2 re-scans all N pair probability maps to collect
    # non-firing pairs. Both passes could be merged but readability tradeoff.
    # Output: up to MAX_EVENTS events per pixel
    appearance_dates = np.full((MAX_EVENTS, H, W), np.datetime64('NaT'), dtype='datetime64[ns]')
    disappearance_dates = np.full((MAX_EVENTS, H, W), np.datetime64('NaT'), dtype='datetime64[ns]')
    # Bracket dates: the 4 key dates per event
    min_t_start = np.full((MAX_EVENTS, H, W), np.datetime64('NaT'), dtype='datetime64[ns]')  # first signal appearance
    max_t_start = np.full(
        (MAX_EVENTS, H, W), np.datetime64('NaT'), dtype='datetime64[ns]'
    )  # start of likely event period
    min_t_end = np.full((MAX_EVENTS, H, W), np.datetime64('NaT'), dtype='datetime64[ns]')  # end of likely event period
    max_t_end = np.full(
        (MAX_EVENTS, H, W), np.datetime64('NaT'), dtype='datetime64[ns]'
    )  # last image with debris increase
    event_peak_prob = np.full((MAX_EVENTS, H, W), np.nan, dtype=np.float32)
    event_n_pairs = np.zeros((MAX_EVENTS, H, W), dtype=np.int32)
    event_n_tracks = np.zeros((MAX_EVENTS, H, W), dtype=np.int32)
    n_events = np.zeros((H, W), dtype=np.int32)
    n_firing_total = np.zeros((H, W), dtype=np.int32)  # for firing ratio

    # Collect non-firing pairs per pixel for disappearance (track-aware)
    pixel_nonfiring_pairs = {}  # (y,x) -> list of (t_start, track)

    for pi in range(N):
        prob = pair_probs[pi]
        has_data = ~np.isnan(prob)
        not_fires = has_data & (prob < threshold)
        if not not_fires.any():
            continue
        ny, nx = np.where(not_fires & candidate_mask)
        for i in range(len(ny)):
            key = (int(ny[i]), int(nx[i]))
            if key not in pixel_nonfiring_pairs:
                pixel_nonfiring_pairs[key] = []
            pixel_nonfiring_pairs[key].append((pair_t_starts[pi], pair_tracks[pi]))

    log.info('  Splitting candidates into events (gap=%dd)...', gap_days)
    n_multi = 0

    cy, cx = np.where(candidate_mask)
    for ni in range(len(cy)):
        y, x = int(cy[ni]), int(cx[ni])
        key = (y, x)

        firing = pixel_firing_pairs.get(key, [])
        if not firing:
            continue

        events = _split_into_events(firing, gap_days=gap_days)
        n_events[y, x] = len(events)
        if len(events) > 1:
            n_multi += 1

        # Sort non-firing pairs by t_start for binary-search-like scan
        nf_pairs = sorted(pixel_nonfiring_pairs.get(key, []), key=lambda p: p[0])

        for ei, event_pairs in enumerate(events):
            if ei >= MAX_EVENTS:
                break

            # Bracket dates from firing pairs
            t_starts = [ts for ts, te, p, tr in event_pairs]
            t_ends = [te for ts, te, p, tr in event_pairs]
            # appearance = max(t_start): conservative estimate — the latest
            # possible onset date. min_t_start gives the earliest possible.
            app = max(t_starts)
            appearance_dates[ei, y, x] = np.datetime64(app)
            min_t_start[ei, y, x] = np.datetime64(min(t_starts))
            max_t_start[ei, y, x] = np.datetime64(max(t_starts))
            min_t_end[ei, y, x] = np.datetime64(min(t_ends))
            max_t_end[ei, y, x] = np.datetime64(max(t_ends))

            # Peak prob in this event
            best_p = max(p for ts, te, p, tr in event_pairs)
            event_peak_prob[ei, y, x] = best_p
            event_n_pairs[ei, y, x] = len(event_pairs)

            # Number of distinct tracks in this event
            tracks_in_event = set(tr for ts, te, p, tr in event_pairs)
            event_n_tracks[ei, y, x] = len(tracks_in_event)

            # Disappearance = min(t_start) of non-firing pairs after
            # appearance, filtered to tracks present in this event.
            for nf_ts, nf_track in nf_pairs:
                if nf_ts > app and nf_track in tracks_in_event:
                    disappearance_dates[ei, y, x] = np.datetime64(nf_ts)
                    break

        # Firing ratio: total firing pairs / total covered pairs
        n_firing_total[y, x] = len(firing)

    log.info('  Pixels with multiple events: %d / %d', n_multi, n_cand)
    if n_cand > 0:
        log.info('  Mean events per candidate: %.1f', n_events[candidate_mask].mean())

    # ── Confidence ───────────────────────────────────────────────────
    # Initialize all score arrays unconditionally (safe when n_cand == 0)
    from scipy.ndimage import uniform_filter

    n_score = np.zeros((H, W), dtype=np.float32)
    spatial_coherence = np.zeros((H, W), dtype=np.float32)
    cold_score = np.zeros((H, W), dtype=np.float32)
    firing_ratio = np.zeros((H, W), dtype=np.float32)
    max_tracks = np.zeros((H, W), dtype=np.float32)
    event_confidence = np.zeros((MAX_EVENTS, H, W), dtype=np.float32)

    if n_cand > 0:
        # Pixel-level scores (shared across events)
        # n_score: diminishing returns above 6 clean dates
        n_score = np.clip(n_dates_clean.astype(np.float32) / 6.0, 0, 1)

        cand_float = candidate_mask.astype(np.float32)
        peak_all = np.nanmax(event_peak_prob, axis=0)  # best peak across events
        peak_for_coh = np.where(candidate_mask, np.nan_to_num(peak_all, nan=0), 0.0)
        local_mean = uniform_filter(peak_for_coh, size=5, mode='constant', cval=0.0)
        local_count = uniform_filter(cand_float, size=5, mode='constant', cval=0.0)
        spatial_coherence = np.where(local_count > 0.01, local_mean / local_count, 0.0).astype(np.float32)
        spatial_coherence = np.clip(spatial_coherence, 0, 1)

        cold_score = np.clip((n_dates_clean.astype(np.float32) - 1) / 4.0, 0, 1)

        # Firing ratio: n_firing / n_covered. High = fires consistently
        firing_ratio = np.where(
            n_covered > 0,
            n_firing_total.astype(np.float32) / n_covered.astype(np.float32),
            0.0,
        ).astype(np.float32)

        max_tracks = np.nanmax(event_n_tracks, axis=0).astype(np.float32)
        max_tracks = np.nan_to_num(max_tracks, nan=0)

        # Per-event confidence: bracket_score and track_score vary by event,
        # while n_score, spatial_coherence, cold_score, firing_ratio are shared.
        for ei in range(MAX_EVENTS):
            has_event = event_n_pairs[ei] > 0
            if not has_event.any():
                continue

            # Bracket score per event
            # 12 = S1 repeat cycle (best achievable), 60 = max useful bracket.
            # At bracket=0 days: (0-12)/60 = -0.2, so 1.2, clipped to 1.0 — correct.
            app_ei = appearance_dates[ei]
            dis_ei = disappearance_dates[ei]
            has_both = ~np.isnat(app_ei) & ~np.isnat(dis_ei) & has_event
            b_days = np.full((H, W), np.nan, dtype=np.float32)
            b_days[has_both] = (
                (dis_ei[has_both].astype('int64') - app_ei[has_both].astype('int64')) / (1e9 * 86400)
            ).astype(np.float32)
            b_score = np.where(
                np.isnan(b_days),
                0.0,
                np.clip(1.0 - (b_days - 12) / 60.0, 0.2, 1.0),
            ).astype(np.float32)

            # Track score per event
            # 1 track = 0.3 (partial credit), 2 = 0.7, 3+ = 1.0
            et = event_n_tracks[ei].astype(np.float32)
            t_score = np.clip((et - 1) / 2.0, 0, 1).astype(np.float32)
            t_score = np.where(et == 1, 0.3, t_score)

            event_confidence[ei, has_event] = np.clip(
                0.20 * n_score[has_event]
                + 0.15 * spatial_coherence[has_event]
                + 0.15 * b_score[has_event]
                + 0.15 * cold_score[has_event]
                + 0.15 * firing_ratio[has_event]
                + 0.20 * t_score[has_event],
                0.0,
                1.0,
            )

    # Backward-compatible per-pixel confidence = max across events
    confidence = np.nanmax(event_confidence, axis=0)
    confidence = np.nan_to_num(confidence, nan=0.0).astype(np.float32)

    # ── Summary ──────────────────────────────────────────────────────
    if n_cand > 0:
        log.info('  Mean confidence: %.3f', np.nanmean(confidence[candidate_mask]))
        log.info('  Mean peak prob: %.3f', np.nanmean(np.nanmax(event_peak_prob, axis=0)[candidate_mask]))
        log.info('  Mean firing ratio: %.3f', firing_ratio[candidate_mask].mean())
        log.info('  Mean max tracks per event: %.1f', max_tracks[candidate_mask].mean())
        log.info('  Mean confirming tracks: %.1f', track_confirms[candidate_mask].mean())
        for nt in range(1, 4):
            n_with = (max_tracks[candidate_mask] >= nt).sum()
            log.info('    %d+ tracks: %d (%.0f%%)', nt, n_with, 100 * n_with / n_cand)
        n_single_only = int(((track_confirms >= 1) & ~cross_track & candidate_mask).sum())
        n_cross_only = int((cross_track & ~(track_confirms >= 1) & candidate_mask).sum())
        n_both = int(((track_confirms >= 1) & cross_track & candidate_mask).sum())
        log.info(
            '  Candidate sources: single-track-only=%d, cross-track-only=%d, both=%d',
            n_single_only,
            n_cross_only,
            n_both,
        )

    # Build output
    out_coords = coords or {}
    result = xr.Dataset(
        {
            'appearance_date': (['event', 'y', 'x'], appearance_dates),
            'disappearance_date': (['event', 'y', 'x'], disappearance_dates),
            'min_t_start': (['event', 'y', 'x'], min_t_start),
            'max_t_start': (['event', 'y', 'x'], max_t_start),
            'min_t_end': (['event', 'y', 'x'], min_t_end),
            'max_t_end': (['event', 'y', 'x'], max_t_end),
            'event_peak_prob': (['event', 'y', 'x'], event_peak_prob),
            'event_n_pairs': (['event', 'y', 'x'], event_n_pairs),
            'event_n_tracks': (['event', 'y', 'x'], event_n_tracks),
            'event_confidence': (['event', 'y', 'x'], event_confidence),
            'n_events': (['y', 'x'], n_events),
            'n_dates_clean': (['y', 'x'], n_dates_clean),
            'n_dates_all': (['y', 'x'], n_dates_all),
            'confidence': (['y', 'x'], confidence),
            'candidate_mask': (['y', 'x'], candidate_mask),
            'n_covered': (['y', 'x'], n_covered),
            'n_firing_total': (['y', 'x'], n_firing_total),
            'firing_ratio': (['y', 'x'], firing_ratio),
            'peak_prob': (['y', 'x'], np.nanmax(event_peak_prob, axis=0)),
            'tracks_confirming': (['y', 'x'], track_confirms),
            'tracks_with_clean_fire': (['y', 'x'], tracks_with_clean),
        },
        coords={**out_coords, 'event': np.arange(MAX_EVENTS)},
    )

    if crs:
        import rioxarray  # noqa: F401

        for var in result.data_vars:
            result[var] = result[var].rio.write_crs(crs)

    return result, unique_dates, date_fires
