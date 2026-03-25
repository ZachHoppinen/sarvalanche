"""Pair-aware temporal onset detection for v3 single-pair detector.

Multi-peak: a pixel can have multiple avalanche events per season.
Firing pairs are clustered into distinct events based on temporal gaps
in the [t_start, t_end] windows. Each event gets its own appearance
date (bracket) and confidence.

Key concepts:
  - A pair fires at a pixel if prob >= threshold and has SAR coverage
  - Each firing pair provides a window [t_start, t_end] bracketing an event
  - Events are separated by gaps: if no short-span pair fires between
    two clusters of firing pairs, they're different events
  - Appearance = max(t_start) within each event's firing pairs
  - Disappearance = min(t_start) of non-firing pairs after the event
  - Confidence from: n_clean_dates, spatial coherence, bracket width,
    cold confirmation
  - Per-pixel HRRR melt weighting (not scene-mean)
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

MAX_EVENTS = 5  # max events per pixel per season


def _compute_date_melt_weights(dates, hrrr_ds):
    """Compute per-date, per-pixel melt weights from HRRR.

    Returns dict mapping date → (H, W) float32 array or scalar 1.0.
    0 = warm/melt, 1 = cold/clean.
    """
    from scipy.ndimage import gaussian_filter

    weights = {}
    if hrrr_ds is None:
        for d in dates:
            weights[d] = 1.0
        return weights

    hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)

    for d in dates:
        diffs = np.abs(hrrr_times - d)
        ci = diffs.argmin()
        if diffs[ci].days > 2:
            weights[d] = 1.0
            continue

        w = np.ones_like(hrrr_ds['t2m_max'].isel(time=0).values, dtype=np.float32)
        if 'pdd_24h' in hrrr_ds:
            pdd = hrrr_ds['pdd_24h'].isel(time=ci).values.astype(np.float32)
            pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
            w = np.minimum(w, np.clip(1.0 - pdd_smooth / 0.1, 0.0, 1.0))
        if 't2m_max' in hrrr_ds:
            t2m = hrrr_ds['t2m_max'].isel(time=ci).values.astype(np.float32)
            t2m_smooth = gaussian_filter(t2m, sigma=15, mode='nearest')
            w = np.minimum(w, np.clip((-t2m_smooth - 3.0) / 5.0, 0.0, 1.0))
        weights[d] = w

    return weights


def _split_into_events(firing_pairs_info, gap_days=18):
    """Split a pixel's firing pairs into distinct events.

    Parameters
    ----------
    firing_pairs_info : list of (t_start, t_end, prob, track)

    Returns
    -------
    events : list of lists, each sublist is firing pairs for one event
    """
    if not firing_pairs_info:
        return []

    sorted_pairs = sorted(firing_pairs_info, key=lambda x: x[1])

    events = [[sorted_pairs[0]]]
    for item in sorted_pairs[1:]:
        last_te = events[-1][-1][1]
        if (item[0] - last_te).days > gap_days:
            events.append([item])
        else:
            events[-1].append(item)

    return events[:MAX_EVENTS]


def run_pair_temporal_onset(
    pair_probs,
    pair_meta,
    threshold=0.2,
    min_dates=2,
    hrrr_ds=None,
    melt_threshold=0.5,
    gap_days=18,
    coords=None,
    crs=None,
):
    """Run pair-aware temporal onset detection with multi-peak support.

    Candidate detection is per-track then combined:
      - Single track: 2+ clean firing dates, OR 1+ clean date with peak prob >= single_track_high_prob
      - Cross-track: 2+ tracks each with 1+ clean firing date

    This handles layover correctly — a pixel only visible to 1 track can still
    be detected if that track is confident enough.

    Returns dataset with per-event arrays (event_rank, y, x).
    """
    H, W = pair_probs[0].shape
    N = len(pair_probs)

    # Parse timestamps
    pair_t_starts = [pd.Timestamp(str(m['t_start'])[:10]) for m in pair_meta]
    pair_t_ends = [pd.Timestamp(str(m['t_end'])[:10]) for m in pair_meta]

    unique_dates = sorted(set(pair_t_ends))
    T = len(unique_dates)

    log.info("Pair temporal onset: %d pairs, %d distinct dates, %dx%d pixels",
             N, T, H, W)

    # Melt weights
    melt_weights = _compute_date_melt_weights(unique_dates, hrrr_ds)

    # Parse track IDs
    pair_tracks = [str(m.get('track', '')) for m in pair_meta]
    unique_tracks = sorted(set(pair_tracks))
    n_tracks = len(unique_tracks)
    log.info("  Tracks: %s", unique_tracks)

    # ── Pass 1: find which pairs fire at each pixel ──────────────────
    date_fires = np.zeros((T, H, W), dtype=bool)
    pixel_firing_pairs = {}  # (y,x) → list of (t_start, t_end, prob, track)
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
            pixel_firing_pairs[key].append(
                (pair_t_starts[pi], pair_t_ends[pi], float(prob[fy[i], fx[i]]), track)
            )

    log.info("  Pixels with any firing pair: %d", len(pixel_firing_pairs))

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
    log.info("  Candidates: %d total — %d single-track (2+ clean dates), %d cross-track (2+ tracks × 1+ clean)",
             n_cand, n_single, n_cross)

    # ── Pass 2: multi-peak event splitting for candidates ────────────
    # Output: up to MAX_EVENTS events per pixel
    appearance_dates = np.full((MAX_EVENTS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")
    disappearance_dates = np.full((MAX_EVENTS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")
    # Bracket dates: the 4 key dates per event
    min_t_start = np.full((MAX_EVENTS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")  # first signal appearance
    max_t_start = np.full((MAX_EVENTS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")  # start of likely event period
    min_t_end = np.full((MAX_EVENTS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")    # end of likely event period
    max_t_end = np.full((MAX_EVENTS, H, W), np.datetime64("NaT"), dtype="datetime64[ns]")    # last image with debris increase
    event_peak_prob = np.full((MAX_EVENTS, H, W), np.nan, dtype=np.float32)
    event_n_pairs = np.zeros((MAX_EVENTS, H, W), dtype=np.int32)
    event_n_tracks = np.zeros((MAX_EVENTS, H, W), dtype=np.int32)
    n_events = np.zeros((H, W), dtype=np.int32)
    n_firing_total = np.zeros((H, W), dtype=np.int32)  # for firing ratio

    # Also collect non-firing pairs per pixel for disappearance
    pixel_nonfiring_tstarts = {}  # (y,x) → sorted list of t_start dates

    for pi in range(N):
        prob = pair_probs[pi]
        has_data = ~np.isnan(prob)
        not_fires = has_data & (prob < threshold)
        if not not_fires.any():
            continue
        ny, nx = np.where(not_fires & candidate_mask)
        for i in range(len(ny)):
            key = (int(ny[i]), int(nx[i]))
            if key not in pixel_nonfiring_tstarts:
                pixel_nonfiring_tstarts[key] = []
            pixel_nonfiring_tstarts[key].append(pair_t_starts[pi])

    log.info("  Splitting candidates into events (gap=%dd)...", gap_days)
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

        nf_tstarts = sorted(pixel_nonfiring_tstarts.get(key, []))

        for ei, event_pairs in enumerate(events):
            if ei >= MAX_EVENTS:
                break

            # Bracket dates from firing pairs
            t_starts = [ts for ts, te, p, tr in event_pairs]
            t_ends = [te for ts, te, p, tr in event_pairs]
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

            # Disappearance = min(t_start) of non-firing pairs after appearance
            for nf_ts in nf_tstarts:
                if nf_ts > app:
                    disappearance_dates[ei, y, x] = np.datetime64(nf_ts)
                    break

        # Firing ratio: total firing pairs / total covered pairs
        n_firing_total[y, x] = len(firing)

    log.info("  Pixels with multiple events: %d / %d", n_multi, n_cand)
    if n_cand > 0:
        log.info("  Mean events per candidate: %.1f", n_events[candidate_mask].mean())

    # ── Confidence ───────────────────────────────────────────────────
    # Per-pixel (not per-event for now — use overall pixel quality)
    from scipy.ndimage import uniform_filter

    confidence = np.zeros((H, W), dtype=np.float32)
    if n_cand > 0:
        n_score = np.clip(n_dates_clean.astype(np.float32) / 6.0, 0, 1)

        cand_float = candidate_mask.astype(np.float32)
        peak_all = np.nanmax(event_peak_prob, axis=0)  # best peak across events
        peak_for_coh = np.where(candidate_mask, np.nan_to_num(peak_all, nan=0), 0.0)
        local_mean = uniform_filter(peak_for_coh, size=5, mode='constant', cval=0.0)
        local_count = uniform_filter(cand_float, size=5, mode='constant', cval=0.0)
        spatial_coherence = np.where(
            local_count > 0.01, local_mean / local_count, 0.0
        ).astype(np.float32)
        spatial_coherence = np.clip(spatial_coherence, 0, 1)

        # Bracket score from first event
        app0 = appearance_dates[0]
        dis0 = disappearance_dates[0]
        has_both = ~np.isnat(app0) & ~np.isnat(dis0)
        bracket_days = np.full((H, W), np.nan, dtype=np.float32)
        bracket_days[has_both] = (
            (dis0[has_both].astype('int64') - app0[has_both].astype('int64'))
            / (1e9 * 86400)
        ).astype(np.float32)
        bracket_score = np.where(
            np.isnan(bracket_days), 0.0,
            np.clip(1.0 - (bracket_days - 12) / 60.0, 0.2, 1.0),
        ).astype(np.float32)

        cold_score = np.clip((n_dates_clean.astype(np.float32) - 1) / 4.0, 0, 1)

        # Firing ratio: n_firing / n_covered. High = fires consistently
        firing_ratio = np.where(
            n_covered > 0,
            n_firing_total.astype(np.float32) / n_covered.astype(np.float32),
            0.0,
        ).astype(np.float32)

        # Multi-track score: max n_tracks across events for this pixel
        # 1 track = 0.3, 2 tracks = 0.7, 3+ tracks = 1.0
        # But allow confident single-track if it's the only track with coverage
        max_tracks = np.nanmax(event_n_tracks, axis=0).astype(np.float32)
        max_tracks = np.nan_to_num(max_tracks, nan=0)
        track_score = np.clip((max_tracks - 1) / 2.0, 0, 1).astype(np.float32)
        # Single track gets partial credit (0.3) — still valid, just less confirmed
        track_score = np.where(max_tracks == 1, 0.3, track_score)

        valid = candidate_mask
        confidence[valid] = np.clip(
            0.20 * n_score[valid]
            + 0.15 * spatial_coherence[valid]
            + 0.15 * bracket_score[valid]
            + 0.15 * cold_score[valid]
            + 0.15 * firing_ratio[valid]
            + 0.20 * track_score[valid],
            0.0, 1.0,
        )

    # ── Summary ──────────────────────────────────────────────────────
    if n_cand > 0:
        log.info("  Mean confidence: %.3f", np.nanmean(confidence[candidate_mask]))
        log.info("  Mean peak prob: %.3f", np.nanmean(peak_all[candidate_mask]))
        log.info("  Mean firing ratio: %.3f", firing_ratio[candidate_mask].mean())
        log.info("  Mean max tracks per event: %.1f", max_tracks[candidate_mask].mean())
        log.info("  Mean confirming tracks: %.1f", track_confirms[candidate_mask].mean())
        for nt in range(1, 4):
            n_with = (max_tracks[candidate_mask] >= nt).sum()
            log.info("    %d+ tracks: %d (%.0f%%)", nt, n_with, 100 * n_with / n_cand)
        n_single_only = int(((track_confirms >= 1) & ~cross_track & candidate_mask).sum())
        n_cross_only = int((cross_track & ~(track_confirms >= 1) & candidate_mask).sum())
        n_both = int(((track_confirms >= 1) & cross_track & candidate_mask).sum())
        log.info("  Candidate sources: single-track-only=%d, cross-track-only=%d, both=%d",
                 n_single_only, n_cross_only, n_both)

    # Build output
    out_coords = coords or {}
    result = xr.Dataset(
        {
            "appearance_date": (["event", "y", "x"], appearance_dates),
            "disappearance_date": (["event", "y", "x"], disappearance_dates),
            "min_t_start": (["event", "y", "x"], min_t_start),
            "max_t_start": (["event", "y", "x"], max_t_start),
            "min_t_end": (["event", "y", "x"], min_t_end),
            "max_t_end": (["event", "y", "x"], max_t_end),
            "event_peak_prob": (["event", "y", "x"], event_peak_prob),
            "event_n_pairs": (["event", "y", "x"], event_n_pairs),
            "event_n_tracks": (["event", "y", "x"], event_n_tracks),
            "n_events": (["y", "x"], n_events),
            "n_dates_clean": (["y", "x"], n_dates_clean),
            "n_dates_all": (["y", "x"], n_dates_all),
            "confidence": (["y", "x"], confidence),
            "candidate_mask": (["y", "x"], candidate_mask),
            "n_covered": (["y", "x"], n_covered),
            "n_firing_total": (["y", "x"], n_firing_total),
            "firing_ratio": (["y", "x"], firing_ratio if n_cand > 0 else np.zeros((H, W), dtype=np.float32)),
            "peak_prob": (["y", "x"], np.nanmax(event_peak_prob, axis=0)),
            "tracks_confirming": (["y", "x"], track_confirms),
            "tracks_with_clean_fire": (["y", "x"], tracks_with_clean),
        },
        coords={**out_coords, "event": np.arange(MAX_EVENTS)},
    )

    if crs:
        for var in result.data_vars:
            if "event" not in result[var].dims:
                result[var] = result[var].rio.write_crs(crs)

    return result, unique_dates, date_fires
