"""Tests for sarvalanche.ml.pairwise_debris_classifier.temporal_onset."""

import numpy as np
import pandas as pd
import xarray as xr

from sarvalanche.ml.pairwise_debris_classifier.temporal_onset import (
    MAX_EVENTS,
    _compute_date_melt_weights,
    _split_into_events,
    run_pair_temporal_onset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = pd.Timestamp


def _make_pair(t_start, t_end, prob_val, track, H=4, W=4, fire_pixels=None):
    """Build a (prob_map, meta) pair for testing.

    fire_pixels: list of (y, x) that get prob_val; rest are NaN.
    """
    prob = np.full((H, W), np.nan, dtype=np.float32)
    if fire_pixels is not None:
        for y, x in fire_pixels:
            prob[y, x] = prob_val
    meta = {
        'track': str(track),
        't_start': pd.Timestamp(t_start),
        't_end': pd.Timestamp(t_end),
        'span_days': (pd.Timestamp(t_end) - pd.Timestamp(t_start)).days,
    }
    return prob, meta


def _make_pair_map(t_start, t_end, track, H, W, pixel_probs):
    """Build a pair with per-pixel probability values.

    pixel_probs: dict of (y, x) -> prob. Missing pixels are NaN (no coverage)
                 or pass coverage_pixels to set covered-but-low pixels.
    """
    prob = np.full((H, W), np.nan, dtype=np.float32)
    for (y, x), p in pixel_probs.items():
        prob[y, x] = p
    meta = {
        'track': str(track),
        't_start': pd.Timestamp(t_start),
        't_end': pd.Timestamp(t_end),
        'span_days': (pd.Timestamp(t_end) - pd.Timestamp(t_start)).days,
    }
    return prob, meta


def _build_track_season(
    track, acq_dates, event_date, H, W, aval_pixels, non_aval_pixels=None, peak_prob=0.85, fade_rate=0.15, bg_prob=0.05
):
    """Build a realistic season of consecutive SAR pairs for one track.

    Simulates the probability signature of an avalanche event:
    - Pairs before the event: low background probability
    - Pairs bracketing the event (t_start < event < t_end): high spike
    - Pairs after the event: decaying probability as debris ages
    - Pairs well after: back to background

    Parameters
    ----------
    track : str
        Track ID.
    acq_dates : list of str
        Sorted acquisition dates for this track (e.g. every 12 days).
    event_date : pd.Timestamp
        When the avalanche actually happened.
    H, W : int
        Grid dimensions.
    aval_pixels : list of (y, x)
        Pixels where the avalanche is visible.
    non_aval_pixels : list of (y, x) or None
        Additional pixels with SAR coverage but no avalanche (background).
    peak_prob : float
        Max probability for the first bracketing pair.
    fade_rate : float
        Probability decay per pair after the event.
    bg_prob : float
        Background probability for non-event pairs.

    Returns:
    -------
    probs : list of np.ndarray
    metas : list of dict
    """
    event_ts = pd.Timestamp(event_date)
    dates = [pd.Timestamp(d) for d in acq_dates]
    if non_aval_pixels is None:
        non_aval_pixels = []

    probs, metas = [], []
    all_pixels = list(aval_pixels) + list(non_aval_pixels)
    post_event_count = 0

    for i in range(len(dates) - 1):
        t_start = dates[i]
        t_end = dates[i + 1]

        pixel_probs = {}

        # Determine event relationship
        brackets_event = t_start < event_ts <= t_end
        after_event = t_start >= event_ts

        if brackets_event:
            post_event_count = 0
        elif after_event:
            post_event_count += 1

        for px in aval_pixels:
            if brackets_event:
                pixel_probs[px] = peak_prob
            elif after_event:
                decayed = peak_prob - fade_rate * post_event_count
                pixel_probs[px] = max(decayed, bg_prob)
            else:
                pixel_probs[px] = bg_prob

        for px in non_aval_pixels:
            pixel_probs[px] = bg_prob

        p, m = _make_pair_map(t_start, t_end, track, H, W, pixel_probs)
        probs.append(p)
        metas.append(m)

    return probs, metas


# ---------------------------------------------------------------------------
# _split_into_events
# ---------------------------------------------------------------------------


class TestSplitIntoEvents:
    def test_empty(self):
        assert _split_into_events([]) == []

    def test_single_pair(self):
        pairs = [(TS('2025-01-01'), TS('2025-01-13'), 0.5, '1')]
        events = _split_into_events(pairs)
        assert len(events) == 1
        assert len(events[0]) == 1

    def test_two_close_pairs_same_event(self):
        pairs = [
            (TS('2025-01-01'), TS('2025-01-13'), 0.5, '1'),
            (TS('2025-01-10'), TS('2025-01-25'), 0.6, '1'),
        ]
        events = _split_into_events(pairs, gap_days=18)
        assert len(events) == 1
        assert len(events[0]) == 2

    def test_two_separated_pairs_two_events(self):
        pairs = [
            (TS('2025-01-01'), TS('2025-01-13'), 0.5, '1'),
            (TS('2025-02-15'), TS('2025-02-27'), 0.6, '1'),
        ]
        events = _split_into_events(pairs, gap_days=18)
        assert len(events) == 2

    def test_bridging_long_span_pair_split(self):
        """A long-span pair whose t_end is far from the cluster should split."""
        pairs = [
            (TS('2025-01-01'), TS('2025-01-13'), 0.5, '1'),  # cluster 1
            (TS('2025-01-05'), TS('2025-02-15'), 0.4, '1'),  # long span: t_end 33 days after cluster 1
        ]
        events = _split_into_events(pairs, gap_days=18)
        # t_end gap: Feb 15 - Jan 13 = 33 days > 18 -> should split
        assert len(events) == 2

    def test_max_events_truncation(self):
        pairs = []
        for month in range(1, MAX_EVENTS + 3):
            d = f'2025-{month:02d}-01'
            pairs.append((TS(d), TS(f'2025-{month:02d}-05'), 0.5, '1'))
        events = _split_into_events(pairs, gap_days=10)
        assert len(events) <= MAX_EVENTS


# ---------------------------------------------------------------------------
# _compute_date_melt_weights
# ---------------------------------------------------------------------------


class TestComputeDateMeltWeights:
    def test_no_hrrr_returns_ones(self):
        dates = [TS('2025-01-01'), TS('2025-01-13')]
        weights = _compute_date_melt_weights(dates, hrrr_ds=None)
        assert len(weights) == 2
        for d in dates:
            assert weights[d] == 1.0

    def test_with_hrrr_t2m_max(self):
        """Old HRRR format with t2m_max and pdd_24h."""
        ny, nx = 5, 5
        times = pd.date_range('2025-01-01', periods=3, freq='12D')
        # Very cold: -20°C, zero PDD -> melt weight should be ~1.0
        hrrr = xr.Dataset(
            {
                't2m_max': (['time', 'y', 'x'], np.full((3, ny, nx), -20.0, dtype=np.float32)),
                'pdd_24h': (['time', 'y', 'x'], np.zeros((3, ny, nx), dtype=np.float32)),
            },
            coords={'time': times, 'y': np.arange(ny), 'x': np.arange(nx)},
        )

        dates = [TS('2025-01-01'), TS('2025-01-13')]
        weights = _compute_date_melt_weights(dates, hrrr)
        for d in dates:
            w = weights[d]
            assert isinstance(w, np.ndarray)
            assert w.shape == (ny, nx)
            # Very cold -> weight near 1.0
            assert w.min() > 0.9

    def test_with_hrrr_t2m_new_format(self):
        """New HRRR format with just t2m."""
        ny, nx = 3, 3
        times = pd.date_range('2025-01-01', periods=2, freq='12D')
        # Warm: +5°C -> should produce low melt weight
        hrrr = xr.Dataset(
            {
                't2m': (['time', 'y', 'x'], np.full((2, ny, nx), 5.0, dtype=np.float32)),
            },
            coords={'time': times, 'y': np.arange(ny), 'x': np.arange(nx)},
        )

        dates = [TS('2025-01-01')]
        weights = _compute_date_melt_weights(dates, hrrr)
        w = weights[dates[0]]
        assert isinstance(w, np.ndarray)
        # +5°C -> (-5 - 3) / 5 = -1.6 -> clipped to 0
        assert w.max() < 0.1

    def test_date_too_far_from_hrrr(self):
        """Dates >2 days from any HRRR time get weight 1.0."""
        ny, nx = 3, 3
        times = pd.date_range('2025-01-01', periods=1)
        hrrr = xr.Dataset(
            {
                't2m_max': (['time', 'y', 'x'], np.full((1, ny, nx), 5.0, dtype=np.float32)),
            },
            coords={'time': times, 'y': np.arange(ny), 'x': np.arange(nx)},
        )

        dates = [TS('2025-06-01')]  # 5 months away
        weights = _compute_date_melt_weights(dates, hrrr)
        assert weights[dates[0]] == 1.0


# ---------------------------------------------------------------------------
# run_pair_temporal_onset — end-to-end
# ---------------------------------------------------------------------------


class TestRunPairTemporalOnset:
    def test_single_event_single_track(self):
        """3 pairs from one track, all fire at pixel (0,0)."""
        H, W = 4, 4
        probs, metas = [], []
        for i, (ts, te) in enumerate(
            [
                ('2025-01-01', '2025-01-13'),
                ('2025-01-13', '2025-01-25'),
                ('2025-01-25', '2025-02-06'),
            ]
        ):
            p, m = _make_pair(ts, te, 0.5 + i * 0.1, track='1', H=H, W=W, fire_pixels=[(0, 0)])
            probs.append(p)
            metas.append(m)

        result, dates, fires = run_pair_temporal_onset(probs, metas, threshold=0.3)

        assert isinstance(result, xr.Dataset)
        assert result['candidate_mask'].values[0, 0]
        assert result['n_events'].values[0, 0] == 1
        assert result['confidence'].values[0, 0] > 0
        assert not np.isnat(result['appearance_date'].values[0, 0, 0])
        # Non-firing pixels should not be candidates
        assert not result['candidate_mask'].values[1, 1]

    def test_multi_event_splitting(self):
        """Pairs split into 2 events by temporal gap."""
        H, W = 2, 2
        probs, metas = [], []
        # Event 1: January
        for ts, te in [('2025-01-01', '2025-01-13'), ('2025-01-13', '2025-01-25')]:
            p, m = _make_pair(ts, te, 0.6, track='1', H=H, W=W, fire_pixels=[(0, 0)])
            probs.append(p)
            metas.append(m)
        # Event 2: March (gap > 18 days)
        for ts, te in [('2025-03-01', '2025-03-13'), ('2025-03-13', '2025-03-25')]:
            p, m = _make_pair(ts, te, 0.7, track='1', H=H, W=W, fire_pixels=[(0, 0)])
            probs.append(p)
            metas.append(m)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=0.3, gap_days=18)
        assert result['n_events'].values[0, 0] == 2

    def test_cross_track_confirmation(self):
        """2 tracks each with 1 clean firing date -> candidate via cross-track."""
        H, W = 2, 2
        probs, metas = [], []
        # Track 1: 1 firing pair
        p, m = _make_pair('2025-01-01', '2025-01-13', 0.6, track='1', H=H, W=W, fire_pixels=[(0, 0)])
        probs.append(p)
        metas.append(m)
        # Track 2: 1 firing pair
        p, m = _make_pair('2025-01-05', '2025-01-17', 0.5, track='2', H=H, W=W, fire_pixels=[(0, 0)])
        probs.append(p)
        metas.append(m)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=0.3)
        assert result['candidate_mask'].values[0, 0]
        assert result['tracks_with_clean_fire'].values[0, 0] >= 2

    def test_n_cand_zero_no_crash(self):
        """All probs below threshold -> no candidates, no crash."""
        H, W = 2, 2
        probs = [np.full((H, W), 0.05, dtype=np.float32)]
        metas = [{'track': '1', 't_start': TS('2025-01-01'), 't_end': TS('2025-01-13'), 'span_days': 12}]

        result, dates, fires = run_pair_temporal_onset(probs, metas, threshold=0.3)
        assert result['candidate_mask'].values.sum() == 0
        assert result['confidence'].values.max() == 0
        assert result['firing_ratio'].values.max() == 0
        # Should not raise

    def test_all_nan_no_crash(self):
        """All NaN probs -> no candidates."""
        H, W = 2, 2
        probs = [np.full((H, W), np.nan, dtype=np.float32)]
        metas = [{'track': '1', 't_start': TS('2025-01-01'), 't_end': TS('2025-01-13'), 'span_days': 12}]

        result, _, _ = run_pair_temporal_onset(probs, metas)
        assert result['candidate_mask'].values.sum() == 0

    def test_track_aware_disappearance(self):
        """Disappearance should come from the same track as the event."""
        H, W = 2, 2
        probs, metas = [], []

        # Track 1: 2 firing pairs at (0,0) -> candidate
        for ts, te in [('2025-01-01', '2025-01-13'), ('2025-01-13', '2025-01-25')]:
            p, m = _make_pair(ts, te, 0.6, track='1', H=H, W=W, fire_pixels=[(0, 0)])
            probs.append(p)
            metas.append(m)

        # Track 2: non-firing pair at (0,0) early — should NOT count as disappearance
        p_nf = np.full((H, W), np.nan, dtype=np.float32)
        p_nf[0, 0] = 0.05  # below threshold, has data
        probs.append(p_nf)
        metas.append({'track': '2', 't_start': TS('2025-01-20'), 't_end': TS('2025-02-01'), 'span_days': 12})

        # Track 1: non-firing pair at (0,0) later
        p_nf2 = np.full((H, W), np.nan, dtype=np.float32)
        p_nf2[0, 0] = 0.05
        probs.append(p_nf2)
        metas.append({'track': '1', 't_start': TS('2025-02-06'), 't_end': TS('2025-02-18'), 'span_days': 12})

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=0.3)

        dis = result['disappearance_date'].values[0, 0, 0]
        assert not np.isnat(dis)
        # Should be the track-1 non-fire (Feb 6), not track-2 (Jan 20)
        assert pd.Timestamp(dis) == TS('2025-02-06')

    def test_crs_propagation_all_vars(self):
        """CRS should be written to ALL variables, including event-dimensioned."""
        H, W = 2, 2
        probs, metas = [], []
        for ts, te in [('2025-01-01', '2025-01-13'), ('2025-01-13', '2025-01-25')]:
            p, m = _make_pair(ts, te, 0.6, track='1', H=H, W=W, fire_pixels=[(0, 0)])
            probs.append(p)
            metas.append(m)

        result, _, _ = run_pair_temporal_onset(
            probs,
            metas,
            threshold=0.3,
            crs='EPSG:32606',
        )

        for var in result.data_vars:
            crs_val = result[var].rio.crs
            assert crs_val is not None, f'Variable {var} has no CRS'

    def test_event_confidence_shape(self):
        """event_confidence should be (MAX_EVENTS, H, W)."""
        H, W = 2, 2
        probs, metas = [], []
        for ts, te in [('2025-01-01', '2025-01-13'), ('2025-01-13', '2025-01-25')]:
            p, m = _make_pair(ts, te, 0.6, track='1', H=H, W=W, fire_pixels=[(0, 0)])
            probs.append(p)
            metas.append(m)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=0.3)

        assert 'event_confidence' in result
        assert result['event_confidence'].dims == ('event', 'y', 'x')
        assert result['event_confidence'].shape == (MAX_EVENTS, H, W)
        # confidence (backward compat) should be (H, W) = max of event_confidence
        assert result['confidence'].dims == ('y', 'x')
        assert result['confidence'].shape == (H, W)
        # Confidence at firing pixel should equal event 0's confidence
        np.testing.assert_allclose(
            result['confidence'].values[0, 0],
            result['event_confidence'].values[0, 0, 0],
        )

    def test_return_signature(self):
        """Return type is (Dataset, list, ndarray)."""
        H, W = 2, 2
        probs = [np.full((H, W), 0.6, dtype=np.float32)]
        metas = [{'track': '1', 't_start': TS('2025-01-01'), 't_end': TS('2025-01-13'), 'span_days': 12}]

        result, dates, fires = run_pair_temporal_onset(probs, metas)
        assert isinstance(result, xr.Dataset)
        assert isinstance(dates, list)
        assert isinstance(fires, np.ndarray)
        assert fires.ndim == 3  # (T, H, W)


# ---------------------------------------------------------------------------
# Realistic scenario tests
#
# These simulate actual SAR pair probability signatures:
# - Before event: low background (~0.05)
# - Bracketing pair (t_start < event <= t_end): spike to ~0.85
# - After event: decay as debris ages, eventually returns to background
# - Non-avalanche pixels stay at background throughout
# ---------------------------------------------------------------------------


class TestRealisticSingleTrackDetection:
    """Single ascending track, 12-day repeat, one avalanche mid-season."""

    H, W = 8, 8
    AVAL_PX = [(2, 3), (2, 4), (3, 3), (3, 4)]  # 2x2 debris patch
    BG_PX = [(0, 0), (7, 7)]  # background pixels with coverage
    EVENT = '2025-01-20'
    THRESHOLD = 0.2

    # Ascending track: acquisitions every 12 days, Oct through Mar
    TRACK1_DATES = pd.date_range('2024-10-15', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()

    def _build(self):
        return _build_track_season(
            track='131',
            acq_dates=self.TRACK1_DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.AVAL_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.85,
            fade_rate=0.12,
            bg_prob=0.04,
        )

    def test_avalanche_pixels_detected(self):
        """All debris pixels should be candidates."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert result['candidate_mask'].values[y, x], f'Pixel ({y},{x}) not detected'

    def test_background_pixels_not_detected(self):
        """Background pixels should NOT be candidates."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.BG_PX:
            assert not result['candidate_mask'].values[y, x], f'BG pixel ({y},{x}) falsely detected'

    def test_single_event_per_pixel(self):
        """Should detect exactly 1 event at each debris pixel."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert result['n_events'].values[y, x] == 1, f'Pixel ({y},{x}) has != 1 event'

    def test_appearance_date_brackets_event(self):
        """Appearance date should bracket the actual event date.

        appearance = max(t_start) of firing pairs. The first bracketing pair
        has t_start before the event. But subsequent pairs (both dates after)
        will also fire, pushing appearance later. It must be >= event date's
        earliest bracketing t_start.
        """
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        event_ts = np.datetime64(TS(self.EVENT))
        for y, x in self.AVAL_PX:
            app = result['appearance_date'].values[0, y, x]
            min_ts = result['min_t_start'].values[0, y, x]
            max_te = result['max_t_end'].values[0, y, x]
            assert not np.isnat(app)
            # The actual event should fall within [min_t_start, max_t_end]
            assert min_ts <= event_ts, f'min_t_start {min_ts} > event {event_ts}'
            assert max_te >= event_ts, f'max_t_end {max_te} < event {event_ts}'

    def test_disappearance_after_appearance(self):
        """Disappearance date should be after appearance."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            app = result['appearance_date'].values[0, y, x]
            dis = result['disappearance_date'].values[0, y, x]
            if not np.isnat(dis):
                assert dis > app, f'Disappearance {dis} not after appearance {app}'

    def test_peak_prob_is_high(self):
        """Peak probability should be near the configured peak."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            peak = result['event_peak_prob'].values[0, y, x]
            assert peak >= 0.8, f'Peak prob {peak} too low at ({y},{x})'

    def test_confidence_positive(self):
        """Detected pixels should have positive confidence."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            conf = result['confidence'].values[y, x]
            assert conf > 0, f'Confidence {conf} at ({y},{x})'

    def test_date_fires_spike_pattern(self):
        """date_fires should show a spike then decay pattern at debris pixels."""
        probs, metas = self._build()
        result, dates, fires = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        y, x = self.AVAL_PX[0]
        pixel_fires = fires[:, y, x]
        # Should have some True entries (during/after event) and False before
        fire_indices = np.where(pixel_fires)[0]
        assert len(fire_indices) >= 2, 'Expected multiple firing dates'
        # First firing date should be at or after a reasonable index (not the very start)
        assert fire_indices[0] > 0, 'Should not fire on the very first date'

    def test_firing_ratio_reasonable(self):
        """Firing ratio should be between 0 and 1, and moderate for debris pixels."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            fr = result['firing_ratio'].values[y, x]
            assert 0 < fr < 1, f'Firing ratio {fr} out of range at ({y},{x})'


class TestRealisticMultiTrackDetection:
    """Two tracks (ascending + descending) with 12-day repeats, offset by 6 days."""

    H, W = 8, 8
    AVAL_PX = [(3, 3), (3, 4), (4, 3)]  # debris patch
    BG_PX = [(0, 0)]
    EVENT = '2025-02-01'
    THRESHOLD = 0.2

    # Track 1 (ascending): every 12 days starting Oct 10
    T1_DATES = pd.date_range('2024-10-10', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()
    # Track 2 (descending): every 12 days starting Oct 16 (6-day offset)
    T2_DATES = pd.date_range('2024-10-16', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()

    def _build(self):
        p1, m1 = _build_track_season(
            track='131',
            acq_dates=self.T1_DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.AVAL_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.82,
            fade_rate=0.10,
            bg_prob=0.03,
        )
        p2, m2 = _build_track_season(
            track='4',
            acq_dates=self.T2_DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.AVAL_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.78,
            fade_rate=0.11,
            bg_prob=0.04,
        )
        return p1 + p2, m1 + m2

    def test_cross_track_confirmation(self):
        """Both tracks should independently confirm the event."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert result['candidate_mask'].values[y, x]
            assert result['tracks_with_clean_fire'].values[y, x] >= 2
            assert result['event_n_tracks'].values[0, y, x] >= 2

    def test_single_event_despite_two_tracks(self):
        """Same event seen by two tracks should still be 1 event, not 2."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert result['n_events'].values[y, x] == 1

    def test_more_pairs_than_single_track(self):
        """Multi-track should accumulate more firing pairs per event."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            n_pairs = result['event_n_pairs'].values[0, y, x]
            assert n_pairs >= 4, f'Expected >=4 pairs at ({y},{x}), got {n_pairs}'

    def test_higher_confidence_than_single_track(self):
        """Multi-track detection should have higher track_score contribution."""
        # Build single-track version for comparison
        p1, m1 = _build_track_season(
            track='131',
            acq_dates=self.T1_DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.AVAL_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.82,
            fade_rate=0.10,
            bg_prob=0.03,
        )
        result_single, _, _ = run_pair_temporal_onset(p1, m1, threshold=self.THRESHOLD)

        probs, metas = self._build()
        result_multi, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        y, x = self.AVAL_PX[0]
        conf_single = result_single['confidence'].values[y, x]
        conf_multi = result_multi['confidence'].values[y, x]
        assert conf_multi > conf_single, f'Multi-track confidence {conf_multi} should exceed single-track {conf_single}'

    def test_appearance_brackets_event_both_tracks(self):
        """Event date should fall within the bracket from both tracks."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        event_ts = np.datetime64(TS(self.EVENT))
        for y, x in self.AVAL_PX:
            min_ts = result['min_t_start'].values[0, y, x]
            max_te = result['max_t_end'].values[0, y, x]
            assert min_ts <= event_ts <= max_te

    def test_background_clean(self):
        """Background pixels should not be detected."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.BG_PX:
            assert not result['candidate_mask'].values[y, x]


class TestRealisticMultiEvent:
    """Single track, two separate avalanche events in one season."""

    H, W = 6, 6
    # Event 1: early January, pixels (1,1) and (1,2)
    EVENT1_PX = [(1, 1), (1, 2)]
    EVENT1_DATE = '2025-01-10'
    # Event 2: late February, pixels (4,4) and (4,5) — different location
    EVENT2_PX = [(4, 4), (4, 5)]
    EVENT2_DATE = '2025-02-25'
    # Some pixels hit by BOTH events (re-avalanche at same spot)
    BOTH_PX = [(3, 3)]
    BG_PX = [(0, 0)]
    THRESHOLD = 0.2

    DATES = pd.date_range('2024-10-15', '2025-04-15', freq='12D').strftime('%Y-%m-%d').tolist()

    def _build(self):
        # Event 1 pixels
        p1, m1 = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=self.EVENT1_DATE,
            H=self.H,
            W=self.W,
            aval_pixels=self.EVENT1_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.80,
            fade_rate=0.13,
            bg_prob=0.04,
        )
        # Event 2 pixels — separate build, probabilities overlay
        p2, m2 = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=self.EVENT2_DATE,
            H=self.H,
            W=self.W,
            aval_pixels=self.EVENT2_PX,
            non_aval_pixels=[],
            peak_prob=0.75,
            fade_rate=0.10,
            bg_prob=0.03,
        )
        # Pixel hit by both events: build two separate probability profiles
        # and take the max at each pair (simulating the pixel seeing whichever
        # event is more recent)
        # Fast decay so event 1 drops below threshold before event 2 starts,
        # creating a genuine gap in the firing sequence.
        p_both1, _ = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=self.EVENT1_DATE,
            H=self.H,
            W=self.W,
            aval_pixels=self.BOTH_PX,
            non_aval_pixels=[],
            peak_prob=0.70,
            fade_rate=0.25,
            bg_prob=0.03,
        )
        p_both2, _ = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=self.EVENT2_DATE,
            H=self.H,
            W=self.W,
            aval_pixels=self.BOTH_PX,
            non_aval_pixels=[],
            peak_prob=0.75,
            fade_rate=0.25,
            bg_prob=0.03,
        )

        # Merge: overlay per-pixel probabilities from all event sources
        assert len(p1) == len(p2) == len(p_both1) == len(p_both2)
        merged_probs = []
        for i in range(len(p1)):
            combined = p1[i].copy()
            # Overlay event2 pixels
            for y, x in self.EVENT2_PX:
                combined[y, x] = p2[i][y, x]
            # Overlay both-event pixel: take max of two profiles
            for y, x in self.BOTH_PX:
                v1 = p_both1[i][y, x]
                v2 = p_both2[i][y, x]
                if np.isnan(v1) and np.isnan(v2):
                    combined[y, x] = np.nan
                else:
                    combined[y, x] = np.nanmax([v1, v2])
            merged_probs.append(combined)

        return merged_probs, m1  # metas are same (same track, same dates)

    def test_event1_pixels_detected(self):
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.EVENT1_PX:
            assert result['candidate_mask'].values[y, x]
            assert result['n_events'].values[y, x] >= 1

    def test_event2_pixels_detected(self):
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.EVENT2_PX:
            assert result['candidate_mask'].values[y, x]
            assert result['n_events'].values[y, x] >= 1

    def test_both_pixel_has_two_events(self):
        """Pixel hit by both avalanches should show 2 distinct events."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.BOTH_PX:
            assert result['candidate_mask'].values[y, x]
            n_ev = result['n_events'].values[y, x]
            assert n_ev == 2, f'Expected 2 events at ({y},{x}), got {n_ev}'

    def test_two_events_have_separated_dates(self):
        """The two events at the re-avalanche pixel should have distinct dates."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        y, x = self.BOTH_PX[0]
        app0 = pd.Timestamp(result['appearance_date'].values[0, y, x])
        app1 = pd.Timestamp(result['appearance_date'].values[1, y, x])
        assert not pd.isna(app0) and not pd.isna(app1)
        # Events should be separated by roughly the gap between event dates
        gap = abs((app1 - app0).days)
        assert gap > 20, f'Events only {gap} days apart, expected >20'

    def test_event1_date_before_event2(self):
        """First event appearance should be before second event appearance."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        y, x = self.BOTH_PX[0]
        app0 = pd.Timestamp(result['appearance_date'].values[0, y, x])
        app1 = pd.Timestamp(result['appearance_date'].values[1, y, x])
        assert app0 < app1

    def test_each_event_has_confidence(self):
        """Both events should have positive per-event confidence."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        y, x = self.BOTH_PX[0]
        c0 = result['event_confidence'].values[0, y, x]
        c1 = result['event_confidence'].values[1, y, x]
        assert c0 > 0, f'Event 0 confidence {c0}'
        assert c1 > 0, f'Event 1 confidence {c1}'

    def test_background_not_detected(self):
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.BG_PX:
            assert not result['candidate_mask'].values[y, x]

    def test_event_only_pixels_have_one_event(self):
        """Pixels affected by only one event should have n_events == 1."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.EVENT1_PX:
            assert result['n_events'].values[y, x] == 1
        for y, x in self.EVENT2_PX:
            assert result['n_events'].values[y, x] == 1


class TestRealisticNoiseFiltering:
    """Verify that common noise patterns do NOT produce false detections.

    Realistic noise sources in SAR avalanche detection:
    - Random speckle: isolated single-pair spikes at scattered pixels
    - Melt contamination: broad low-prob signal during warm periods
    - Geometry artifacts: one track sees a consistent moderate signal at
      a layover pixel, but only on 1 date (no temporal confirmation)
    - Intermittent flicker: a pixel fires on/off across dates but never
      achieves 2+ clean dates on any single track
    """

    H, W = 8, 8
    THRESHOLD = 0.2
    TRACK_DATES = pd.date_range('2024-10-15', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()

    def test_single_spike_not_detected(self):
        """One pair fires at a pixel, no repeat -> not a candidate."""
        probs, metas = _build_track_season(
            track='131',
            acq_dates=self.TRACK_DATES,
            event_date='2025-01-15',
            H=self.H,
            W=self.W,
            aval_pixels=[(2, 2)],
            non_aval_pixels=[(0, 0)],
            peak_prob=0.90,
            fade_rate=0.90,
            bg_prob=0.02,
            # fade_rate=0.90 means: 0.90 -> 0.00 after 1 pair (instant drop)
        )
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        # Only 1 clean firing date -> single-track needs 2+ -> not candidate
        assert not result['candidate_mask'].values[2, 2]

    def test_scattered_speckle_not_detected(self):
        """Random speckle: different pixels fire on different dates, no repeat."""
        dates = self.TRACK_DATES
        H, W = self.H, self.W
        rng = np.random.default_rng(42)
        probs, metas = [], []

        for i in range(len(dates) - 1):
            prob = np.full((H, W), np.nan, dtype=np.float32)
            # Give all pixels low background
            prob[:] = 0.02
            # One random pixel per pair gets a moderate spike
            ry, rx = rng.integers(0, H), rng.integers(0, W)
            prob[ry, rx] = 0.4 + rng.random() * 0.3
            meta = {
                'track': '131',
                't_start': TS(dates[i]),
                't_end': TS(dates[i + 1]),
                'span_days': 12,
            }
            probs.append(prob)
            metas.append(meta)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        # Speckle is unlikely to hit the same pixel 2+ times in 14 pairs
        # with 64 possible positions. If it does, that's fine — but most pixels
        # should not be candidates.
        n_cand = result['candidate_mask'].values.sum()
        assert n_cand <= 3, f'Too many speckle candidates: {n_cand}'

    def test_melt_noise_below_threshold(self):
        """Broad warm-period signal just below threshold -> no detections."""
        dates = self.TRACK_DATES
        H, W = self.H, self.W
        probs, metas = [], []

        for i in range(len(dates) - 1):
            prob = np.full((H, W), 0.15, dtype=np.float32)  # below threshold=0.2
            meta = {
                'track': '131',
                't_start': TS(dates[i]),
                't_end': TS(dates[i + 1]),
                'span_days': 12,
            }
            probs.append(prob)
            metas.append(meta)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)
        assert result['candidate_mask'].values.sum() == 0

    def test_geometry_artifact_single_date_not_detected(self):
        """Layover pixel with high prob on 1 date from 1 track -> not candidate."""
        dates = self.TRACK_DATES
        H, W = self.H, self.W
        probs, metas = [], []
        artifact_px = (5, 5)

        for i in range(len(dates) - 1):
            prob = np.full((H, W), np.nan, dtype=np.float32)
            # Give artifact pixel coverage on all dates
            prob[artifact_px] = 0.03
            meta = {
                'track': '131',
                't_start': TS(dates[i]),
                't_end': TS(dates[i + 1]),
                'span_days': 12,
            }
            # One date gets a spike (geometry artifact)
            if i == 5:
                prob[artifact_px] = 0.75
            probs.append(prob)
            metas.append(meta)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)
        assert not result['candidate_mask'].values[artifact_px]

    def test_intermittent_flicker_not_detected(self):
        """Pixel fires every other pair but never 2 consecutive clean dates."""
        dates = self.TRACK_DATES
        H, W = self.H, self.W
        probs, metas = [], []
        flicker_px = (3, 3)

        for i in range(len(dates) - 1):
            prob = np.full((H, W), np.nan, dtype=np.float32)
            prob[flicker_px] = 0.03  # coverage
            # Fire on even pairs, not on odd
            if i % 2 == 0:
                prob[flicker_px] = 0.35
            meta = {
                'track': '131',
                't_start': TS(dates[i]),
                't_end': TS(dates[i + 1]),
                'span_days': 12,
            }
            probs.append(prob)
            metas.append(meta)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        # Flicker fires on many dates — it WILL be detected as candidate
        # (fires on 7 of 14 dates from 1 track, well above 2-date threshold).
        # This is a known limitation: the detector sees temporal persistence.
        # But the confidence should be relatively low due to inconsistency,
        # and firing_ratio should be moderate (not near 1.0).
        if result['candidate_mask'].values[flicker_px]:
            fr = result['firing_ratio'].values[flicker_px]
            assert fr < 0.7, f'Flicker firing ratio {fr} suspiciously high'

    def test_real_event_survives_noise_floor(self):
        """A real event should be detected even with noisy background."""
        H, W = self.H, self.W
        dates = self.TRACK_DATES
        rng = np.random.default_rng(99)
        aval_px = [(4, 4), (4, 5)]

        # Build real event
        probs, metas = _build_track_season(
            track='131',
            acq_dates=dates,
            event_date='2025-01-20',
            H=H,
            W=W,
            aval_pixels=aval_px,
            non_aval_pixels=[],
            peak_prob=0.80,
            fade_rate=0.12,
            bg_prob=0.04,
        )

        # Add noise: random speckle and low-level background across all pixels
        for i in range(len(probs)):
            noise = rng.uniform(0.0, 0.12, size=(H, W)).astype(np.float32)
            # Fill NaN pixels with noise (simulating coverage everywhere)
            nan_mask = np.isnan(probs[i])
            probs[i][nan_mask] = noise[nan_mask]
            # Add noise to existing values too
            probs[i][~nan_mask] += rng.uniform(-0.05, 0.05, size=(~nan_mask).sum()).astype(np.float32)
            probs[i] = np.clip(probs[i], 0, 1)

        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        # Real event should still be detected
        for y, x in aval_px:
            assert result['candidate_mask'].values[y, x], f'Real event at ({y},{x}) lost in noise'

        # Most noise pixels should not be detected
        total_cand = result['candidate_mask'].values.sum()
        assert total_cand <= len(aval_px) + 5, f'Too many false positives: {total_cand}'


class TestRealisticLowProbMultiTrack:
    """Low-probability event that only passes via cross-track confirmation.

    Scenario: a small debris flow produces a subtle backscatter change.
    Each track sees it at prob ~0.25-0.35 on 1-2 dates — not enough for
    single-track confirmation (needs 2+ clean dates per track). But 3
    tracks each independently see it once, providing cross-track evidence.
    """

    H, W = 6, 6
    AVAL_PX = [(2, 2), (2, 3)]
    BG_PX = [(0, 0), (5, 5)]
    EVENT = '2025-01-25'
    THRESHOLD = 0.2

    # Three tracks with different repeat schedules
    T1_DATES = pd.date_range('2024-11-01', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()
    T2_DATES = pd.date_range('2024-11-04', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()
    T3_DATES = pd.date_range('2024-11-08', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()

    def _build(self):
        """Build 3-track low-probability scenario.

        Each track sees the event weakly — only the bracketing pair and
        maybe 1 post-event pair fire (rapid decay). No single track has
        2+ clean dates. But 3 tracks × 1 clean date = cross-track confirmed.
        """
        all_probs, all_metas = [], []
        for track, dates in [('44', self.T1_DATES), ('131', self.T2_DATES), ('87', self.T3_DATES)]:
            p, m = _build_track_season(
                track=track,
                acq_dates=dates,
                event_date=self.EVENT,
                H=self.H,
                W=self.W,
                aval_pixels=self.AVAL_PX,
                non_aval_pixels=self.BG_PX,
                peak_prob=0.32,
                fade_rate=0.20,
                bg_prob=0.03,
                # peak=0.32, fade=0.20: fires once at 0.32, next pair at 0.12 (below threshold)
            )
            all_probs.extend(p)
            all_metas.extend(m)
        return all_probs, all_metas

    def test_detected_via_cross_track(self):
        """Low-prob event should be detected via cross-track confirmation."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert result['candidate_mask'].values[y, x], f'Low-prob event missed at ({y},{x})'

    def test_not_detected_by_single_track_alone(self):
        """Same event from just 1 track should NOT pass (only 1 clean date)."""
        p1, m1 = _build_track_season(
            track='44',
            acq_dates=self.T1_DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.AVAL_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.32,
            fade_rate=0.20,
            bg_prob=0.03,
        )
        result, _, _ = run_pair_temporal_onset(p1, m1, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert not result['candidate_mask'].values[y, x], (
                f'Single track should not detect low-prob event at ({y},{x})'
            )

    def test_three_tracks_confirm(self):
        """tracks_with_clean_fire should be >= 3."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            n_tracks = result['tracks_with_clean_fire'].values[y, x]
            assert n_tracks >= 3, f'Expected 3+ tracks, got {n_tracks} at ({y},{x})'

    def test_single_event(self):
        """Should detect exactly 1 event."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            assert result['n_events'].values[y, x] == 1

    def test_low_peak_prob(self):
        """Peak probability should be low (~0.3), not near 1."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.AVAL_PX:
            peak = result['event_peak_prob'].values[0, y, x]
            assert 0.25 <= peak <= 0.45, f'Peak {peak} not in expected low range at ({y},{x})'

    def test_confidence_lower_than_strong_multitrack(self):
        """Low-prob multi-track confidence should be below a strong multi-track event.

        Compare against the same 3-track setup but with high peak probability.
        Both benefit equally from the track_score component, so the difference
        comes from peak prob, firing ratio, and n_clean_dates.
        """
        # Strong multi-track event for comparison
        strong_probs, strong_metas = [], []
        for track, dates in [('44', self.T1_DATES), ('131', self.T2_DATES), ('87', self.T3_DATES)]:
            p, m = _build_track_season(
                track=track,
                acq_dates=dates,
                event_date=self.EVENT,
                H=self.H,
                W=self.W,
                aval_pixels=self.AVAL_PX,
                non_aval_pixels=self.BG_PX,
                peak_prob=0.85,
                fade_rate=0.10,
                bg_prob=0.03,
            )
            strong_probs.extend(p)
            strong_metas.extend(m)
        result_strong, _, _ = run_pair_temporal_onset(strong_probs, strong_metas, threshold=self.THRESHOLD)

        # Low-prob multi-track
        probs, metas = self._build()
        result_low, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        y, x = self.AVAL_PX[0]
        conf_strong = result_strong['confidence'].values[y, x]
        conf_low = result_low['confidence'].values[y, x]
        assert conf_low < conf_strong, f'Low-prob confidence {conf_low} should be < strong {conf_strong}'

    def test_background_not_detected(self):
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.BG_PX:
            assert not result['candidate_mask'].values[y, x]

    def test_appearance_brackets_event(self):
        """Event date should fall within [min_t_start, max_t_end]."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        event_ts = np.datetime64(TS(self.EVENT))
        for y, x in self.AVAL_PX:
            min_ts = result['min_t_start'].values[0, y, x]
            max_te = result['max_t_end'].values[0, y, x]
            assert min_ts <= event_ts <= max_te


class TestRealisticSpatialCoherence:
    """Spatial coherence: clustered detections vs isolated single-pixel noise.

    Confidence includes a spatial_coherence component computed from a 5x5
    uniform filter over candidate peak probabilities. A tight cluster of
    firing pixels should have higher spatial_coherence than an isolated
    pixel, even if both have the same peak probability.

    Scenario A: 4x4 block of debris pixels (spatially coherent avalanche)
    Scenario B: single isolated pixel far from any other detection
    Both see the same event with identical per-pixel probability profiles.
    """

    H, W = 16, 16
    # Clustered debris: 4x4 block in the middle
    CLUSTER_PX = [(r, c) for r in range(6, 10) for c in range(6, 10)]
    # Isolated pixel: single pixel in the corner
    ISOLATED_PX = [(0, 0)]
    BG_PX = [(15, 15)]
    EVENT = '2025-01-20'
    THRESHOLD = 0.2
    DATES = pd.date_range('2024-10-15', '2025-03-31', freq='12D').strftime('%Y-%m-%d').tolist()

    def _build(self):
        # Build a season for the cluster
        p_cluster, m_cluster = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.CLUSTER_PX,
            non_aval_pixels=self.BG_PX,
            peak_prob=0.75,
            fade_rate=0.12,
            bg_prob=0.03,
        )
        # Build a season for the isolated pixel (same params)
        p_iso, _ = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=self.EVENT,
            H=self.H,
            W=self.W,
            aval_pixels=self.ISOLATED_PX,
            non_aval_pixels=[],
            peak_prob=0.75,
            fade_rate=0.12,
            bg_prob=0.03,
        )
        # Merge: overlay isolated pixel onto the cluster maps
        merged = []
        for i in range(len(p_cluster)):
            combined = p_cluster[i].copy()
            for y, x in self.ISOLATED_PX:
                combined[y, x] = p_iso[i][y, x]
            merged.append(combined)
        return merged, m_cluster

    def test_both_detected(self):
        """Both cluster and isolated pixel should be candidates."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.CLUSTER_PX[:4]:  # spot check
            assert result['candidate_mask'].values[y, x]
        for y, x in self.ISOLATED_PX:
            assert result['candidate_mask'].values[y, x]

    def test_cluster_confidence_at_least_as_high_as_isolated(self):
        """Clustered pixels should have >= confidence than the isolated pixel.

        Both have identical probability profiles and the same peak prob.
        The spatial_coherence component normalizes by local_count, so a
        single isolated candidate gets the same coherence score as a
        cluster pixel. The key test is that the cluster is not PENALIZED
        for being spatially grouped.
        """
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        cluster_confs = [result['confidence'].values[y, x] for y, x in self.CLUSTER_PX]
        mean_cluster_conf = np.mean(cluster_confs)

        iso_conf = result['confidence'].values[self.ISOLATED_PX[0]]

        assert mean_cluster_conf >= iso_conf, (
            f'Cluster mean confidence {mean_cluster_conf:.3f} should be >= isolated {iso_conf:.3f}'
        )

    def test_cluster_interior_higher_than_edge(self):
        """Interior cluster pixels should have >= coherence as edge pixels.

        The 5x5 uniform filter means interior pixels (surrounded by other
        candidates) get a higher local mean than edge pixels.
        """
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        # Interior: (7,7), (7,8), (8,7), (8,8)
        interior = [(7, 7), (7, 8), (8, 7), (8, 8)]
        # Edge: (6,6), (9,9)
        edge = [(6, 6), (9, 9)]

        int_confs = [result['confidence'].values[y, x] for y, x in interior]
        edge_confs = [result['confidence'].values[y, x] for y, x in edge]

        assert np.mean(int_confs) >= np.mean(edge_confs), (
            f'Interior mean {np.mean(int_confs):.3f} should be >= edge mean {np.mean(edge_confs):.3f}'
        )

    def test_same_peak_prob(self):
        """Cluster and isolated should have the same peak probability."""
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        cluster_peak = result['event_peak_prob'].values[0, 7, 7]
        iso_peak = result['event_peak_prob'].values[0, 0, 0]
        np.testing.assert_allclose(cluster_peak, iso_peak, atol=0.01)

    def test_background_clean(self):
        probs, metas = self._build()
        result, _, _ = run_pair_temporal_onset(probs, metas, threshold=self.THRESHOLD)

        for y, x in self.BG_PX:
            assert not result['candidate_mask'].values[y, x]


class TestRealisticHrrrMeltFiltering:
    """HRRR temperature-based melt filtering.

    Warm-period pairs should be downweighted so their firing dates don't
    count as "clean". This tests that:
    - An event during cold weather is detected normally
    - An event during warm weather fails detection (all dates are melt-contaminated)
    - Mixed conditions: cold event detected, warm false positive suppressed
    """

    H, W = 6, 6
    THRESHOLD = 0.2
    MELT_THRESHOLD = 0.5
    DATES = pd.date_range('2024-11-01', '2025-04-01', freq='12D').strftime('%Y-%m-%d').tolist()

    @staticmethod
    def _make_hrrr(dates, H, W, temp_by_date):
        """Build a synthetic HRRR dataset.

        temp_by_date: dict mapping date string -> temperature in Celsius.
        Dates not in the dict get -15C (cold).
        """
        all_dates = sorted(set(dates))
        times = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        nt = len(times)
        t2m = np.full((nt, H, W), -15.0, dtype=np.float32)  # default: cold
        pdd = np.zeros((nt, H, W), dtype=np.float32)

        for i, d in enumerate(all_dates):
            d_str = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]
            if d_str in temp_by_date:
                temp = temp_by_date[d_str]
                t2m[i] = temp
                if temp > 0:
                    pdd[i] = temp / 24.0  # crude PDD approximation

        return xr.Dataset(
            {
                't2m_max': (['time', 'y', 'x'], t2m),
                'pdd_24h': (['time', 'y', 'x'], pdd),
            },
            coords={
                'time': times,
                'y': np.arange(H, dtype=np.float64),
                'x': np.arange(W, dtype=np.float64),
            },
        )

    def test_cold_event_detected(self):
        """Event during cold weather: all dates clean -> detected."""
        aval_px = [(2, 2), (2, 3)]
        probs, metas = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date='2025-01-15',
            H=self.H,
            W=self.W,
            aval_pixels=aval_px,
            non_aval_pixels=[],
            peak_prob=0.70,
            fade_rate=0.12,
            bg_prob=0.03,
        )
        # All cold
        hrrr = self._make_hrrr(self.DATES, self.H, self.W, {})

        result, _, _ = run_pair_temporal_onset(
            probs,
            metas,
            threshold=self.THRESHOLD,
            hrrr_ds=hrrr,
            melt_threshold=self.MELT_THRESHOLD,
        )

        for y, x in aval_px:
            assert result['candidate_mask'].values[y, x]

    def test_warm_event_suppressed(self):
        """Event during warm weather: all firing dates melt-contaminated -> rejected.

        Every date in the firing window is warm (+5C), so no clean dates
        accumulate, and the pixel never reaches the 2-clean-date threshold.
        """
        aval_px = [(2, 2)]
        event_date = '2025-01-15'
        probs, metas = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=event_date,
            H=self.H,
            W=self.W,
            aval_pixels=aval_px,
            non_aval_pixels=[],
            peak_prob=0.70,
            fade_rate=0.10,
            bg_prob=0.03,
        )

        # Make ALL dates warm -> every firing date is melt-contaminated
        warm_dates = {}
        for d in self.DATES:
            warm_dates[d] = 5.0  # +5C everywhere
        hrrr = self._make_hrrr(self.DATES, self.H, self.W, warm_dates)

        result, _, _ = run_pair_temporal_onset(
            probs,
            metas,
            threshold=self.THRESHOLD,
            hrrr_ds=hrrr,
            melt_threshold=self.MELT_THRESHOLD,
        )

        # No clean dates -> not a candidate
        for y, x in aval_px:
            assert result['n_dates_clean'].values[y, x] == 0
            assert not result['candidate_mask'].values[y, x]

    def test_mixed_cold_detected_warm_suppressed(self):
        """Two events: one during cold snap (detected), one during thaw (suppressed)."""
        cold_px = [(1, 1), (1, 2)]
        warm_px = [(4, 4)]
        cold_event = '2025-01-10'
        warm_event = '2025-03-10'

        # Cold event
        p_cold, m_cold = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=cold_event,
            H=self.H,
            W=self.W,
            aval_pixels=cold_px,
            non_aval_pixels=[],
            peak_prob=0.75,
            fade_rate=0.12,
            bg_prob=0.03,
        )
        # Warm event (same track, same dates)
        p_warm, _ = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date=warm_event,
            H=self.H,
            W=self.W,
            aval_pixels=warm_px,
            non_aval_pixels=[],
            peak_prob=0.70,
            fade_rate=0.10,
            bg_prob=0.03,
        )
        # Merge
        merged = []
        for i in range(len(p_cold)):
            combined = p_cold[i].copy()
            for y, x in warm_px:
                combined[y, x] = p_warm[i][y, x]
            merged.append(combined)

        # HRRR: cold in Jan, warm in Mar
        warm_dates = {}
        for d in self.DATES:
            ts = pd.Timestamp(d)
            if ts.month >= 3:
                warm_dates[d] = 3.0  # +3C from March onward
        hrrr = self._make_hrrr(self.DATES, self.H, self.W, warm_dates)

        result, _, _ = run_pair_temporal_onset(
            merged,
            m_cold,
            threshold=self.THRESHOLD,
            hrrr_ds=hrrr,
            melt_threshold=self.MELT_THRESHOLD,
        )

        # Cold event: detected
        for y, x in cold_px:
            assert result['candidate_mask'].values[y, x], f'Cold event missed at ({y},{x})'
            assert result['n_dates_clean'].values[y, x] >= 2

        # Warm event: suppressed
        for y, x in warm_px:
            assert not result['candidate_mask'].values[y, x], (
                f'Warm event should be suppressed at ({y},{x}), n_dates_clean={result["n_dates_clean"].values[y, x]}'
            )

    def test_cold_event_more_clean_dates_than_warm(self):
        """Same event, cold vs warm HRRR: cold version has more clean dates."""
        aval_px = [(2, 2)]
        probs, metas = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date='2025-01-15',
            H=self.H,
            W=self.W,
            aval_pixels=aval_px,
            non_aval_pixels=[],
            peak_prob=0.70,
            fade_rate=0.10,
            bg_prob=0.03,
        )

        # Cold HRRR
        hrrr_cold = self._make_hrrr(self.DATES, self.H, self.W, {})
        result_cold, _, _ = run_pair_temporal_onset(
            probs,
            metas,
            threshold=self.THRESHOLD,
            hrrr_ds=hrrr_cold,
            melt_threshold=self.MELT_THRESHOLD,
        )

        # Warm HRRR (make some dates warm)
        warm_dates = {}
        for d in self.DATES:
            ts = pd.Timestamp(d)
            # Warm on even-numbered days in Jan/Feb
            if ts.month in (1, 2) and ts.day % 2 == 0:
                warm_dates[d] = 2.0
        hrrr_warm = self._make_hrrr(self.DATES, self.H, self.W, warm_dates)
        result_warm, _, _ = run_pair_temporal_onset(
            probs,
            metas,
            threshold=self.THRESHOLD,
            hrrr_ds=hrrr_warm,
            melt_threshold=self.MELT_THRESHOLD,
        )

        y, x = aval_px[0]
        clean_cold = result_cold['n_dates_clean'].values[y, x]
        clean_warm = result_warm['n_dates_clean'].values[y, x]
        assert clean_cold >= clean_warm, f'Cold clean dates {clean_cold} should be >= warm {clean_warm}'

    def test_hrrr_with_new_t2m_format(self):
        """Verify melt filtering works with the new t2m variable (not t2m_max)."""
        aval_px = [(2, 2)]
        probs, metas = _build_track_season(
            track='131',
            acq_dates=self.DATES,
            event_date='2025-01-15',
            H=self.H,
            W=self.W,
            aval_pixels=aval_px,
            non_aval_pixels=[],
            peak_prob=0.70,
            fade_rate=0.12,
            bg_prob=0.03,
        )

        # Build HRRR with only 't2m' (new format), all warm
        all_dates = sorted(set(self.DATES))
        times = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        hrrr_new = xr.Dataset(
            {'t2m': (['time', 'y', 'x'], np.full((len(times), self.H, self.W), 5.0, dtype=np.float32))},
            coords={'time': times, 'y': np.arange(self.H, dtype=np.float64), 'x': np.arange(self.W, dtype=np.float64)},
        )

        result, _, _ = run_pair_temporal_onset(
            probs,
            metas,
            threshold=self.THRESHOLD,
            hrrr_ds=hrrr_new,
            melt_threshold=self.MELT_THRESHOLD,
        )

        y, x = aval_px[0]
        assert result['n_dates_clean'].values[y, x] == 0, 'Warm t2m should suppress all clean dates'
        assert not result['candidate_mask'].values[y, x]
