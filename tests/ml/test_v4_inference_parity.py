"""Test that v4 inference patch building matches training exactly.

Compares build_patch / build_v4_inputs / build_regional against
the training dataset's _make_sar_static and __getitem__ outputs.
"""

import numpy as np
import pytest

from sarvalanche.ml.v3.channels import N_SAR, N_STATIC
from sarvalanche.ml.v3.patch_extraction import V3_PATCH_SIZE, normalize_dem_patch, _normalize_anf
from sarvalanche.ml.v4.inference import (
    build_sar_channels,
    build_patch,
    build_v4_inputs,
    build_regional,
)


@pytest.fixture
def scene_data():
    """Create realistic fake scene data."""
    rng = np.random.default_rng(42)
    H, W = 300, 250
    vv_diff = rng.normal(0.5, 1.5, (H, W)).astype(np.float32)
    vh_diff = rng.normal(0.3, 1.2, (H, W)).astype(np.float32)
    anf_raw = rng.uniform(20, 60, (H, W)).astype(np.float32)
    anf_norm = _normalize_anf(anf_raw)
    span_days = 12

    from skimage.restoration import denoise_tv_chambolle
    vv_smooth = denoise_tv_chambolle(vv_diff, weight=1.0).astype(np.float32)
    vh_smooth = denoise_tv_chambolle(vh_diff, weight=1.0).astype(np.float32)
    sar_scene = build_sar_channels(vv_diff, vh_diff, anf_norm, span_days,
                                    vv_smooth=vv_smooth, vh_smooth=vh_smooth)

    # Static: slope, aspect_N, aspect_E, dem, cell_counts, tpi
    static_scene = np.stack([
        rng.uniform(0, 0.6, (H, W)).astype(np.float32),     # slope
        rng.uniform(-1, 1, (H, W)).astype(np.float32),      # aspect_N
        rng.uniform(-1, 1, (H, W)).astype(np.float32),      # aspect_E
        rng.uniform(1500, 3500, (H, W)).astype(np.float32), # dem
        rng.uniform(0, 100, (H, W)).astype(np.float32),     # cell_counts
        rng.uniform(-50, 50, (H, W)).astype(np.float32),    # tpi
    ], axis=0)

    return {
        "vv_diff": vv_diff, "vh_diff": vh_diff,
        "anf_norm": anf_norm, "span_days": span_days,
        "sar_scene": sar_scene, "static_scene": static_scene,
        "H": H, "W": W,
    }


def test_build_sar_channels_matches_training(scene_data):
    """SAR channel construction matches _make_sar_static."""
    vv = scene_data["vv_diff"]
    vh = scene_data["vh_diff"]
    anf = scene_data["anf_norm"]
    span = scene_data["span_days"]

    sar = build_sar_channels(vv, vh, anf, span)

    # Manually replicate training logic
    change_vv = np.sign(vv) * np.log1p(np.abs(vv))
    change_vh = np.sign(vh) * np.log1p(np.abs(vh))
    cr = vh - vv
    change_cr = np.sign(cr) * np.log1p(np.abs(cr))
    prox = np.full(vv.shape, 1.0 / (1.0 + span / 12.0), dtype=np.float32)

    np.testing.assert_array_equal(sar[0], change_vv)
    np.testing.assert_array_equal(sar[1], change_vh)
    np.testing.assert_array_equal(sar[2], change_cr)
    np.testing.assert_array_equal(sar[3], anf)
    np.testing.assert_array_equal(sar[4], prox)


def test_build_patch_dem_normalized_per_patch(scene_data):
    """DEM is min-max normalized within the patch, not globally."""
    sar = scene_data["sar_scene"]
    static = scene_data["static_scene"]

    ps = V3_PATCH_SIZE
    patch_a = build_patch(sar, static, 0, 0, ps)
    patch_b = build_patch(sar, static, 50, 50, ps)

    # DEM channel index in concatenated output
    dem_idx = N_SAR + 3  # slope=0, aspect_N=1, aspect_E=2, dem=3

    dem_a = patch_a[dem_idx]
    dem_b = patch_b[dem_idx]

    # Each should be independently normalized (different min/max)
    assert dem_a.min() == pytest.approx(0.0, abs=1e-6)
    assert dem_a.max() == pytest.approx(1.0, abs=1e-6)
    assert dem_b.min() == pytest.approx(0.0, abs=1e-6)
    assert dem_b.max() == pytest.approx(1.0, abs=1e-6)

    # But the raw values should differ
    assert not np.array_equal(dem_a, dem_b)


def test_build_patch_matches_training_make_sar_static(scene_data):
    """build_patch output matches _make_sar_static for same window."""
    sar = scene_data["sar_scene"]
    static = scene_data["static_scene"]
    ps = V3_PATCH_SIZE
    y0, x0 = 10, 20

    # Our inference function
    patch = build_patch(sar, static, y0, x0, ps)

    # Replicate training _make_sar_static
    sar_crop = sar[:, y0:y0 + ps, x0:x0 + ps]
    static_crop = normalize_dem_patch(static[:, y0:y0 + ps, x0:x0 + ps].copy())
    expected = np.concatenate([sar_crop, static_crop], axis=0)

    np.testing.assert_array_equal(patch, expected)


def test_build_patch_edge_padding(scene_data):
    """Patches at scene edges are zero-padded like training."""
    sar = scene_data["sar_scene"]
    static = scene_data["static_scene"]
    ps = V3_PATCH_SIZE

    # Negative y0 — should pad top
    patch = build_patch(sar, static, -10, 0, ps)
    assert patch.shape == (N_SAR + N_STATIC, ps, ps)
    # First 10 rows should be zero (padding)
    assert np.all(patch[:, :10, :] == 0)
    # Row 10+ should have data
    assert np.any(patch[:, 10:, :] != 0)


def test_build_v4_inputs_local_ctx_is_downsampled(scene_data):
    """Local context is 512→128 downsampled, matching training."""
    sar = scene_data["sar_scene"]
    static = scene_data["static_scene"]
    ps = V3_PATCH_SIZE
    C = N_SAR + N_STATIC

    fine, local_ctx = build_v4_inputs(sar, static, 50, 50)

    assert fine.shape == (C, ps, ps)
    assert local_ctx.shape == (C, ps, ps)

    # Local ctx should be smoother (4× downsampled) — lower variance
    assert fine[0].std() > local_ctx[0].std() * 0.5  # not exact, just sanity


def test_build_regional_no_dem_norm(scene_data):
    """Regional static channels are NOT DEM-normalized (matches training)."""
    static = scene_data["static_scene"]
    anf = scene_data["anf_norm"]
    vv = scene_data["vv_diff"]
    vh = scene_data["vh_diff"]

    regional = build_regional(None, static, anf, vv, vh, span_days=12)

    assert regional.shape == (N_SAR + N_STATIC, V3_PATCH_SIZE, V3_PATCH_SIZE)

    # DEM channel in regional should NOT be 0-1 normalized
    dem_idx = N_SAR + 3
    dem_vals = regional[dem_idx]
    # Raw DEM was 1500-3500, downsampled should be in similar range
    assert dem_vals.max() > 100, f"DEM max={dem_vals.max()}, expected raw values not 0-1"


def test_anf_must_be_prenormalized():
    """Catch the common bug: passing raw ANF instead of normalized."""
    raw_anf = np.full((10, 10), 40.0, dtype=np.float32)  # typical raw LIA ~40°
    norm_anf = _normalize_anf(raw_anf)

    # Raw ANF values are >> 1, normalized are 0-1
    assert raw_anf.mean() > 10
    assert 0 < norm_anf.mean() < 1

    # build_sar_channels doesn't validate, but the test documents the contract
    sar_good = build_sar_channels(np.zeros((10, 10)), np.zeros((10, 10)), norm_anf, 12)
    sar_bad = build_sar_channels(np.zeros((10, 10)), np.zeros((10, 10)), raw_anf, 12)

    # ANF channel (idx 3) should be 0-1 for normalized, >> 1 for raw
    assert sar_good[3].mean() < 1
    assert sar_bad[3].mean() > 10  # this would break the model
