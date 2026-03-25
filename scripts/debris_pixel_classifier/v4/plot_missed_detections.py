"""Plot missed D2.5+ detections: DEM, d_empirical, melt-filtered, v4 prob, + 4 closest pairs.

2×4 grid per observation:
  Top:    DEM | d_empirical | d_empirical_melt_filtered | v4 peak_prob
  Bottom: 4 closest same-track VV change pairs bracketing the obs date

Usage:
    conda run --no-capture-output -n sarvalanche python scripts/debris_pixel_classifier/v4/plot_missed_detections.py \
        --nc "local/issw/snfac/netcdfs/Sawtooth_&_Western_Smoky_Mtns/season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc" \
        --eval-csv local/issw/snfac/v3_experiment/v4_snfac_eval_sawtooth.csv \
        --onset-nc local/issw/snfac/v3_experiment/v4_snfac_onset_sawtooth.nc \
        --hrrr local/issw/snfac/hrrr_temperature_sawtooth_2425.nc \
        --zone Sawtooth
"""

import argparse
import ast
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
import rioxarray  # noqa: F401
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.v3.channels import N_INPUT, N_SAR, N_STATIC
from sarvalanche.ml.v3.patch_extraction import (
    V3_PATCH_SIZE,
    build_static_stack,
    normalize_dem_patch,
)
from sarvalanche.ml.v4.model import MultiScaleDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
WINDOW_PX = 60  # pixels around obs point (~1.8km at 30m)


def parse_location_point(loc_str):
    d = ast.literal_eval(loc_str)
    return float(d["lat"]), float(d["lng"])


def get_pixel_coords(ds, lat, lng):
    """Convert lat/lng to pixel y, x indices."""
    from pyproj import Transformer
    crs = str(ds.rio.crs)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    px, py = transformer.transform(lng, lat)
    x_idx = int(np.abs(ds.x.values - px).argmin())
    y_idx = int(np.abs(ds.y.values - py).argmin())
    return y_idx, x_idx


def _get_tracks(ds):
    """Group time indices by track so we only diff within the same orbit."""
    times = pd.DatetimeIndex(ds.time.values)
    track_ids = ds.track.values
    vv = ds["VV"].values  # (time, y, x), already in dB

    tracks = {}
    for tid in np.unique(track_ids):
        if np.isnan(tid):
            continue
        mask = track_ids == tid
        tracks[int(tid)] = {
            "times": times[mask],
            "vv": vv[mask],
        }
    return tracks


def _hrrr_melt_weight(hrrr_ds, sar_date, hrrr_times, pdd_threshold=0.1):
    """Compute per-pixel melt weight from HRRR. 0=warm/melt, 1=cold/clean."""
    time_diffs = np.abs(hrrr_times - sar_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 2:
        return None
    if 'pdd_24h' in hrrr_ds and 't2m_max' in hrrr_ds:
        pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
        t2m = hrrr_ds['t2m_max'].isel(time=ci).values
        pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
        t2m_smooth = gaussian_filter(t2m, sigma=15, mode='nearest')
        pdd_weight = np.clip(1.0 - pdd_smooth / pdd_threshold, 0.0, 1.0)
        t2m_weight = np.clip((-t2m_smooth - 3.0) / 5.0, 0.0, 1.0)
        return np.minimum(pdd_weight, t2m_weight).astype(np.float32)
    elif 'pdd_24h' in hrrr_ds:
        pdd = hrrr_ds['pdd_24h'].isel(time=ci).values
        pdd_smooth = gaussian_filter(pdd, sigma=15, mode='nearest')
        return np.clip(1.0 - pdd_smooth / pdd_threshold, 0.0, 1.0).astype(np.float32)
    return None


def compute_d_empirical_at_date(ds, obs_date, hrrr_ds=None, max_span_days=60):
    """Compute d_empirical and melt-filtered d_empirical for a given date.

    Returns (d_emp, d_emp_melt, n_pairs).
    """
    tracks = _get_tracks(ds)
    H, W = ds.sizes["y"], ds.sizes["x"]
    obs_dt = pd.Timestamp(obs_date)

    # Build HRRR cache
    hrrr_cache = {}
    if hrrr_ds is not None:
        hrrr_times = pd.DatetimeIndex(hrrr_ds.time.values)
        all_times = pd.DatetimeIndex(ds.time.values)
        for t in all_times:
            w = _hrrr_melt_weight(hrrr_ds, t, hrrr_times)
            if w is not None:
                hrrr_cache[t] = w

    wsum = np.zeros((H, W), dtype=np.float64)
    wt = np.zeros_like(wsum)
    wsum_melt = np.zeros_like(wsum)
    wt_melt = np.zeros_like(wsum)
    n_pairs = 0

    for tid, td in tracks.items():
        times = td["times"]
        vv = td["vv"]
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if times[i] > obs_dt or times[j] < obs_dt:
                    continue
                span = (times[j] - times[i]).days
                if span > max_span_days or span < 1:
                    continue
                diff = vv[j] - vv[i]
                w = 1.0 / (1.0 + span / 12.0)
                valid = np.isfinite(diff)
                wsum[valid] += diff[valid] * w
                wt[valid] += w
                n_pairs += 1

                # Melt-filtered: weight by min(melt_weight) of the two endpoints
                mw_i = hrrr_cache.get(times[i])
                mw_j = hrrr_cache.get(times[j])
                if mw_i is not None and mw_j is not None:
                    mw = np.minimum(mw_i, mw_j)
                elif mw_i is not None:
                    mw = mw_i
                elif mw_j is not None:
                    mw = mw_j
                else:
                    mw = None
                if mw is not None:
                    w_melt = w * mw
                    wsum_melt[valid] += diff[valid] * w_melt[valid]
                    wt_melt[valid] += w_melt[valid]
                else:
                    # No HRRR data — include unweighted
                    wsum_melt[valid] += diff[valid] * w
                    wt_melt[valid] += w

    d_emp = np.where(wt > 0, wsum / wt, np.nan).astype(np.float32)
    d_emp_melt = np.where(wt_melt > 0, wsum_melt / wt_melt, np.nan).astype(np.float32)
    return d_emp, d_emp_melt, n_pairs


def get_closest_pairs(ds, obs_date, n_pairs=4, max_span_days=60):
    """Get the n closest (shortest span) same-track pairs that bracket obs_date."""
    tracks = _get_tracks(ds)
    obs_dt = pd.Timestamp(obs_date)
    candidates = []

    # Also need VH and ANF for SAR channel construction
    vh_all = ds["VH"].values if "VH" in ds else None  # (time, y, x)
    track_ids = ds.track.values  # (time,)
    # ANF is (static_track, y, x) — build lookup by track ID
    anf_by_track = {}
    if "anf" in ds:
        static_tracks = ds.static_track.values
        anf_data = ds["anf"].values
        for si, st in enumerate(static_tracks):
            anf_by_track[int(st)] = anf_data[si]

    for tid, td in tracks.items():
        times = td["times"]
        vv = td["vv"]
        mask = track_ids == tid
        vh_track = vh_all[mask] if vh_all is not None else None
        anf_track = anf_by_track.get(tid)

        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if times[i] > obs_dt or times[j] < obs_dt:
                    continue
                span = (times[j] - times[i]).days
                if span > max_span_days or span < 1:
                    continue
                vv_diff = vv[j] - vv[i]
                vh_diff = vh_track[j] - vh_track[i] if vh_track is not None else np.zeros_like(vv_diff)
                candidates.append({
                    "track": tid,
                    "t_start": times[i], "t_end": times[j],
                    "span": span,
                    "diff": vv_diff,
                    "vv_diff_raw": vv_diff,
                    "vh_diff_raw": vh_diff,
                    "anf": anf_track if anf_track is not None else None,
                })

    candidates.sort(key=lambda x: x["span"])
    return candidates[:n_pairs]


def build_sar_channels(pair):
    """Build 5-ch SAR stack matching training pipeline: change_vv, change_vh, change_cr, anf, proximity."""
    vv = pair["vv_diff_raw"].astype(np.float32)
    vh = pair["vh_diff_raw"].astype(np.float32)
    change_vv = np.sign(vv) * np.log1p(np.abs(vv))
    change_vh = np.sign(vh) * np.log1p(np.abs(vh))
    cr = vh - vv
    change_cr = np.sign(cr) * np.log1p(np.abs(cr))
    anf = pair["anf"] if pair["anf"] is not None else np.zeros_like(vv)
    prox = np.full(vv.shape, 1.0 / (1.0 + pair["span"] / 12.0), dtype=np.float32)
    return np.stack([change_vv, change_vh, change_cr, anf, prox], axis=0)


def run_v4_pair_inference(pair, static_scene, model, device, patch_size=V3_PATCH_SIZE, stride=64):
    """Run v4 model on a single pair, return full-scene probability map."""
    sar = build_sar_channels(pair)
    H, W = sar.shape[1], sar.shape[2]

    # Pad scene for local context extraction
    margin = (patch_size * 4 - patch_size) // 2  # 192
    static_normed = normalize_dem_patch(static_scene.copy())
    scene = np.concatenate([sar, static_normed], axis=0)
    C = scene.shape[0]
    padded = np.zeros((C, H + 2 * margin, W + 2 * margin), dtype=np.float32)
    padded[:, margin:margin + H, margin:margin + W] = scene

    # Regional: whole scene downsampled to 128×128
    sar_r = resize(sar.transpose(1, 2, 0), (patch_size, patch_size),
                   order=1, preserve_range=True).transpose(2, 0, 1).astype(np.float32)
    static_r = resize(static_normed.transpose(1, 2, 0), (patch_size, patch_size),
                      order=1, preserve_range=True).transpose(2, 0, 1).astype(np.float32)
    regional = np.concatenate([sar_r, static_r], axis=0)
    regional_t = torch.from_numpy(regional[np.newaxis]).to(device)

    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    coords = [(y0, x0)
              for y0 in range(0, H - patch_size + 1, stride)
              for x0 in range(0, W - patch_size + 1, stride)]

    model.eval()
    batch_size = 64
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            B = len(batch_coords)
            fine_arr = np.empty((B, C, patch_size, patch_size), dtype=np.float32)
            ctx_arr = np.empty((B, C, patch_size, patch_size), dtype=np.float32)

            for bi, (y0, x0) in enumerate(batch_coords):
                py, px = y0 + margin, x0 + margin
                fine_arr[bi] = padded[:, py:py + patch_size, px:px + patch_size]
                ctx_size = patch_size * 4
                cy = y0 + margin - (ctx_size - patch_size) // 2
                cx = x0 + margin - (ctx_size - patch_size) // 2
                ctx_full = padded[:, cy:cy + ctx_size, cx:cx + ctx_size]
                ctx_arr[bi] = ctx_full.reshape(C, patch_size, 4, patch_size, 4).mean(axis=(2, 4))

            fine_t = torch.from_numpy(fine_arr).to(device)
            ctx_t = torch.from_numpy(ctx_arr).to(device)
            reg_t = regional_t.expand(B, -1, -1, -1)
            logits = model(fine_t, ctx_t, reg_t)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for bi, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[bi]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    prob_map = np.where(count > 0, prob_sum / count, np.nan).astype(np.float32)
    no_cov = np.abs(sar[0]) < 1e-6
    prob_map[no_cov] = np.nan
    return prob_map


def plot_one_observation(ds, hrrr_ds, static_scene, model, device, obs_row, out_dir):
    """Plot 2×4 grid for one missed observation."""
    lat, lng = parse_location_point(obs_row["location_point"])
    yi, xi = get_pixel_coords(ds, lat, lng)
    obs_date = obs_row["date"]
    loc_name = obs_row["location_name"]
    d_size = obs_row["d_size"]
    n_cand = obs_row["n_candidate_px"]

    H, W = ds.sizes["y"], ds.sizes["x"]
    y0 = max(0, yi - WINDOW_PX)
    y1 = min(H, yi + WINDOW_PX)
    x0 = max(0, xi - WINDOW_PX)
    x1 = min(W, xi + WINDOW_PX)
    # Relative obs marker position
    my, mx = yi - y0, xi - x0

    # Compute d_empirical + melt filtered
    d_emp, d_emp_melt, n_pairs_total = compute_d_empirical_at_date(ds, obs_date, hrrr_ds=hrrr_ds)

    # DEM
    dem = ds["dem"].values

    # Get 4 closest pairs
    pairs = get_closest_pairs(ds, obs_date, n_pairs=4)

    # v4 probability from closest pair
    v4_prob = None
    if pairs and pairs[0] is not None and model is not None:
        log.info("    Running v4 inference on closest pair (trk%d, %dd)...",
                 pairs[0]["track"], pairs[0]["span"])
        v4_prob = run_v4_pair_inference(pairs[0], static_scene, model, device)
    # Pad to 4 if fewer
    while len(pairs) < 4:
        pairs.append(None)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))

    marker_kw = dict(marker="+", color="k", markersize=10, markeredgewidth=2)
    vmin_db, vmax_db = -2, 4

    # ── Top row: DEM, d_empirical, d_emp_melt, v4 prob ──
    # DEM
    ax = axes[0, 0]
    dem_crop = dem[y0:y1, x0:x1]
    im = ax.imshow(dem_crop, cmap="terrain", origin="upper", interpolation="nearest")
    ax.plot(mx, my, **marker_kw)
    ax.set_title("DEM", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")

    # d_empirical
    ax = axes[0, 1]
    im = ax.imshow(d_emp[y0:y1, x0:x1], cmap="RdBu_r", vmin=vmin_db, vmax=vmax_db,
                   origin="upper", interpolation="nearest")
    ax.plot(mx, my, **marker_kw)
    ax.set_title(f"d_empirical\n({n_pairs_total} pairs)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

    # d_empirical melt filtered
    ax = axes[0, 2]
    im = ax.imshow(d_emp_melt[y0:y1, x0:x1], cmap="RdBu_r", vmin=vmin_db, vmax=vmax_db,
                   origin="upper", interpolation="nearest")
    ax.plot(mx, my, **marker_kw)
    ax.set_title("d_emp melt_filtered", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

    # v4 probability from closest pair
    ax = axes[0, 3]
    if v4_prob is not None:
        im = ax.imshow(v4_prob[y0:y1, x0:x1], cmap="hot_r", vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
        ax.plot(mx, my, **marker_kw)
        p = pairs[0]
        ax.set_title(f"v4 prob (closest)\ntrk{p['track']} {p['t_start'].strftime('%m-%d')}→{p['t_end'].strftime('%m-%d')}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="prob")
    else:
        ax.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Bottom row: 4 closest pairs ──
    for pi in range(4):
        ax = axes[1, pi]
        pair = pairs[pi]
        if pair is None:
            ax.set_visible(False)
            continue
        diff_crop = pair["diff"][y0:y1, x0:x1]
        im = ax.imshow(diff_crop, cmap="RdBu_r", vmin=vmin_db, vmax=vmax_db,
                       origin="upper", interpolation="nearest")
        ax.plot(mx, my, **marker_kw)
        t0 = pair["t_start"].strftime("%m-%d")
        t1 = pair["t_end"].strftime("%m-%d")
        ax.set_title(f"VV {t0}→{t1}\ntrk{pair['track']} ({pair['span']}d)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

    status = "spatial_only" if n_cand > 0 else "missed"
    fig.suptitle(
        f"D{d_size} {loc_name} ({obs_date}) [{status}, {n_cand}px]",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    safe_name = loc_name.replace("/", "_").replace(" ", "_")[:40]
    fname = f"missed_D{d_size}_{obs_date}_{safe_name}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--hrrr", type=Path, default=None)
    parser.add_argument("--obs", type=Path, default=ROOT / "local/issw/snfac/snfac_obs_2021_2025.csv")
    parser.add_argument("--zone", type=str, default="Sawtooth")
    parser.add_argument("--season", type=str, default="2024-2025")
    parser.add_argument("--base-ch", type=int, default=16)
    parser.add_argument("--min-dsize", type=float, default=2.5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # Load dataset
    log.info("Loading %s", args.nc)
    ds = load_netcdf_to_dataset(args.nc)
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds["time"] = pd.DatetimeIndex(ds["time"].values)
    if any(var.chunks is not None for var in ds.variables.values()):
        ds = ds.load()

    # HRRR
    hrrr_ds = None
    if args.hrrr and args.hrrr.exists():
        hrrr_ds = xr.open_dataset(args.hrrr)
        log.info("Loaded HRRR from %s", args.hrrr)

    # Build static stack
    log.info("Building static stack...")
    static_scene = build_static_stack(ds, hrrr_ds=hrrr_ds)

    # Load v4 model
    raw = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw
    # Detect in_ch from checkpoint — handle both old and new architectures
    for key in ["encoder.enc1.block.0.weight", "encoder.enc1.0.block.0.weight"]:
        if key in state_dict:
            in_ch = state_dict[key].shape[1]
            break
    else:
        in_ch = N_INPUT
    model = MultiScaleDetector(in_ch=in_ch, base_ch=args.base_ch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Loaded v4 model: %d params", sum(p.numel() for p in model.parameters()))

    # Load eval results + obs metadata
    v4 = pd.read_csv(args.eval_csv)
    obs_all = pd.read_csv(args.obs)
    obs_all["date"] = pd.to_datetime(obs_all["date"])
    season_parts = args.season.split("-")
    season = obs_all[
        (obs_all["date"] >= f"{season_parts[0]}-11-01")
        & (obs_all["date"] <= f"{season_parts[1]}-04-30")
        & obs_all["zone_name"].str.contains(args.zone, na=False)
    ]
    merged = v4.merge(
        season[["id", "location_point", "name"]],
        left_on="obs_id", right_on="id", how="left", suffixes=("", "_obs"),
    )

    # Filter to missed D2.5+
    missed = merged[
        (merged["d_size"] >= args.min_dsize) & (~merged["detected"])
    ].copy()
    log.info("Missed D%.1f+: %d observations", args.min_dsize, len(missed))

    out_dir = args.out_dir or Path("figures/snfac/v4_missed_detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in missed.iterrows():
        log.info("Plotting: D%.1f %s (%s)", row["d_size"], row["location_name"], row["date"])
        try:
            plot_one_observation(ds, hrrr_ds, static_scene, model, device, row, out_dir)
        except Exception as e:
            log.warning("  Failed: %s", e, exc_info=True)

    log.info("Done. %d plots in %s", len(missed), out_dir)


if __name__ == "__main__":
    main()
