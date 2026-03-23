"""
Generate detection GIFs for Banner Summit 2022-2023:
  1. Cumulative detections
  2. Rolling 3-step window

Both include SNFAC danger rating label.
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/issw")
NC_DIR = BASE / "netcdfs" / "Banner_Summit"
ONSET_NC = NC_DIR / "v2_season_inference" / "temporal_onset.nc"
CNN_NC = NC_DIR / "v2_season_inference" / "season_v2_debris_probabilities.nc"
SAR_NC = NC_DIR / "season_2022-2023_Banner_Summit.nc"
DANGER_CSV = BASE / "high_low_framing_outputs" / "v1_high_danger_output" / "snfac_dangers_2021_2025.csv"
OUT_DIR = BASE / "figures" / "gif"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path("/Users/zmhoppinen/Documents/sarvalanche/src")))

SEASON_START = np.datetime64("2022-11-01")
SEASON_END = np.datetime64("2023-05-01")
ZONE_NAME = "Banner Summit"
CONF_THRESH = 0.6
WINDOW = 3

# Crop bounds (set to None for full extent)
CROP_LAT = (44.30, 44.45)
CROP_LON = (-115.4, -115.15)

DANGER_LABELS = {1: "LOW", 2: "MODERATE", 3: "CONSIDERABLE", 4: "HIGH", 5: "EXTREME"}
DANGER_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#1a1a1a"}


def make_hillshade(dem, azimuth=315, altitude=45):
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(dem, 1, 1)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    return np.clip(hs, 0, 1)


def load_danger_ratings():
    df = pd.read_csv(DANGER_CSV)
    zone = df[df["zone_name"] == ZONE_NAME].copy()
    zone["date"] = pd.to_datetime(zone["date"])
    return zone.groupby("date")["danger_above_current"].max().to_dict()


def get_danger_for_date(danger_dict, target_date):
    td = pd.Timestamp(target_date)
    if td in danger_dict:
        return int(danger_dict[td])
    dates = sorted(d for d in danger_dict if d <= td)
    if dates:
        return int(danger_dict[dates[-1]])
    return None


def generate_gif(mode, onset_dates_arr, cnn, hillshade, x, y, high_conf, peak_prob,
                 onset_step_idx, onset_days, cnn_times, danger_dict, total_days):
    """Generate one GIF. mode = 'cumulative' or 'rolling'."""

    frame_paths = []
    in_season = [(i, t) for i, t in enumerate(cnn_times) if SEASON_START <= t.to_numpy() <= SEASON_END]

    for frame_num, (t_idx, t_date) in enumerate(in_season):
        t_np = t_date.to_numpy()
        current_days = (t_np - SEASON_START).astype("timedelta64[D]").astype(float)

        if mode == "cumulative":
            det_mask = high_conf & (onset_dates_arr <= t_np)
        else:  # rolling
            window_start = max(t_idx - WINDOW + 1, 0)
            det_mask = high_conf & (onset_step_idx >= window_start) & (onset_step_idx <= t_idx)

        n_det = det_mask.sum()
        cnn_prob = cnn["debris_probability"].sel(time=t_date).values

        figw = 14 if not CROP_LAT else 14
        figh = 8 if not CROP_LAT else 12
        fig, ax = plt.subplots(figsize=(figw, figh), dpi=100)

        # Lightened hillshade
        ax.imshow(hillshade, extent=[x.min(), x.max(), y.min(), y.max()],
                  cmap="gray", vmin=0.0, vmax=0.7, aspect="auto", origin="upper")

        # CNN prob overlay
        prob_rgba = np.zeros((*cnn_prob.shape, 4), dtype=np.float32)
        prob_mask = (~np.isnan(cnn_prob)) & (cnn_prob > 0.3)
        prob_rgba[prob_mask, 0] = 1.0
        prob_rgba[prob_mask, 1] = 0.85
        prob_rgba[prob_mask, 2] = 0.0
        prob_rgba[prob_mask, 3] = np.clip(cnn_prob[prob_mask] * 0.35, 0, 0.35)
        ax.imshow(prob_rgba, extent=[x.min(), x.max(), y.min(), y.max()],
                  aspect="auto", origin="upper")

        # Detections
        det_rgba = np.zeros((*det_mask.shape, 4), dtype=np.float32)
        if det_mask.any():
            if mode == "cumulative":
                # Color by onset date
                cmap = plt.cm.turbo
                norm = mcolors.Normalize(vmin=0, vmax=total_days)
                det_colors = cmap(norm(onset_days[det_mask]))
                det_rgba[det_mask, :3] = det_colors[:, :3]
                det_rgba[det_mask, 3] = 0.85
            else:
                # Color by probability, fade by recency
                probs = np.clip(peak_prob[det_mask], 0.5, 1.0)
                normed = (probs - 0.5) / 0.5
                det_rgba[det_mask, 0] = 1.0
                det_rgba[det_mask, 1] = 0.2 * (1.0 - normed)
                det_rgba[det_mask, 2] = 0.0
                # Recency alpha
                recency = np.zeros_like(onset_step_idx, dtype=np.float32)
                steps = onset_step_idx[det_mask].astype(np.float32)
                window_start_val = max(t_idx - WINDOW + 1, 0)
                if window_start_val < t_idx:
                    recency[det_mask] = 0.4 + 0.6 * (steps - window_start_val) / (t_idx - window_start_val)
                else:
                    recency[det_mask] = 1.0
                det_rgba[det_mask, 3] = recency[det_mask] * 0.9

        ax.imshow(det_rgba, extent=[x.min(), x.max(), y.min(), y.max()],
                  aspect="auto", origin="upper")

        # Danger rating
        danger_level = get_danger_for_date(danger_dict, t_np)
        if danger_level and danger_level in DANGER_LABELS:
            label = DANGER_LABELS[danger_level]
            color = DANGER_COLORS[danger_level]
            ax.text(0.02, 0.97, f"Danger: {label}", transform=ax.transAxes,
                    fontsize=16, fontweight="bold", color=color, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=color, linewidth=2, alpha=0.9))

        date_str = pd.Timestamp(t_np).strftime("%Y-%m-%d")
        mode_label = "Cumulative" if mode == "cumulative" else f"Rolling {WINDOW}-step window"
        ax.set_title(
            f"Banner Summit — Avalanche Detections ({mode_label})\n"
            f"{date_str}  |  {n_det:,} high-confidence pixels",
            fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if mode == "cumulative":
            # Onset date colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.6])
            cmap_cb = plt.cm.turbo
            norm_cb = mcolors.Normalize(vmin=0, vmax=total_days)
            cb = ColorbarBase(cbar_ax, cmap=cmap_cb, norm=norm_cb, orientation="vertical")
            month_days, month_labels = [], []
            for m in range(11, 13):
                d = (np.datetime64(f"2022-{m:02d}-01") - SEASON_START).astype("timedelta64[D]").astype(float)
                month_days.append(d); month_labels.append(f"2022-{m:02d}")
            for m in range(1, 6):
                d = (np.datetime64(f"2023-{m:02d}-01") - SEASON_START).astype("timedelta64[D]").astype(float)
                month_days.append(d); month_labels.append(f"2023-{m:02d}")
            cb.set_ticks(month_days); cb.set_ticklabels(month_labels)
            cb.set_label("Onset date", fontsize=10)

        # Progress bar
        progress = current_days / total_days
        bar_ax = fig.add_axes([0.12, 0.04, 0.75, 0.015])
        bar_ax.barh(0, progress, height=1.0, color="firebrick", edgecolor="black", linewidth=0.5)
        bar_ax.set_xlim(0, 1); bar_ax.set_yticks([]); bar_ax.set_xticks([])
        bar_ax.set_frame_on(True)

        frame_path = OUT_DIR / f"_banner_{mode}_frame_{frame_num:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)
        log.info(f"  [{mode}] Frame {frame_num}: {date_str}, {n_det:,} det, danger={danger_level}")
        sys.stdout.flush()

    # Assemble GIF
    frames = [Image.open(fp) for fp in frame_paths]
    for _ in range(5):
        frames.append(frames[-1])

    suffix = "_inset" if CROP_LAT else ""
    gif_path = OUT_DIR / f"banner_2022_2023_detections_{mode}{suffix}.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=500, loop=0)
    log.info(f"Saved: {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")

    for fp in frame_paths:
        fp.unlink()


def main():
    log.info("Loading data...")
    onset = xr.open_dataset(ONSET_NC)
    cnn = xr.open_dataset(CNN_NC)
    cnn_times = pd.DatetimeIndex(cnn.time.values)

    sar = xr.open_dataset(SAR_NC)
    dem = sar["dem"].values
    hillshade = make_hillshade(dem)

    danger_dict = load_danger_ratings()
    log.info(f"Loaded danger ratings for {len(danger_dict)} dates")

    conf = onset["confidence"].values
    spike = onset["spike_flag"].values
    pre = onset["pre_existing"].values
    valid = ~np.isnan(conf)

    high_conf = valid & (conf >= CONF_THRESH) & ~spike & ~pre
    log.info(f"High confidence detections (>={CONF_THRESH}): {high_conf.sum()}")

    peak_prob = onset["peak_prob"].values
    onset_step_idx = onset["onset_step_idx"].values
    onset_dates = onset["onset_date"].values
    onset_days = (onset_dates - SEASON_START).astype("timedelta64[D]").astype(float)
    total_days = (SEASON_END - SEASON_START).astype("timedelta64[D]").astype(float)

    x = onset.x.values
    y = onset.y.values

    # Crop to inset region if specified
    if CROP_LAT and CROP_LON:
        x_mask = (x >= CROP_LON[0]) & (x <= CROP_LON[1])
        y_mask = (y >= CROP_LAT[0]) & (y <= CROP_LAT[1])
        xi = np.where(x_mask)[0]
        yi = np.where(y_mask)[0]
        x = x[x_mask]
        y = y[y_mask]
        hillshade = hillshade[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
        high_conf = high_conf[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
        peak_prob = peak_prob[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
        onset_step_idx = onset_step_idx[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
        onset_days = onset_days[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
        onset_dates = onset_dates[yi[0]:yi[-1]+1, xi[0]:xi[-1]+1]
        cnn = cnn.isel(y=slice(yi[0], yi[-1]+1), x=slice(xi[0], xi[-1]+1))
        log.info(f"Cropped to lat {CROP_LAT}, lon {CROP_LON}: {len(y)}x{len(x)} pixels")

    # Generate both GIFs
    for mode in ["cumulative", "rolling"]:
        generate_gif(mode, onset_dates, cnn, hillshade, x, y, high_conf, peak_prob,
                     onset_step_idx, onset_days, cnn_times, danger_dict, total_days)

    log.info("Done!")


if __name__ == "__main__":
    main()
