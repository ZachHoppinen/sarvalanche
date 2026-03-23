"""
Animated GIF showing high-confidence avalanche detections in a rolling 5-step
window over the 2024-2025 season in the Sawtooth zone, with SNFAC danger rating.

Each frame shows:
  - Background: lightened hillshade from DEM
  - Colored overlay: high-confidence detections from the current and previous 4 steps
  - Danger rating label (HIGH/CONSIDERABLE/MODERATE/LOW) in upper-left
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
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/issw")
NC_DIR = BASE / "netcdfs" / "Sawtooth_&_Western_Smoky_Mtns"
ONSET_NC = NC_DIR / "v2_season_inference" / "temporal_onset.nc"
CNN_NC = NC_DIR / "v2_season_inference" / "season_v2_debris_probabilities.nc"
SAR_NC = NC_DIR / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
DANGER_CSV = BASE / "high_low_framing_outputs" / "v1_high_danger_output" / "snfac_dangers_2021_2025.csv"
OUT_DIR = BASE / "figures" / "gif"

sys.path.insert(0, str(Path("/Users/zmhoppinen/Documents/sarvalanche/src")))

WINDOW = 3  # rolling window in time steps

# Crop bounds (set to None for full extent)
CROP_LAT = (43.8, 44.1)
CROP_LON = (-115.15, -114.9)

# Danger rating: 1=Low, 2=Moderate, 3=Considerable, 4=High, 5=Extreme
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
    """Load SNFAC danger ratings for Sawtooth zone, one max-above-treeline value per date."""
    df = pd.read_csv(DANGER_CSV)
    saw = df[df["zone_name"] == "Sawtooth & Western Smoky Mtns"].copy()
    saw["date"] = pd.to_datetime(saw["date"])
    # Take max danger_above_current per date (handles duplicate rows)
    daily = saw.groupby("date")["danger_above_current"].max().to_dict()
    return daily


def get_danger_for_date(danger_dict, target_date):
    """Get danger rating for date, falling back to nearest previous date."""
    td = pd.Timestamp(target_date)
    # Try exact match first
    if td in danger_dict:
        return int(danger_dict[td])
    # Fall back to most recent prior date
    dates = sorted(d for d in danger_dict if d <= td)
    if dates:
        return int(danger_dict[dates[-1]])
    return None


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

    # High-confidence detection mask
    conf = onset["confidence"].values
    spike = onset["spike_flag"].values
    pre = onset["pre_existing"].values
    valid = ~np.isnan(conf)

    CONF_THRESH = 0.6
    high_conf = valid & (conf >= CONF_THRESH) & ~spike & ~pre
    log.info(f"High confidence detections (>={CONF_THRESH}): {high_conf.sum()}")

    peak_prob = onset["peak_prob"].values
    onset_step_idx = onset["onset_step_idx"].values

    season_start = np.datetime64("2024-11-01")
    season_end = np.datetime64("2025-05-01")
    total_days = (season_end - season_start).astype("timedelta64[D]").astype(float)

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
        # Slice CNN probs too
        cnn = cnn.isel(y=slice(yi[0], yi[-1]+1), x=slice(xi[0], xi[-1]+1))
        log.info(f"Cropped to lat {CROP_LAT}, lon {CROP_LON}: {len(y)}x{len(x)} pixels")

    # Filter to in-season frames
    in_season = [(i, t) for i, t in enumerate(cnn_times) if season_start <= t.to_numpy() <= season_end]

    frame_paths = []
    for frame_num, (t_idx, t_date) in enumerate(in_season):
        t_np = t_date.to_numpy()
        current_days = (t_np - season_start).astype("timedelta64[D]").astype(float)

        # Rolling window: detections from steps [t_idx - WINDOW + 1, t_idx]
        window_start = max(t_idx - WINDOW + 1, 0)
        window_mask = high_conf & (onset_step_idx >= window_start) & (onset_step_idx <= t_idx)
        n_det = window_mask.sum()

        # Recency within window for alpha: most recent = brightest
        recency = np.zeros_like(onset_step_idx, dtype=np.float32)
        if window_mask.any():
            steps = onset_step_idx[window_mask].astype(np.float32)
            # Map to [0.4, 1.0]: oldest in window = 0.4, newest = 1.0
            if window_start < t_idx:
                recency[window_mask] = 0.4 + 0.6 * (steps - window_start) / (t_idx - window_start)
            else:
                recency[window_mask] = 1.0

        figw = 14 if CROP_LAT else 10
        figh = 12 if CROP_LAT else 14
        fig, ax = plt.subplots(figsize=(figw, figh), dpi=100)

        # Lightened hillshade
        ax.imshow(
            hillshade,
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="gray",
            vmin=0.0, vmax=0.7,
            aspect="auto",
            origin="upper",
        )

        # Current CNN prob overlay
        cnn_prob = cnn["debris_probability"].sel(time=t_date).values
        prob_rgba = np.zeros((*cnn_prob.shape, 4), dtype=np.float32)
        prob_mask = (~np.isnan(cnn_prob)) & (cnn_prob > 0.3)
        prob_rgba[prob_mask, 0] = 1.0
        prob_rgba[prob_mask, 1] = 0.85
        prob_rgba[prob_mask, 2] = 0.0
        prob_rgba[prob_mask, 3] = np.clip(cnn_prob[prob_mask] * 0.35, 0, 0.35)

        ax.imshow(
            prob_rgba,
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            origin="upper",
        )

        # Rolling-window detections: red, alpha scaled by recency
        det_rgba = np.zeros((*window_mask.shape, 4), dtype=np.float32)
        if window_mask.any():
            probs = np.clip(peak_prob[window_mask], 0.5, 1.0)
            normed = (probs - 0.5) / 0.5
            det_rgba[window_mask, 0] = 1.0
            det_rgba[window_mask, 1] = 0.2 * (1.0 - normed)
            det_rgba[window_mask, 2] = 0.0
            det_rgba[window_mask, 3] = recency[window_mask] * 0.9

        ax.imshow(
            det_rgba,
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            origin="upper",
        )

        # Danger rating label
        danger_level = get_danger_for_date(danger_dict, t_np)
        if danger_level is not None and danger_level in DANGER_LABELS:
            label = DANGER_LABELS[danger_level]
            color = DANGER_COLORS[danger_level]
            ax.text(
                0.02, 0.97, f"Danger: {label}",
                transform=ax.transAxes,
                fontsize=16, fontweight="bold",
                color=color,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, linewidth=2, alpha=0.9),
            )

        date_str = pd.Timestamp(t_np).strftime("%Y-%m-%d")
        ax.set_title(
            f"Sawtooth Zone — Avalanche Detections (rolling {WINDOW}-step window)\n"
            f"{date_str}  |  {n_det:,} high-confidence pixels",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Progress bar
        progress = current_days / total_days
        bar_ax = fig.add_axes([0.12, 0.04, 0.75, 0.015])
        bar_ax.barh(0, progress, height=1.0, color="firebrick", edgecolor="black", linewidth=0.5)
        bar_ax.set_xlim(0, 1)
        bar_ax.set_yticks([])
        bar_ax.set_xticks([])
        bar_ax.set_frame_on(True)

        frame_path = OUT_DIR / f"_gif_rolling_frame_{frame_num:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)
        log.info(f"  Frame {frame_num}: {date_str}, {n_det:,} detections, danger={danger_level}")
        sys.stdout.flush()

    log.info(f"Assembling GIF from {len(frame_paths)} frames...")
    frames = [Image.open(fp) for fp in frame_paths]
    for _ in range(5):
        frames.append(frames[-1])

    suffix = "_inset" if CROP_LAT else ""
    gif_path = OUT_DIR / f"sawtooth_2024_2025_detections_rolling{suffix}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
    )
    log.info(f"Saved GIF: {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")

    for fp in frame_paths:
        fp.unlink()

    log.info("Done!")


if __name__ == "__main__":
    main()
