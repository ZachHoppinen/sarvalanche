"""
Create an animated GIF showing high-confidence avalanche detections accumulating
over the 2024-2025 season in the Sawtooth zone.

Each frame shows:
  - Background: hillshade from DEM
  - Colored overlay: high-confidence detections that have onset_date <= current frame date
  - Color = onset date (colormap across season)
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
import matplotlib.dates as mdates
from matplotlib.colorbar import ColorbarBase
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/issw")
NC_DIR = BASE / "netcdfs" / "Sawtooth_&_Western_Smoky_Mtns"
ONSET_NC = NC_DIR / "v2_season_inference" / "temporal_onset.nc"
CNN_NC = NC_DIR / "v2_season_inference" / "season_v2_debris_probabilities.nc"
SAR_NC = NC_DIR / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
OUT_DIR = BASE / "figures" / "gif"

sys.path.insert(0, str(Path("/Users/zmhoppinen/Documents/sarvalanche/src")))


def make_hillshade(dem, azimuth=315, altitude=45):
    """Simple hillshade from DEM array."""
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(dem, 1, 1)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    return np.clip(hs, 0, 1)


def main():
    # Load data
    log.info("Loading temporal onset data...")
    onset = xr.open_dataset(ONSET_NC)

    log.info("Loading CNN probabilities...")
    cnn = xr.open_dataset(CNN_NC)
    cnn_times = pd.DatetimeIndex(cnn.time.values)

    log.info("Loading SAR dataset for DEM...")
    sar = xr.open_dataset(SAR_NC)
    dem = sar["dem"].values
    hillshade = make_hillshade(dem)

    # High-confidence detection mask
    conf = onset["confidence"].values
    spike = onset["spike_flag"].values
    pre = onset["pre_existing"].values
    valid = ~np.isnan(conf)

    CONF_THRESH = 0.6
    high_conf = valid & (conf >= CONF_THRESH) & ~spike & ~pre
    log.info(f"High confidence detections (>={CONF_THRESH}): {high_conf.sum()}")

    # Onset dates as numeric (days since season start)
    onset_dates = onset["onset_date"].values  # datetime64
    season_start = np.datetime64("2024-11-01")
    season_end = np.datetime64("2025-05-01")

    # Convert onset dates to float days for colormap
    onset_days = (onset_dates - season_start).astype("timedelta64[D]").astype(float)
    total_days = (season_end - season_start).astype("timedelta64[D]").astype(float)

    # Colormap for onset dates
    cmap = plt.cm.turbo
    norm = mcolors.Normalize(vmin=0, vmax=total_days)

    # Coordinates
    x = onset.x.values
    y = onset.y.values

    # Generate frames
    frame_paths = []
    for t_idx, t_date in enumerate(cnn_times):
        t_np = t_date.to_numpy()
        if t_np < season_start or t_np > season_end:
            continue

        current_days = (t_np - season_start).astype("timedelta64[D]").astype(float)

        # Detections with onset <= current date
        detected_so_far = high_conf & (onset_dates <= t_np)

        # Current CNN probability frame
        cnn_prob = cnn["debris_probability"].sel(time=t_date).values

        fig, ax = plt.subplots(figsize=(10, 14), dpi=100)

        # Background: hillshade
        ax.imshow(
            hillshade,
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="gray",
            vmin=0.2, vmax=1.0,
            aspect="auto",
            origin="upper",
        )

        # Current CNN probability as semi-transparent yellow
        prob_rgba = np.zeros((*cnn_prob.shape, 4), dtype=np.float32)
        prob_mask = (~np.isnan(cnn_prob)) & (cnn_prob > 0.3)
        prob_rgba[prob_mask, 0] = 1.0  # R
        prob_rgba[prob_mask, 1] = 1.0  # G
        prob_rgba[prob_mask, 2] = 0.0  # B
        prob_rgba[prob_mask, 3] = np.clip(cnn_prob[prob_mask] * 0.4, 0, 0.4)  # alpha

        ax.imshow(
            prob_rgba,
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            origin="upper",
        )

        # Accumulated high-confidence detections colored by onset date
        det_rgba = np.zeros((*detected_so_far.shape, 4), dtype=np.float32)
        if detected_so_far.any():
            det_colors = cmap(norm(onset_days[detected_so_far]))
            det_rgba[detected_so_far, :3] = det_colors[:, :3]
            det_rgba[detected_so_far, 3] = 0.85

        ax.imshow(
            det_rgba,
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            origin="upper",
        )

        # Title and date
        date_str = pd.Timestamp(t_np).strftime("%Y-%m-%d")
        n_det = detected_so_far.sum()
        ax.set_title(
            f"Sawtooth Zone — Avalanche Detections\n{date_str}  |  {n_det:,} high-confidence pixels",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Colorbar for onset date
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
        cb = ColorbarBase(
            cbar_ax, cmap=cmap, norm=norm,
            orientation="vertical",
        )
        # Label with month ticks
        month_days = []
        month_labels = []
        for m in range(11, 13):
            d = (np.datetime64(f"2024-{m:02d}-01") - season_start).astype("timedelta64[D]").astype(float)
            month_days.append(d)
            month_labels.append(f"2024-{m:02d}")
        for m in range(1, 6):
            d = (np.datetime64(f"2025-{m:02d}-01") - season_start).astype("timedelta64[D]").astype(float)
            month_days.append(d)
            month_labels.append(f"2025-{m:02d}")
        cb.set_ticks(month_days)
        cb.set_ticklabels(month_labels)
        cb.set_label("Onset date", fontsize=10)

        # Progress bar
        progress = current_days / total_days
        bar_ax = fig.add_axes([0.12, 0.04, 0.75, 0.015])
        bar_ax.barh(0, progress, height=1.0, color=cmap(norm(current_days)), edgecolor="black", linewidth=0.5)
        bar_ax.set_xlim(0, 1)
        bar_ax.set_yticks([])
        bar_ax.set_xticks([])
        bar_ax.set_frame_on(True)

        frame_path = OUT_DIR / f"_gif_frame_{t_idx:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)
        log.info(f"  Frame {t_idx}: {date_str}, {n_det:,} detections")
        sys.stdout.flush()

    # Assemble GIF
    log.info(f"Assembling GIF from {len(frame_paths)} frames...")
    frames = [Image.open(fp) for fp in frame_paths]

    # Add extra frames at the end to pause
    for _ in range(5):
        frames.append(frames[-1])

    gif_path = OUT_DIR / "sawtooth_2024_2025_detections.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=400,  # ms per frame
        loop=0,
    )
    log.info(f"Saved GIF: {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")

    # Cleanup temp frames
    for fp in frame_paths:
        fp.unlink()

    log.info("Done!")


if __name__ == "__main__":
    main()
