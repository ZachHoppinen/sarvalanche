"""
Animated GIF showing high-confidence avalanche detections at each time step only
(no accumulation) over the 2024-2025 season in the Sawtooth zone.

Each frame shows:
  - Background: lightened hillshade from DEM
  - Colored overlay: high-confidence detections whose onset_date matches the current step
  - Color = CNN probability intensity (hot colormap)
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
NC_DIR = BASE / "netcdfs" / "Sawtooth_&_Western_Smoky_Mtns"
ONSET_NC = NC_DIR / "v2_season_inference" / "temporal_onset.nc"
CNN_NC = NC_DIR / "v2_season_inference" / "season_v2_debris_probabilities.nc"
SAR_NC = NC_DIR / "season_2024_2025_Sawtooth_&_Western_Smoky_Mtns.nc"
OUT_DIR = BASE / "figures" / "gif"

sys.path.insert(0, str(Path("/Users/zmhoppinen/Documents/sarvalanche/src")))


def make_hillshade(dem, azimuth=315, altitude=45):
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(dem, 1, 1)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    hs = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    return np.clip(hs, 0, 1)


def main():
    log.info("Loading data...")
    onset = xr.open_dataset(ONSET_NC)
    cnn = xr.open_dataset(CNN_NC)
    cnn_times = pd.DatetimeIndex(cnn.time.values)

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

    onset_dates = onset["onset_date"].values
    peak_prob = onset["peak_prob"].values

    season_start = np.datetime64("2024-11-01")
    season_end = np.datetime64("2025-05-01")
    total_days = (season_end - season_start).astype("timedelta64[D]").astype(float)

    x = onset.x.values
    y = onset.y.values

    # Build a mapping: for each CNN time step, which pixels have onset at that step?
    # Match onset_date to nearest CNN time step
    onset_step_idx = onset["onset_step_idx"].values

    frame_paths = []
    for t_idx, t_date in enumerate(cnn_times):
        t_np = t_date.to_numpy()
        if t_np < season_start or t_np > season_end:
            continue

        current_days = (t_np - season_start).astype("timedelta64[D]").astype(float)

        # Detections whose onset matches this time step
        current_detections = high_conf & (onset_step_idx == t_idx)
        n_det = current_detections.sum()

        # Current CNN probability
        cnn_prob = cnn["debris_probability"].sel(time=t_date).values

        fig, ax = plt.subplots(figsize=(10, 14), dpi=100)

        # Lightened hillshade background
        ax.imshow(
            hillshade,
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="gray",
            vmin=0.0, vmax=0.7,
            aspect="auto",
            origin="upper",
        )

        # Current CNN probability as semi-transparent warm overlay
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

        # Current-step high-confidence detections in bright red/orange
        det_rgba = np.zeros((*current_detections.shape, 4), dtype=np.float32)
        if current_detections.any():
            # Color by peak probability intensity
            probs = np.clip(peak_prob[current_detections], 0.5, 1.0)
            normed = (probs - 0.5) / 0.5  # 0-1
            det_rgba[current_detections, 0] = 1.0                    # R
            det_rgba[current_detections, 1] = 0.2 * (1.0 - normed)  # G fades with intensity
            det_rgba[current_detections, 2] = 0.0                    # B
            det_rgba[current_detections, 3] = 0.9

        ax.imshow(
            det_rgba,
            extent=[x.min(), x.max(), y.min(), y.max()],
            aspect="auto",
            origin="upper",
        )

        date_str = pd.Timestamp(t_np).strftime("%Y-%m-%d")
        ax.set_title(
            f"Sawtooth Zone — New Avalanche Detections\n{date_str}  |  {n_det:,} new high-confidence pixels",
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

        frame_path = OUT_DIR / f"_gif_current_frame_{t_idx:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)
        log.info(f"  Frame {t_idx}: {date_str}, {n_det:,} new detections")
        sys.stdout.flush()

    log.info(f"Assembling GIF from {len(frame_paths)} frames...")
    frames = [Image.open(fp) for fp in frame_paths]
    for _ in range(5):
        frames.append(frames[-1])

    gif_path = OUT_DIR / "sawtooth_2024_2025_detections_current_only.gif"
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
