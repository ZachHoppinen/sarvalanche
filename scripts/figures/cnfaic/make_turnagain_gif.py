"""
Animated GIF showing high-confidence avalanche detections in a rolling window
over the 2024-2025 season for Turnagain Pass, with danger rating and
observed avalanche markers.

Each frame shows:
  - Background: lightened hillshade from DEM
  - Yellow overlay: current CNN probability
  - Red overlay: high-confidence detections (rolling window)
  - Triangles: observed avalanches within ±3 days of current step, sized by D-size
  - Danger rating label in upper-left
"""

import ast
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
from matplotlib.lines import Line2D
from PIL import Image

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic")
NC_DIR = BASE / "netcdfs" / "Turnagain_Pass_and_Girdwood"
ONSET_NC = NC_DIR / "v2_season_inference_2024-2025" / "temporal_onset.nc"
CNN_NC = NC_DIR / "v2_season_inference_2024-2025" / "season_v2_debris_probabilities.nc"
SAR_NC = NC_DIR / "season_2024-2025_Turnagain_Pass_and_Girdwood.nc"
DANGER_CSV = BASE / "cnfaic_dangers_2020_2025.csv"
OBS_CSV = BASE / "cnfaic_obs_2020_2025.csv"
OUT_DIR = BASE / "figures" / "gif"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 3  # rolling window in time steps
OBS_WINDOW_DAYS = 3  # show obs within ±N days of current step

# Danger rating
DANGER_LABELS = {1: "LOW", 2: "MODERATE", 3: "CONSIDERABLE", 4: "HIGH", 5: "EXTREME"}
DANGER_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#1a1a1a"}

# D-size marker sizing and colors
DSIZE_SIZES = {1.0: 40, 1.5: 60, 2.0: 90, 2.5: 130, 3.0: 180, 3.5: 220, 4.0: 280}
DSIZE_COLORS = {
    1.0: "#64B5F6", 1.5: "#42A5F5", 2.0: "#FFA726", 2.5: "#FF7043",
    3.0: "#E53935", 3.5: "#B71C1C", 4.0: "#7B1FA2",
}


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
    tp = df[df["zone_name"] == "Turnagain Pass and Girdwood"].copy()
    tp["date"] = pd.to_datetime(tp["date"])
    return tp.groupby("date")["danger_above_current"].max().to_dict()


def get_danger_for_date(danger_dict, target_date):
    td = pd.Timestamp(target_date)
    if td in danger_dict:
        return int(danger_dict[td])
    dates = sorted(d for d in danger_dict if d <= td)
    if dates:
        return int(danger_dict[dates[-1]])
    return None


def load_observations():
    """Load Turnagain obs with parsed lat/lng."""
    df = pd.read_csv(OBS_CSV)
    df = df[df["zone_name"] == "Turnagain Pass and Girdwood"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2024-10-01") & (df["date"] <= "2025-05-01")]

    lats, lngs = [], []
    for pt_str in df["location_point"]:
        try:
            pt = ast.literal_eval(str(pt_str))
            lats.append(pt["lat"])
            lngs.append(pt["lng"])
        except Exception:
            lats.append(np.nan)
            lngs.append(np.nan)
    df["lat"] = lats
    df["lng"] = lngs
    df = df.dropna(subset=["lat", "lng"])
    return df


def main():
    log.info("Loading data...")
    onset = xr.open_dataset(ONSET_NC)
    cnn = xr.open_dataset(CNN_NC)
    cnn_times = pd.DatetimeIndex(cnn.time.values)

    sar = xr.open_dataset(SAR_NC)
    dem = sar["dem"].values
    hillshade = make_hillshade(dem)
    sar.close()

    danger_dict = load_danger_ratings()
    obs_df = load_observations()
    log.info(f"Loaded {len(obs_df)} observations, danger for {len(danger_dict)} dates")

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

    # Filter to in-season frames
    in_season = [(i, t) for i, t in enumerate(cnn_times) if season_start <= t.to_numpy() <= season_end]

    frame_paths = []
    for frame_num, (t_idx, t_date) in enumerate(in_season):
        t_np = t_date.to_numpy()
        current_days = (t_np - season_start).astype("timedelta64[D]").astype(float)

        # Rolling window detections
        window_start = max(t_idx - WINDOW + 1, 0)
        window_mask = high_conf & (onset_step_idx >= window_start) & (onset_step_idx <= t_idx)
        n_det = window_mask.sum()

        # Recency alpha
        recency = np.zeros_like(onset_step_idx, dtype=np.float32)
        if window_mask.any():
            steps = onset_step_idx[window_mask].astype(np.float32)
            if window_start < t_idx:
                recency[window_mask] = 0.4 + 0.6 * (steps - window_start) / (t_idx - window_start)
            else:
                recency[window_mask] = 1.0

        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)

        # Hillshade background
        ax.imshow(
            hillshade,
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="gray", vmin=0.0, vmax=0.7,
            aspect="auto", origin="upper",
        )

        # Current CNN prob overlay (yellow)
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
            aspect="auto", origin="upper",
        )

        # Rolling-window detections (red)
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
            aspect="auto", origin="upper",
        )

        # Observation markers within ±OBS_WINDOW_DAYS
        t_pd = pd.Timestamp(t_np)
        nearby_obs = obs_df[
            (obs_df["date"] >= t_pd - pd.Timedelta(days=OBS_WINDOW_DAYS)) &
            (obs_df["date"] <= t_pd + pd.Timedelta(days=OBS_WINDOW_DAYS))
        ]
        if len(nearby_obs) > 0:
            for _, ob in nearby_obs.iterrows():
                ds = ob["d_size"] if not pd.isna(ob["d_size"]) else 1.5
                closest_ds = min(DSIZE_SIZES.keys(), key=lambda k: abs(k - ds))
                sz = DSIZE_SIZES[closest_ds]
                clr = DSIZE_COLORS[closest_ds]
                ax.scatter(
                    ob["lng"], ob["lat"],
                    s=sz, c=clr, marker="^",
                    edgecolors="white", linewidths=1.2,
                    zorder=10,
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
                color=color, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, linewidth=2, alpha=0.9),
            )

        # Obs count label
        n_obs = len(nearby_obs)
        ax.text(
            0.02, 0.90,
            f"Obs (±{OBS_WINDOW_DAYS}d): {n_obs}",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            color="#333", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#999", linewidth=1, alpha=0.9),
        )

        date_str = pd.Timestamp(t_np).strftime("%Y-%m-%d")
        ax.set_title(
            f"Turnagain Pass — Avalanche Detections (rolling {WINDOW}-step window)\n"
            f"{date_str}  |  {n_det:,} high-confidence pixels",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # D-size legend
        legend_elements = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor=DSIZE_COLORS[d],
                   markersize=8 + d * 2, markeredgecolor="white", markeredgewidth=0.8,
                   label=f"D{d:.0f}" if d == int(d) else f"D{d}")
            for d in [1.0, 2.0, 3.0, 4.0]
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  title="Obs D-size", title_fontsize=10, framealpha=0.9)

        # Progress bar
        progress = current_days / total_days
        bar_ax = fig.add_axes([0.12, 0.04, 0.75, 0.015])
        bar_ax.barh(0, progress, height=1.0, color="firebrick", edgecolor="black", linewidth=0.5)
        bar_ax.set_xlim(0, 1)
        bar_ax.set_yticks([])
        bar_ax.set_xticks([])
        bar_ax.set_frame_on(True)

        frame_path = OUT_DIR / f"_turnagain_frame_{frame_num:03d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)
        log.info(f"  Frame {frame_num}: {date_str}, {n_det:,} det, {n_obs} obs, danger={danger_level}")
        sys.stdout.flush()

    log.info(f"Assembling GIF from {len(frame_paths)} frames...")
    frames = [Image.open(fp) for fp in frame_paths]
    # Hold last frame longer
    for _ in range(5):
        frames.append(frames[-1])

    gif_path = OUT_DIR / "turnagain_2024_2025_detections_rolling.gif"
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
