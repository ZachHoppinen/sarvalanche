"""Plot example temporal bump patterns and spatial growth for selected pixels."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import ndimage
from pathlib import Path

OUT_DIR = Path("local/issw/tau_testing")

TAU_FILES = {
    6: OUT_DIR / "tau6" / "season_v2_debris_probabilities.nc",
    12: OUT_DIR / "tau12" / "season_v2_debris_probabilities.nc",
    18: OUT_DIR / "tau18" / "season_v2_debris_probabilities.nc",
}

# High-prob pixels near the target locations
EXAMPLES = {
    "Pt1 -- Sawtooth (43.87, -115.12)": {"y": 1936, "x": 293},
    "Pt2 -- Smoky Mtns (43.79, -114.96)": {"y": 2209, "x": 876},
    "Pt3 -- (44.11, -115.11)": {"y": 1065, "x": 331},
    "Pt4 -- (44.15, -115.00)": {"y": 920, "x": 742},
    "Pt5 -- (44.33, -114.85)": {"y": 265, "x": 1279},
}

SPATIAL_RADIUS = 15  # pixels for spatial context
SIGMA = 3.0  # Gaussian sigma for smoothing


def load_cubes():
    """Load all tau probability cubes."""
    cubes = {}
    for tau, path in TAU_FILES.items():
        ds = xr.open_dataset(path)
        cubes[tau] = {
            "probs": ds["debris_probability"].values,
            "times": pd.DatetimeIndex(ds["time"].values),
        }
        ds.close()
    return cubes


def gaussian_smooth_frame(frame, sigma):
    """NaN-aware Gaussian smoothing of a single (H, W) frame."""
    valid = ~np.isnan(frame)
    filled = np.where(valid, frame, 0.0)
    blurred_v = ndimage.gaussian_filter(filled, sigma=sigma, mode="constant", cval=0)
    blurred_w = ndimage.gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="constant", cval=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(blurred_w > 1e-10, blurred_v / blurred_w, 0.0)


def plot_temporal_bumps(cubes, examples, out_path):
    """Plot time series of pixel prob + Gaussian-smoothed neighborhood for each tau."""
    n_pts = len(examples)
    n_taus = len(cubes)

    fig, axes = plt.subplots(n_pts, n_taus, figsize=(5 * n_taus, 4 * n_pts), squeeze=False)
    fig.suptitle("Temporal Probability Bumps: Pixel vs Neighborhood", fontsize=14, fontweight="bold", y=1.02)

    for row, (label, pix) in enumerate(examples.items()):
        yi, xi = pix["y"], pix["x"]

        for col, (tau, data) in enumerate(sorted(cubes.items())):
            ax = axes[row, col]
            probs = data["probs"]
            times = data["times"]
            T = len(times)

            # Pixel time series
            pixel_ts = probs[:, yi, xi]

            # Average of top-5 neighbors (within SPATIAL_RADIUS)
            r = SPATIAL_RADIUS
            y_lo = max(0, yi - r)
            y_hi = min(probs.shape[1], yi + r + 1)
            x_lo = max(0, xi - r)
            x_hi = min(probs.shape[2], xi + r + 1)

            patch_ts = probs[:, y_lo:y_hi, x_lo:x_hi]
            # Neighborhood mean
            neigh_mean = np.nanmean(patch_ts, axis=(1, 2))

            # Gaussian-smoothed at this pixel
            gauss_ts = np.zeros(T, dtype=np.float32)
            for t in range(T):
                smoothed = gaussian_smooth_frame(probs[t], SIGMA)
                gauss_ts[t] = smoothed[yi, xi]

            dates = times.to_pydatetime()
            ax.plot(dates, pixel_ts, "o-", markersize=3, linewidth=1.5, label="This pixel", color="C0")
            ax.plot(dates, gauss_ts, "s-", markersize=2, linewidth=1.5, label=f"Gaussian (s={SIGMA})", color="C1")
            ax.plot(dates, neigh_mean, "^-", markersize=2, linewidth=1, label="Neighborhood mean", color="C2", alpha=0.7)

            # Mark peak
            pk = int(np.nanargmax(pixel_ts))
            ax.axvline(dates[pk], color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.annotate(f"peak: {dates[pk].strftime('%m-%d')}", xy=(dates[pk], pixel_ts[pk]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7, color="red")

            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"tau={tau}", fontsize=11)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.2)

            if col == 0:
                ax.set_ylabel(f"{label}\nProbability", fontsize=9)
            if row == 0 and col == n_taus - 1:
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spatial_evolution(cubes, examples, out_dir, steps_around_peak=4):
    """Plot spatial probability maps at +-N steps around peak for each tau."""
    for label, pix in examples.items():
        yi, xi = pix["y"], pix["x"]
        safe_label = label.split("--")[0].strip().replace(" ", "")

        for tau, data in sorted(cubes.items()):
            probs = data["probs"]
            times = data["times"]

            pixel_ts = probs[:, yi, xi]
            pk = int(np.nanargmax(pixel_ts))

            # Steps to show: pk-N to pk+N
            t_indices = list(range(max(0, pk - steps_around_peak), min(len(times), pk + steps_around_peak + 1)))
            n_frames = len(t_indices)

            r = SPATIAL_RADIUS
            y_lo = max(0, yi - r)
            y_hi = min(probs.shape[1], yi + r + 1)
            x_lo = max(0, xi - r)
            x_hi = min(probs.shape[2], xi + r + 1)

            fig, axes = plt.subplots(2, n_frames, figsize=(2.8 * n_frames, 5.5),
                                     gridspec_kw={"height_ratios": [1, 1]})
            if n_frames == 1:
                axes = axes.reshape(2, 1)

            fig.suptitle(f"{label} -- tau={tau}  (peak at {times[pk].strftime('%Y-%m-%d')})",
                         fontsize=12, fontweight="bold")

            # Row 1: raw probability patches
            vmin, vmax = 0, 1
            for i, t in enumerate(t_indices):
                ax = axes[0, i]
                patch = probs[t, y_lo:y_hi, x_lo:x_hi]
                im = ax.imshow(patch, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", aspect="equal")

                # Mark center pixel
                cy = yi - y_lo
                cx = xi - x_lo
                ax.plot(cx, cy, "k+", markersize=8, markeredgewidth=1.5)

                date_str = times[t].strftime("%m-%d")
                offset = t - pk
                sign = "+" if offset > 0 else ""
                ax.set_title(f"{date_str}\n({sign}{offset})", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

                if t == pk:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(2)

            axes[0, 0].set_ylabel("Raw probability", fontsize=9)

            # Row 2: Gaussian-smoothed patches
            for i, t in enumerate(t_indices):
                ax = axes[1, i]
                smoothed_full = gaussian_smooth_frame(probs[t], SIGMA)
                patch_smooth = smoothed_full[y_lo:y_hi, x_lo:x_hi]
                im = ax.imshow(patch_smooth, vmin=vmin, vmax=vmax, cmap="RdYlBu_r", aspect="equal")

                cy = yi - y_lo
                cx = xi - x_lo
                ax.plot(cx, cy, "k+", markersize=8, markeredgewidth=1.5)
                ax.set_xticks([])
                ax.set_yticks([])

                if t == pk:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(2)

            axes[1, 0].set_ylabel(f"Gaussian s={SIGMA}", fontsize=9)

            # Colorbar
            fig.subplots_adjust(right=0.92)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
            fig.colorbar(im, cax=cbar_ax, label="Probability")

            plt.tight_layout(rect=[0, 0, 0.92, 0.95])
            fname = out_dir / f"spatial_evolution_{safe_label}_tau{tau}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fname}")


if __name__ == "__main__":
    cubes = load_cubes()
    plot_temporal_bumps(cubes, EXAMPLES, OUT_DIR / "bump_examples.png")
    plot_spatial_evolution(cubes, EXAMPLES, OUT_DIR)
