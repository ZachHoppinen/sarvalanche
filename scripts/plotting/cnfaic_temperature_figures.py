"""Generate HRRR temperature visualization figures for CNFAIC area."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import pandas as pd

FIGDIR = Path("local/cnfaic/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# Load data
hrrr = xr.open_dataset("local/cnfaic/hrrr_temperature_test.nc")
sar_ds = xr.open_dataset(
    "local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/"
    "season_2025-2026_Turnagain_Pass_and_Girdwood.nc"
)
dem = sar_ds["dem"].values
if dem.ndim == 3:
    dem = dem[0]

# Coordinate extent for imshow
x = hrrr.x.values
y = hrrr.y.values
extent = [x.min(), x.max(), y.min(), y.max()]

# CNN detection counts (human_only >0.5)
cnn_counts = {
    pd.Timestamp("2025-11-09"): 35729,
    pd.Timestamp("2025-11-21"): 17804,
    pd.Timestamp("2025-12-03"): 12071,
    pd.Timestamp("2025-12-15"): 127189,
    pd.Timestamp("2025-12-27"): 10254,
    pd.Timestamp("2026-01-08"): 2452,
    pd.Timestamp("2026-02-01"): 4516,
    pd.Timestamp("2026-02-13"): 107737,
    pd.Timestamp("2026-02-25"): 89365,
    pd.Timestamp("2026-03-09"): 2290,
}

# ─────────────────────────────────────────────────────────────────────
# Figure 1: Season time series
# ─────────────────────────────────────────────────────────────────────
print("Generating Figure 1: hrrr_season_timeseries.png")

t2m_max_spatial_max = hrrr["t2m_max"].max(dim=["y", "x"]).values
t2m_mean_spatial_mean = hrrr["t2m_mean"].mean(dim=["y", "x"]).values
t2m_mean_spatial_min = hrrr["t2m_mean"].min(dim=["y", "x"]).values
times = hrrr.time.values

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(times, t2m_max_spatial_max, "r-o", label="Spatial max of T2m_max", ms=4)
ax1.plot(times, t2m_mean_spatial_mean, "k-o", label="Spatial mean of T2m_mean", ms=4)
ax1.plot(times, t2m_mean_spatial_min, "b-o", label="Spatial min of T2m_mean", ms=4)
ax1.axhline(0, color="gray", ls="--", lw=1, label="0 C (freezing)")

# Shade above-freezing periods
for i in range(len(times) - 1):
    if t2m_mean_spatial_mean[i] > 0:
        ax1.axvspan(times[i], times[min(i + 1, len(times) - 1)],
                    alpha=0.15, color="red")

ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature (C)")
ax1.legend(loc="upper left", fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)

# Secondary y-axis for CNN counts
ax2 = ax1.twinx()
cnn_dates = list(cnn_counts.keys())
cnn_vals = list(cnn_counts.values())
# Only plot counts whose dates are in the temperature data
hrrr_dates_set = set(pd.to_datetime(times).normalize())
mask = [d in hrrr_dates_set for d in cnn_dates]
cnn_dates_plot = [d for d, m in zip(cnn_dates, mask) if m]
cnn_vals_plot = [v for v, m in zip(cnn_vals, mask) if m]

ax2.bar(cnn_dates_plot, cnn_vals_plot, width=1.5, alpha=0.3, color="purple",
        label="CNN detection pixels")
ax2.set_ylabel("CNN Detection Pixel Count", color="purple")
ax2.tick_params(axis="y", labelcolor="purple")
ax2.legend(loc="upper right", fontsize=9)

ax1.set_title("HRRR-AK 2m Temperature vs CNN Detection Volume")
fig.tight_layout()
fig.savefig(FIGDIR / "hrrr_season_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────
# Figures 2-5: Side-by-side DEM + temperature maps
# ─────────────────────────────────────────────────────────────────────
map_dates = ["2025-12-03", "2025-12-15", "2026-02-03", "2026-02-13"]

for date_str in map_dates:
    print(f"Generating Figure: hrrr_temp_{date_str}.png")
    t2m_max_slice = hrrr["t2m_max"].sel(time=date_str, method="nearest")
    t2m_mean_slice = hrrr["t2m_mean"].sel(time=date_str, method="nearest")
    actual_date = pd.Timestamp(t2m_max_slice.time.values).strftime("%Y-%m-%d")

    tmax_data = t2m_max_slice.values
    tmean_data = t2m_mean_slice.values

    stats = (
        f"Tmax: [{np.nanmin(tmax_data):.1f}, {np.nanmax(tmax_data):.1f}] C  "
        f"Tmean: [{np.nanmin(tmean_data):.1f}, {np.nanmax(tmean_data):.1f}] C"
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # DEM
    ax = axes[0]
    im0 = ax.imshow(dem, extent=extent, origin="lower", cmap="gist_earth",
                     aspect="auto")
    # Contours every 200m
    Y, X = np.meshgrid(y, x, indexing="ij")
    levels = np.arange(0, np.nanmax(dem) + 200, 200)
    levels = levels[levels >= np.nanmin(dem[~np.isnan(dem)])]
    ax.contour(X, Y, dem, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    ax.set_title("DEM (m)")
    plt.colorbar(im0, ax=ax, shrink=0.8)

    # t2m_max
    norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=10)
    ax = axes[1]
    im1 = ax.imshow(tmax_data, extent=extent, origin="lower",
                     cmap="RdBu_r", norm=norm, aspect="auto")
    ax.contour(X, Y, tmax_data, levels=[0], colors="k", linewidths=1.5)
    ax.set_title(f"t2m_max ({actual_date})")
    plt.colorbar(im1, ax=ax, shrink=0.8, label="C")

    # t2m_mean
    ax = axes[2]
    im2 = ax.imshow(tmean_data, extent=extent, origin="lower",
                     cmap="RdBu_r", norm=norm, aspect="auto")
    ax.contour(X, Y, tmean_data, levels=[0], colors="k", linewidths=1.5)
    ax.set_title(f"t2m_mean ({actual_date})")
    plt.colorbar(im2, ax=ax, shrink=0.8, label="C")

    fig.suptitle(f"{actual_date}  |  {stats}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"hrrr_temp_{date_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────
# Figure 6: Freezing elevation estimate
# ─────────────────────────────────────────────────────────────────────
print("Generating Figure 6: hrrr_freezing_elevation.png")

# Bin pixels by DEM elevation (200m bands), find 0C crossing for each date
dem_flat = dem.ravel()
valid = ~np.isnan(dem_flat)
dem_valid = dem_flat[valid]

elev_min = np.nanmin(dem) // 200 * 200
elev_max = (np.nanmax(dem) // 200 + 1) * 200
bin_edges = np.arange(elev_min, elev_max + 200, 200)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_idx = np.digitize(dem_valid, bin_edges) - 1  # 0-based bin index

freezing_elevations = []
dates_out = []

for ti in range(len(times)):
    tmax_flat = hrrr["t2m_max"].isel(time=ti).values.ravel()[valid]

    # Mean t2m_max per elevation band
    band_means = np.full(len(bin_centers), np.nan)
    for bi in range(len(bin_centers)):
        mask_bi = bin_idx == bi
        if mask_bi.sum() > 10:
            band_means[bi] = np.nanmean(tmax_flat[mask_bi])

    # Find 0C crossing via linear interpolation
    good = ~np.isnan(band_means)
    if good.sum() < 2:
        freezing_elevations.append(np.nan)
        dates_out.append(times[ti])
        continue

    bc = bin_centers[good]
    bm = band_means[good]

    # Look for sign changes (temperature should decrease with elevation)
    crossings = []
    for j in range(len(bm) - 1):
        if (bm[j] >= 0 and bm[j + 1] < 0) or (bm[j] <= 0 and bm[j + 1] > 0):
            # Linear interpolation
            frac = (0 - bm[j]) / (bm[j + 1] - bm[j])
            cross_elev = bc[j] + frac * (bc[j + 1] - bc[j])
            crossings.append(cross_elev)

    if crossings:
        # Take the highest crossing as the freezing level
        freezing_elevations.append(max(crossings))
    elif np.all(bm > 0):
        # Everything above freezing
        freezing_elevations.append(bc[-1] + 200)
    elif np.all(bm < 0):
        # Everything below freezing
        freezing_elevations.append(bc[0] - 200)
    else:
        freezing_elevations.append(np.nan)
    dates_out.append(times[ti])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_out, freezing_elevations, "ko-", ms=5, label="Estimated freezing level (Tmax)")
ax.axhline(700, color="green", ls="--", lw=1, alpha=0.7, label="~Treeline (700m)")
ax.axhline(1500, color="brown", ls="--", lw=1, alpha=0.7, label="~Ridgeline (1500m)")
ax.set_xlabel("Date")
ax.set_ylabel("Estimated Freezing Elevation (m)")
ax.set_title("Estimated Freezing Level by SAR Date")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(FIGDIR / "hrrr_freezing_elevation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nAll figures saved to {FIGDIR.resolve()}")
