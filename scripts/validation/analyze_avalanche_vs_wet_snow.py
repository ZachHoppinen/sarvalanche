"""Analyze how avalanche signatures differ from wet snow background.

For each CNFAIC observation, extract SAR stats at the observation point
vs the surrounding background, and check if the avalanche signal survives
after accounting for wet snow effects.
"""

import sys
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

NC = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc")
FIG_DIR = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic/sample_detections/figures")


def main():
    ds = xr.open_dataset(NC)
    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds["track"].values
    dem = ds["dem"].values
    x_coords = ds.x.values
    y_coords = ds.y.values

    # Load observations
    obs = pd.read_csv("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic/cnfaic_obs_all.csv")
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.dropna(subset=["location_point", "date"])
    lats, lngs = zip(*obs["location_point"].map(
        lambda s: (float(ast.literal_eval(s)["lat"]), float(ast.literal_eval(s)["lng"]))))
    obs["lat"] = lats
    obs["lng"] = lngs
    obs = obs[(obs["date"] >= "2025-11-01") & (obs["date"] <= "2026-03-13")]

    def latlon_to_pixel(lat, lon):
        xi = np.argmin(np.abs(x_coords - lon))
        yi = np.argmin(np.abs(y_coords - lat))
        return yi, xi

    results = []
    for _, ob in obs.iterrows():
        yi, xi = latlon_to_pixel(ob["lat"], ob["lng"])
        if yi < 25 or yi >= dem.shape[0] - 25 or xi < 25 or xi >= dem.shape[1] - 25:
            continue

        ob_date = ob["date"]
        elev = dem[yi, xi]

        # Find nearest Track 160 timestep AFTER observation
        for i, t in enumerate(times):
            if int(tracks[i]) != 160:
                continue
            if t <= ob_date:
                continue
            if (t - ob_date).days > 14:
                break

            # Find Track 160 timestep BEFORE
            before_i = None
            for j in range(i - 1, -1, -1):
                if int(tracks[j]) == 160 and times[j] < ob_date:
                    before_i = j
                    break
            if before_i is None:
                continue

            # 5x5 window at observation
            slc = (slice(yi - 2, yi + 3), slice(xi - 2, xi + 3))
            vv_b = ds["VV"].isel(time=before_i).values[slc]
            vv_a = ds["VV"].isel(time=i).values[slc]
            vh_b = ds["VH"].isel(time=before_i).values[slc]
            vh_a = ds["VH"].isel(time=i).values[slc]

            valid = ~np.isnan(vv_b) & ~np.isnan(vv_a) & ~np.isnan(vh_b) & ~np.isnan(vh_a)
            if valid.sum() < 5:
                continue

            d_vv = (vv_a - vv_b)[valid].mean()
            d_vh = (vh_a - vh_b)[valid].mean()
            cr_b = (vh_b - vv_b)[valid].mean()
            cr_a = (vh_a - vv_a)[valid].mean()
            d_cr = cr_a - cr_b
            vv_before_mean = vv_b[valid].mean()

            # Background: annulus 10-25 px from point, same elevation band (±100m)
            bg_y = slice(max(0, yi - 25), min(dem.shape[0], yi + 26))
            bg_x = slice(max(0, xi - 25), min(dem.shape[1], xi + 26))

            dem_patch = dem[bg_y, bg_x]
            vv_bg_b = ds["VV"].isel(time=before_i).values[bg_y, bg_x]
            vv_bg_a = ds["VV"].isel(time=i).values[bg_y, bg_x]
            vh_bg_b = ds["VH"].isel(time=before_i).values[bg_y, bg_x]
            vh_bg_a = ds["VH"].isel(time=i).values[bg_y, bg_x]

            # Distance from center
            cy, cx = 25, 25  # center of 51x51 patch
            yy, xx = np.mgrid[:dem_patch.shape[0], :dem_patch.shape[1]]
            dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)

            # Annulus: 10-25 px, same elevation band
            elev_ok = np.abs(dem_patch - elev) < 200  # within 200m elevation
            annulus = (dist >= 10) & (dist <= 25) & elev_ok
            bg_valid = annulus & ~np.isnan(vv_bg_b) & ~np.isnan(vv_bg_a) & ~np.isnan(vh_bg_b) & ~np.isnan(vh_bg_a)

            if bg_valid.sum() < 50:
                continue

            bg_d_vv = (vv_bg_a - vv_bg_b)[bg_valid].mean()
            bg_d_vh = (vh_bg_a - vh_bg_b)[bg_valid].mean()
            bg_d_cr = ((vh_bg_a - vv_bg_a) - (vh_bg_b - vv_bg_b))[bg_valid].mean()
            bg_d_vv_std = (vv_bg_a - vv_bg_b)[bg_valid].std()
            bg_vv_before = vv_bg_b[bg_valid].mean()

            results.append({
                "date": ob_date.strftime("%Y-%m-%d"),
                "location": ob["location_name"],
                "d_size": ob["d_size"],
                "elev": elev,
                "pair": f'{times[before_i].strftime("%m/%d")}->{t.strftime("%m/%d")}',
                "span_days": (t - times[before_i]).days,
                # Absolute
                "d_vv": d_vv, "d_vh": d_vh, "d_cr": d_cr,
                "vv_before": vv_before_mean,
                # Background
                "bg_d_vv": bg_d_vv, "bg_d_vh": bg_d_vh, "bg_d_cr": bg_d_cr,
                "bg_d_vv_std": bg_d_vv_std, "bg_vv_before": bg_vv_before,
                # Relative (local anomaly)
                "rel_d_vv": d_vv - bg_d_vv,
                "rel_d_vh": d_vh - bg_d_vh,
                "rel_d_cr": d_cr - bg_d_cr,
                # Z-score
                "z_vv": (d_vv - bg_d_vv) / max(bg_d_vv_std, 0.1),
            })
            break

    rdf = pd.DataFrame(results)
    rdf.to_csv(FIG_DIR / "obs_vs_background.csv", index=False)
    print(f"Matched {len(rdf)} observations to SAR pairs")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n=== Absolute backscatter change ===")
    print(f"  Obs d_VV mean:       {rdf['d_vv'].mean():+.2f} dB")
    print(f"  Background d_VV:     {rdf['bg_d_vv'].mean():+.2f} dB")
    print(f"  Relative (obs-bg):   {rdf['rel_d_vv'].mean():+.2f} dB")
    print(f"  Obs > background:    {(rdf['rel_d_vv'] > 0).sum()}/{len(rdf)} "
          f"({100*(rdf['rel_d_vv'] > 0).mean():.0f}%)")

    print("\n=== Z-score (obs VV change vs local background) ===")
    print(f"  Mean z-score:        {rdf['z_vv'].mean():.2f}")
    print(f"  z > 1:               {(rdf['z_vv'] > 1).sum()}/{len(rdf)}")
    print(f"  z > 2:               {(rdf['z_vv'] > 2).sum()}/{len(rdf)}")

    print("\n=== Cross-pol ratio change ===")
    print(f"  Obs d(VH-VV):        {rdf['d_cr'].mean():+.3f} dB")
    print(f"  Background d(VH-VV): {rdf['bg_d_cr'].mean():+.3f} dB")
    print(f"  Relative:            {rdf['rel_d_cr'].mean():+.3f} dB")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Absolute d_VV: obs vs background
    ax = axes[0, 0]
    ax.scatter(rdf["bg_d_vv"], rdf["d_vv"], c=rdf["d_size"], cmap="YlOrRd",
               s=40, edgecolor="k", lw=0.5, vmin=1, vmax=4)
    lim = max(abs(rdf["bg_d_vv"].max()), abs(rdf["d_vv"].max())) + 1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Background d_VV (dB)")
    ax.set_ylabel("Obs point d_VV (dB)")
    ax.set_title("Absolute VV change:\nObs vs Background")

    # 2. Relative d_VV histogram
    ax = axes[0, 1]
    ax.hist(rdf["rel_d_vv"], bins=25, color="tab:blue", alpha=0.7, edgecolor="k")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(rdf["rel_d_vv"].mean(), color="red", ls="-", lw=2, label=f'mean={rdf["rel_d_vv"].mean():+.2f}')
    ax.set_xlabel("Relative d_VV (obs − background, dB)")
    ax.set_ylabel("Count")
    ax.set_title("Local Anomaly:\nObs VV change above background")
    ax.legend()

    # 3. Z-score histogram
    ax = axes[0, 2]
    ax.hist(rdf["z_vv"], bins=25, color="tab:orange", alpha=0.7, edgecolor="k")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(rdf["z_vv"].mean(), color="red", ls="-", lw=2, label=f'mean={rdf["z_vv"].mean():.2f}')
    ax.set_xlabel("Z-score (VV change)")
    ax.set_ylabel("Count")
    ax.set_title("Z-score:\nObs anomaly / local σ")
    ax.legend()

    # 4. Cross-pol ratio change: obs vs background
    ax = axes[1, 0]
    ax.scatter(rdf["bg_d_cr"], rdf["d_cr"], c=rdf["d_size"], cmap="YlOrRd",
               s=40, edgecolor="k", lw=0.5, vmin=1, vmax=4)
    lim_cr = max(abs(rdf["bg_d_cr"].max()), abs(rdf["d_cr"].max())) + 0.5
    ax.plot([-lim_cr, lim_cr], [-lim_cr, lim_cr], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Background d(VH−VV) (dB)")
    ax.set_ylabel("Obs point d(VH−VV) (dB)")
    ax.set_title("Cross-pol ratio change:\nObs vs Background")

    # 5. Relative d_cr vs relative d_vv
    ax = axes[1, 1]
    sc = ax.scatter(rdf["rel_d_vv"], rdf["rel_d_cr"], c=rdf["d_size"], cmap="YlOrRd",
                    s=50, edgecolor="k", lw=0.5, vmin=1, vmax=4)
    ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
    ax.axvline(0, color="k", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("Relative d_VV (dB)")
    ax.set_ylabel("Relative d(VH−VV) (dB)")
    ax.set_title("Joint Anomaly Space:\nVV change vs Cross-pol change")
    plt.colorbar(sc, ax=ax, label="D-size")

    # 6. Before VV level vs relative change (noise floor effect)
    ax = axes[1, 2]
    sc = ax.scatter(rdf["vv_before"], rdf["rel_d_vv"], c=rdf["d_size"], cmap="YlOrRd",
                    s=50, edgecolor="k", lw=0.5, vmin=1, vmax=4)
    ax.axhline(0, color="k", ls="--", lw=0.5, alpha=0.5)
    ax.axvline(-20, color="gray", ls=":", lw=1, label="Noise floor")
    ax.set_xlabel("VV before (dB)")
    ax.set_ylabel("Relative d_VV (dB)")
    ax.set_title("Noise Floor Effect:\nCan we detect when before is dark?")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="D-size")

    fig.suptitle("Avalanche Signal vs Wet Snow Background (2025–2026 Turnagain Pass)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "avalanche_vs_wet_snow_signal.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()

    # ── Elevation-band normalization test ─────────────────────────────────
    print("\n=== Elevation-band analysis ===")
    # Group observations by elevation band
    rdf["elev_band"] = pd.cut(rdf["elev"], bins=[0, 500, 800, 1100, 1400, 2000],
                               labels=["0-500m", "500-800m", "800-1100m", "1100-1400m", "1400m+"])
    for band, grp in rdf.groupby("elev_band", observed=True):
        if len(grp) < 2:
            continue
        print(f"  {band}: n={len(grp)}, "
              f"abs d_vv={grp['d_vv'].mean():+.2f}, "
              f"bg d_vv={grp['bg_d_vv'].mean():+.2f}, "
              f"relative={grp['rel_d_vv'].mean():+.2f}, "
              f"z={grp['z_vv'].mean():.2f}")


if __name__ == "__main__":
    main()
