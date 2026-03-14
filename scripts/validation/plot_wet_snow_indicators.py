"""Plot wet snow signatures in SAR: cross-pol ratio, noise floor, spatial patterns."""

import sys
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

NC = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic/netcdfs/Turnagain_Pass_and_Girdwood/season_2025-2026_Turnagain_Pass_and_Girdwood.nc")
FIG_DIR = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic/sample_detections/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SNOTEL_CSV = """\
Date,TAVG
2025-11-01,33
2025-11-09,16
2025-11-11,15
2025-11-21,31
2025-11-23,22
2025-11-27,32
2025-11-28,29
2025-11-30,34
2025-12-03,34
2025-12-05,24
2025-12-06,13
2025-12-10,2
2025-12-15,9
2025-12-17,6
2025-12-22,-4
2025-12-27,16
2025-12-29,8
2026-01-03,-7
2026-01-08,-5
2026-01-10,28
2026-01-15,31
2026-01-18,37
2026-01-25,29
2026-01-28,33
2026-01-30,33
2026-02-01,31
2026-02-03,36
2026-02-05,35
2026-02-08,32
2026-02-09,24
2026-02-13,21
2026-02-15,23
2026-02-20,8
2026-02-25,-1
2026-02-27,-5
2026-03-05,19
2026-03-09,7
2026-03-11,0
"""


def load_snotel():
    df = pd.read_csv(StringIO(SNOTEL_CSV))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df["TAVG_C"] = (df["TAVG"] - 32) * 5 / 9
    return df


def compute_stats(ds):
    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds["track"].values
    rows = []
    for i, t in enumerate(times):
        if t < pd.Timestamp("2025-11-01") or t > pd.Timestamp("2026-03-13"):
            continue
        vv = ds["VV"].isel(time=i).values
        vh = ds["VH"].isel(time=i).values
        both = ~np.isnan(vv) & ~np.isnan(vh)
        if both.sum() < 1000:
            continue
        vv_v, vh_v = vv[both], vh[both]
        cr = vh_v - vv_v
        rows.append({
            "date": t, "track": int(tracks[i]),
            "vv_mean": vv_v.mean(), "vv_std": vv_v.std(),
            "vv_p05": np.percentile(vv_v, 5),
            "vv_below_m20": 100 * (vv_v < -20).mean(),
            "vh_mean": vh_v.mean(), "vh_std": vh_v.std(),
            "vh_below_m30": 100 * (vh_v < -30).mean(),
            "cr_mean": cr.mean(), "cr_std": cr.std(),
            "cr_p05": np.percentile(cr, 5), "cr_p95": np.percentile(cr, 95),
        })
    return pd.DataFrame(rows)


def plot_timeseries(df, snotel):
    t160 = df[df["track"] == 160].sort_values("date")

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True,
                             gridspec_kw={"hspace": 0.06})

    # Temperature
    ax = axes[0]
    ax.bar(snotel.index, snotel["TAVG_C"],
           color=["red" if v > 0 else "tab:blue" for v in snotel["TAVG_C"]],
           width=1, alpha=0.6)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Temp (°C)")
    ax.set_title("Wet Snow Indicators from SAR Scene Statistics (Track 160)",
                 fontsize=13, fontweight="bold")

    # Cross-pol ratio
    ax = axes[1]
    ax.plot(t160["date"], t160["cr_mean"], "o-", color="tab:blue", ms=6, lw=1.5, label="Mean VH−VV")
    ax.fill_between(t160["date"], t160["cr_p05"], t160["cr_p95"], alpha=0.15, color="tab:blue")
    ax.set_ylabel("VH − VV (dB)")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("← wetter (less depolarized)", xy=(0.01, 0.95),
                xycoords="axes fraction", fontsize=8, color="gray", va="top")
    ax.annotate("drier (more depolarized) →", xy=(0.01, 0.05),
                xycoords="axes fraction", fontsize=8, color="gray", va="bottom")

    # Noise floor %
    ax = axes[2]
    ax.plot(t160["date"], t160["vv_below_m20"], "o-", color="tab:red", ms=6, lw=1.5,
            label="% VV < −20 dB")
    ax.plot(t160["date"], t160["vh_below_m30"], "s-", color="tab:blue", ms=5, lw=1.5,
            label="% VH < −30 dB")
    ax.set_ylabel("% at noise floor")
    ax.legend(fontsize=8, loc="upper right")

    # Scene std
    ax = axes[3]
    ax.plot(t160["date"], t160["vv_std"], "o-", color="tab:red", ms=6, lw=1.5, label="VV σ")
    ax.plot(t160["date"], t160["vh_std"], "s-", color="tab:blue", ms=5, lw=1.5, label="VH σ")
    ax.set_ylabel("Scene σ (dB)")
    ax.legend(fontsize=8, loc="upper right")

    # Mean backscatter
    ax = axes[4]
    ax.plot(t160["date"], t160["vv_mean"], "o-", color="tab:red", ms=6, lw=1.5, label="VV mean")
    ax.plot(t160["date"], t160["vh_mean"], "s-", color="tab:blue", ms=5, lw=1.5, label="VH mean")
    ax.set_ylabel("Backscatter (dB)")
    ax.legend(fontsize=8, loc="upper right")

    for a in axes:
        a.axvspan(pd.Timestamp("2025-11-27"), pd.Timestamp("2025-12-04"), color="red", alpha=0.06)
        a.axvspan(pd.Timestamp("2026-01-15"), pd.Timestamp("2026-02-08"), color="red", alpha=0.06)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlim(pd.Timestamp("2025-11-01"), pd.Timestamp("2026-03-13"))

    plt.tight_layout()
    path = FIG_DIR / "wet_snow_indicators_timeseries.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_detail(ds, df, snotel):
    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds["track"].values
    t160 = df[df["track"] == 160].sort_values("date")

    dates_compare = {
        "2025-12-17": ("Cold/Dry (Dec 17)", "tab:blue"),
        "2026-02-03": ("Wet/Warm (Feb 03)", "tab:red"),
        "2026-02-15": ("Refrozen (Feb 15)", "tab:green"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Cross-pol ratio histograms
    ax = axes[0, 0]
    bins_cr = np.linspace(-18, 2, 150)
    for date_str, (label, color) in dates_compare.items():
        dt = pd.Timestamp(date_str)
        for i, t in enumerate(times):
            if abs((t - dt).total_seconds()) < 86400 * 2 and int(tracks[i]) == 160:
                vv = ds["VV"].isel(time=i).values
                vh = ds["VH"].isel(time=i).values
                both = ~np.isnan(vv) & ~np.isnan(vh)
                cr = vh[both] - vv[both]
                ax.hist(cr, bins=bins_cr, alpha=0.4, color=color, density=True,
                        histtype="stepfilled", label=label)
                ax.hist(cr, bins=bins_cr, color=color, density=True, histtype="step", lw=1.5)
                break
    ax.set_xlabel("VH − VV (dB)")
    ax.set_ylabel("Density")
    ax.set_title("Cross-Pol Ratio Distribution")
    ax.legend(fontsize=9)

    # Top right: VV vs VH scatter
    ax = axes[0, 1]
    for date_str, (label, color) in dates_compare.items():
        dt = pd.Timestamp(date_str)
        for i, t in enumerate(times):
            if abs((t - dt).total_seconds()) < 86400 * 2 and int(tracks[i]) == 160:
                vv = ds["VV"].isel(time=i).values
                vh = ds["VH"].isel(time=i).values
                both = ~np.isnan(vv) & ~np.isnan(vh)
                vv_s, vh_s = vv[both], vh[both]
                rng = np.random.default_rng(42)
                idx = rng.choice(len(vv_s), min(5000, len(vv_s)), replace=False)
                ax.scatter(vv_s[idx], vh_s[idx], s=1, alpha=0.3, color=color, label=label)
                break
    ax.set_xlabel("VV (dB)")
    ax.set_ylabel("VH (dB)")
    ax.set_title("VV vs VH Scatter (5k sample)")
    ax.plot([-30, 5], [-30, 5], "k--", lw=0.5, alpha=0.3)
    ax.legend(fontsize=9, markerscale=5)
    ax.set_xlim(-30, 5)
    ax.set_ylim(-40, 5)

    # Bottom left: VH histograms
    ax = axes[1, 0]
    bins_vh = np.linspace(-42, 5, 150)
    for date_str, (label, color) in dates_compare.items():
        dt = pd.Timestamp(date_str)
        for i, t in enumerate(times):
            if abs((t - dt).total_seconds()) < 86400 * 2 and int(tracks[i]) == 160:
                vh = ds["VH"].isel(time=i).values
                v = vh[~np.isnan(vh)]
                ax.hist(v, bins=bins_vh, alpha=0.4, color=color, density=True,
                        histtype="stepfilled", label=label)
                ax.hist(v, bins=bins_vh, color=color, density=True, histtype="step", lw=1.5)
                break
    ax.set_xlabel("VH Backscatter (dB)")
    ax.set_ylabel("Density")
    ax.set_title("VH Distribution")
    ax.axvline(-30, color="gray", ls=":", lw=1)
    ax.legend(fontsize=9)

    # Bottom right: Noise floor % vs temperature
    ax = axes[1, 1]
    for _, row in t160.iterrows():
        nearest_idx = np.argmin(np.abs(snotel.index - row["date"]))
        temp = snotel.iloc[nearest_idx]["TAVG_C"]
        ax.scatter(temp, row["vv_below_m20"], color="tab:red", s=60, zorder=3,
                   edgecolor="k", lw=0.5)
        ax.annotate(row["date"].strftime("%m/%d"), xy=(temp, row["vv_below_m20"]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("SNOTEL Avg Temperature (°C)")
    ax.set_ylabel("% VV pixels < −20 dB")
    ax.set_title("Noise Floor % vs Temperature")
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.5)

    fig.suptitle("Wet Snow Signatures in SAR: Track 160, Turnagain Pass 2025–2026",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "wet_snow_signatures_detail.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_spatial(ds):
    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds["track"].values

    dates_compare = {
        "2025-12-17": "Cold/Dry (Dec 17)",
        "2026-02-03": "Wet/Warm (Feb 03)",
        "2026-02-15": "Refrozen (Feb 15)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ims_vv = []
    ims_cr = []
    for col, (date_str, label) in enumerate(dates_compare.items()):
        dt = pd.Timestamp(date_str)
        for i, t in enumerate(times):
            if abs((t - dt).total_seconds()) < 86400 * 2 and int(tracks[i]) == 160:
                vv = ds["VV"].isel(time=i).values
                vh = ds["VH"].isel(time=i).values
                both = ~np.isnan(vv) & ~np.isnan(vh)
                cr = np.full_like(vv, np.nan)
                cr[both] = vh[both] - vv[both]

                im1 = axes[0, col].imshow(vv, vmin=-25, vmax=0, cmap="gray")
                axes[0, col].set_title(f"VV: {label}", fontsize=10)
                axes[0, col].axis("off")
                ims_vv.append(im1)

                im2 = axes[1, col].imshow(cr, vmin=-12, vmax=-3, cmap="RdYlBu")
                axes[1, col].set_title(f"VH−VV: {label}", fontsize=10)
                axes[1, col].axis("off")
                ims_cr.append(im2)
                break

    fig.colorbar(ims_vv[-1], ax=axes[0, :], label="VV (dB)", shrink=0.8, pad=0.02)
    fig.colorbar(ims_cr[-1], ax=axes[1, :], label="VH−VV (dB)", shrink=0.8, pad=0.02)

    fig.suptitle("Spatial Pattern of Wet Snow in SAR (Track 160)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "wet_snow_spatial.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    snotel = load_snotel()
    ds = xr.open_dataset(NC)
    print("Computing stats...")
    df = compute_stats(ds)

    print("Plotting timeseries...")
    plot_timeseries(df, snotel)

    print("Plotting detail...")
    plot_detail(ds, df, snotel)

    print("Plotting spatial...")
    plot_spatial(ds)

    print("Done!")


if __name__ == "__main__":
    main()
