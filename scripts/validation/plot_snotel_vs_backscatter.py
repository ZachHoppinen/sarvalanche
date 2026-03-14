"""
Plot SNOTEL temperature/SWE alongside scene-mean SAR backscatter for
2025-2026 Turnagain Pass. Shows how freeze-thaw drives backscatter swings.
"""

import sys
from pathlib import Path
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# ── SNOTEL data (Turnagain Pass, station 954) ────────────────────────────
SNOTEL_CSV = """\
Date,TAVG,TMAX,TMIN,SNWD,SWE,PREC
2025-11-01,33.0,36.0,29.0,18,3.8,13.8
2025-11-02,29.0,35.0,26.0,19,4.7,15.1
2025-11-03,29.0,34.0,26.0,18,4.7,15.1
2025-11-04,27.0,33.0,25.0,21,4.9,15.3
2025-11-05,23.0,28.0,19.0,20,4.8,15.3
2025-11-06,20.0,22.0,19.0,20,4.8,15.3
2025-11-07,23.0,27.0,20.0,34,5.7,16.2
2025-11-08,26.0,30.0,21.0,34,6.4,17.0
2025-11-09,16.0,23.0,12.0,39,7.0,17.6
2025-11-10,24.0,28.0,15.0,37,7.0,17.6
2025-11-11,15.0,18.0,13.0,35,7.0,17.6
2025-11-12,20.0,25.0,14.0,34,7.0,17.6
2025-11-13,24.0,28.0,21.0,32,7.0,17.6
2025-11-14,25.0,30.0,19.0,31,6.9,17.6
2025-11-15,18.0,26.0,13.0,30,6.8,17.6
2025-11-16,22.0,29.0,17.0,30,6.8,17.6
2025-11-17,29.0,32.0,27.0,29,6.8,17.6
2025-11-18,28.0,33.0,26.0,32,6.9,17.7
2025-11-19,28.0,33.0,25.0,30,6.8,17.7
2025-11-20,33.0,35.0,31.0,30,6.9,17.8
2025-11-21,31.0,33.0,28.0,40,8.5,19.3
2025-11-22,25.0,29.0,22.0,48,9.1,20.2
2025-11-23,22.0,25.0,18.0,44,9.1,20.2
2025-11-24,20.0,22.0,19.0,42,9.1,20.2
2025-11-25,21.0,25.0,18.0,40,9.1,20.2
2025-11-26,29.0,32.0,23.0,38,9.1,20.2
2025-11-27,32.0,34.0,30.0,38,9.1,20.2
2025-11-28,29.0,32.0,25.0,55,10.4,21.7
2025-11-29,31.0,36.0,26.0,50,10.4,21.7
2025-11-30,34.0,35.0,33.0,47,10.7,22.0
2025-12-01,31.0,35.0,30.0,44,10.8,22.1
2025-12-02,34.0,37.0,32.0,43,10.8,22.1
2025-12-03,34.0,35.0,31.0,42,10.9,22.2
2025-12-04,33.0,36.0,30.0,41,10.9,22.2
2025-12-05,24.0,30.0,17.0,41,10.9,22.2
2025-12-06,13.0,18.0,10.0,39,10.9,22.3
2025-12-07,7.0,12.0,3.0,42,11.0,22.4
2025-12-08,10.0,14.0,7.0,41,11.0,22.5
2025-12-09,11.0,16.0,4.0,42,11.0,22.5
2025-12-10,2.0,5.0,0.0,41,11.0,22.5
2025-12-11,7.0,13.0,3.0,40,11.0,22.5
2025-12-12,16.0,22.0,10.0,40,11.0,22.5
2025-12-13,11.0,16.0,7.0,40,11.0,22.5
2025-12-14,8.0,14.0,5.0,40,11.0,22.5
2025-12-15,9.0,14.0,5.0,40,11.0,22.5
2025-12-16,6.0,9.0,4.0,40,11.0,22.5
2025-12-17,6.0,8.0,3.0,40,11.0,22.5
2025-12-18,7.0,14.0,3.0,40,11.0,22.5
2025-12-19,3.0,8.0,0.0,40,11.0,22.5
2025-12-20,3.0,7.0,1.0,40,11.0,22.5
2025-12-21,-2.0,5.0,-7.0,40,11.0,22.5
2025-12-22,-4.0,1.0,-7.0,40,11.1,22.5
2025-12-23,0.0,7.0,-5.0,40,11.1,22.5
2025-12-24,13.0,20.0,7.0,40,11.1,22.5
2025-12-25,8.0,17.0,2.0,40,11.1,22.5
2025-12-26,14.0,19.0,5.0,39,11.1,22.5
2025-12-27,16.0,22.0,5.0,41,11.1,22.5
2025-12-28,1.0,5.0,-2.0,42,11.2,22.6
2025-12-29,8.0,11.0,4.0,41,11.2,22.6
2025-12-30,11.0,21.0,5.0,49,11.6,23.2
2025-12-31,7.0,13.0,0.0,50,11.6,23.3
2026-01-01,-1.0,2.0,-3.0,50,11.6,23.3
2026-01-02,-5.0,-2.0,-7.0,49,11.6,23.3
2026-01-03,-7.0,-5.0,-9.0,48,11.6,23.3
2026-01-04,-1.0,13.0,-10.0,48,11.6,23.3
2026-01-05,18.0,23.0,12.0,49,11.6,23.4
2026-01-06,14.0,24.0,5.0,50,11.6,23.5
2026-01-07,-4.0,5.0,-9.0,53,11.8,23.7
2026-01-08,-5.0,16.0,-13.0,52,11.8,23.7
2026-01-09,13.0,19.0,5.0,51,11.9,23.7
2026-01-10,28.0,33.0,12.0,52,12.0,23.8
2026-01-11,14.0,21.0,11.0,58,12.6,24.4
2026-01-12,5.0,11.0,1.0,57,12.6,24.4
2026-01-13,15.0,25.0,5.0,57,12.6,24.4
2026-01-14,20.0,26.0,15.0,57,12.6,24.5
2026-01-15,31.0,34.0,26.0,56,12.6,24.5
2026-01-16,29.0,35.0,21.0,64,13.7,25.3
2026-01-17,33.0,38.0,21.0,65,14.3,25.9
2026-01-18,37.0,43.0,33.0,64,14.5,26.1
2026-01-19,37.0,43.0,31.0,63,14.5,26.1
2026-01-20,35.0,41.0,31.0,62,14.6,26.2
2026-01-21,32.0,37.0,30.0,61,14.6,26.2
2026-01-22,33.0,37.0,29.0,60,14.6,26.2
2026-01-23,35.0,40.0,29.0,60,14.6,26.2
2026-01-24,30.0,34.0,28.0,59,14.7,26.2
2026-01-25,29.0,33.0,26.0,59,14.7,26.2
2026-01-26,25.0,27.0,21.0,59,14.7,26.2
2026-01-27,28.0,35.0,20.0,61,15.0,26.5
2026-01-28,33.0,36.0,30.0,62,15.1,26.6
2026-01-29,34.0,37.0,31.0,64,15.4,27.1
2026-01-30,33.0,35.0,31.0,76,17.0,28.3
2026-01-31,33.0,35.0,28.0,72,17.4,29.2
2026-02-01,31.0,37.0,25.0,70,17.4,29.3
2026-02-02,34.0,35.0,33.0,69,17.4,29.3
2026-02-03,36.0,40.0,33.0,66,17.5,29.5
2026-02-04,37.0,42.0,35.0,64,17.9,30.8
2026-02-05,35.0,37.0,34.0,62,18.0,30.9
2026-02-06,32.0,35.0,28.0,61,18.2,31.5
2026-02-07,35.0,39.0,31.0,60,18.2,31.5
2026-02-08,32.0,35.0,28.0,59,18.2,31.5
2026-02-09,24.0,29.0,19.0,59,18.2,31.5
2026-02-10,25.0,29.0,16.0,59,18.2,31.5
2026-02-11,26.0,29.0,23.0,64,18.7,32.0
2026-02-12,25.0,28.0,22.0,64,18.7,32.0
2026-02-13,21.0,27.0,17.0,63,18.7,32.0
2026-02-14,24.0,28.0,20.0,62,18.7,32.0
2026-02-15,23.0,26.0,19.0,65,19.0,32.3
2026-02-16,14.0,23.0,8.0,65,19.0,32.3
2026-02-17,14.0,19.0,10.0,64,19.1,32.4
2026-02-18,20.0,24.0,15.0,64,19.1,32.4
2026-02-19,14.0,20.0,2.0,65,19.1,32.4
2026-02-20,8.0,16.0,3.0,64,19.1,32.4
2026-02-21,4.0,12.0,-1.0,64,19.1,32.4
2026-02-22,13.0,21.0,7.0,64,19.1,32.4
2026-02-23,29.0,37.0,20.0,63,19.1,32.4
2026-02-24,24.0,30.0,10.0,63,19.1,32.4
2026-02-25,-1.0,10.0,-7.0,63,19.1,32.4
2026-02-26,-5.0,1.0,-10.0,63,19.1,32.4
2026-02-27,-5.0,4.0,-9.0,62,19.1,32.4
2026-02-28,-6.0,3.0,-10.0,62,19.1,32.4
2026-03-01,-2.0,9.0,-10.0,62,19.1,32.4
2026-03-02,1.0,10.0,-4.0,62,19.1,32.4
2026-03-03,1.0,9.0,-4.0,62,19.1,32.4
2026-03-04,4.0,14.0,-4.0,62,19.1,32.4
2026-03-05,19.0,26.0,12.0,62,19.1,32.4
2026-03-06,28.0,33.0,22.0,62,19.1,32.4
2026-03-07,16.0,22.0,9.0,71,19.9,33.1
2026-03-08,9.0,15.0,3.0,72,20.0,33.2
2026-03-09,7.0,13.0,1.0,71,20.0,33.2
2026-03-10,-1.0,7.0,-7.0,70,20.0,33.2
2026-03-11,0.0,10.0,-8.0,70,20.0,33.2
2026-03-12,4.0,16.0,-3.0,70,20.0,33.2
"""

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic")
NC = BASE / "netcdfs" / "Turnagain_Pass_and_Girdwood" / "season_2025-2026_Turnagain_Pass_and_Girdwood.nc"
FIG_DIR = BASE / "sample_detections" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_snotel():
    df = pd.read_csv(StringIO(SNOTEL_CSV))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    # Convert F to C
    for col in ["TAVG", "TMAX", "TMIN"]:
        df[f"{col}_C"] = (df[col] - 32) * 5 / 9
    return df


def load_sar_stats(ds):
    """Compute scene-mean VV backscatter and std per time step."""
    times = pd.DatetimeIndex(ds.time.values)
    tracks = ds["track"].values
    rows = []
    for i, t in enumerate(times):
        vv = ds["VV"].isel(time=i).values
        valid = vv[~np.isnan(vv)]
        if len(valid) == 0:
            continue
        rows.append({
            "date": t,
            "track": int(tracks[i]),
            "vv_mean": np.mean(valid),
            "vv_std": np.std(valid),
            "vv_p05": np.percentile(valid, 5),
            "vv_p95": np.percentile(valid, 95),
            "n_valid": len(valid),
        })
    return pd.DataFrame(rows)


DANGER_COLORS = {
    0: "#cccccc",  # no rating
    1: "#50b848",  # low
    2: "#f4e500",  # moderate
    3: "#f7941e",  # considerable
    4: "#ed1c24",  # high
    5: "#1a1a1a",  # extreme
}
DANGER_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "Considerable", 4: "High", 5: "Extreme"}


def fetch_cnfaic_danger(start="2025-11-01", end="2026-03-13"):
    """Fetch daily avalanche danger ratings for Turnagain Pass from avalanche.org."""
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://www.avalanche.org",
        "referer": "https://www.avalanche.org/",
        "user-agent": "Mozilla/5.0",
    }
    url = "https://api.avalanche.org/v2/public/products"
    params = {
        "avalanche_center_id": "CNFAIC",
        "date_start": start,
        "date_end": end,
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()

    rows = []
    for item in r.json():
        if item["product_type"] != "forecast":
            continue
        zones = item.get("forecast_zone", [])
        if not any("Turnagain" in z.get("name", "") for z in zones):
            continue
        danger = item.get("danger", [])
        if not danger:
            continue
        date = item["start_date"][:10]
        d = danger[0]
        rows.append({
            "date": pd.Timestamp(date),
            "above": d.get("upper", 0),
            "near": d.get("middle", 0),
            "below": d.get("lower", 0),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
    df = df.set_index("date")
    return df


def _plot_danger_bar(ax, danger_df, col="above", label="Above treeline"):
    """Plot danger as a colored bar chart on an axis."""
    dates = danger_df.index
    levels = danger_df[col].values
    colors = [DANGER_COLORS.get(int(lv), "#cccccc") for lv in levels]

    ax.bar(dates, np.ones(len(dates)), width=1.0, color=colors,
           edgecolor="none", alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(f"Danger\n({label})", fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    seen = sorted(set(int(lv) for lv in levels))
    handles = [Patch(facecolor=DANGER_COLORS[lv], label=DANGER_LABELS[lv]) for lv in seen]
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=len(seen))


def plot_season_overview(snotel, sar, fig_path, danger=None):
    """4-panel figure: danger, temperature, snow depth/SWE, scene-mean backscatter."""
    n_panels = 4 if danger is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"hspace": 0.08,
                                          "height_ratios": [0.6, 1, 1, 1] if danger is not None else [1, 1, 1]})

    panel_idx = 0

    # ── Panel 0: Danger rating ────────────────────────────────────────────
    if danger is not None:
        _plot_danger_bar(axes[panel_idx], danger, col="above", label="Above treeline")
        axes[panel_idx].set_title(
            "Turnagain Pass 2025–2026: Avalanche Danger, SNOTEL (954), Sentinel-1 Backscatter",
            fontsize=13, fontweight="bold")
        panel_idx += 1
    else:
        axes[panel_idx].set_title(
            "Turnagain Pass 2025–2026: SNOTEL (954) + Sentinel-1 Backscatter",
            fontsize=13, fontweight="bold")

    # ── Panel 1: Temperature ──────────────────────────────────────────────
    ax = axes[panel_idx]
    ax.fill_between(snotel.index, snotel["TMIN_C"], snotel["TMAX_C"],
                    alpha=0.25, color="tab:red", label="Tmin–Tmax")
    ax.plot(snotel.index, snotel["TAVG_C"], color="tab:red", lw=1.5,
            label="Tavg")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", fontsize=9)

    # Shade above-freezing periods
    above_freezing = snotel["TMAX_C"] > 0
    for start, end in _contiguous_spans(snotel.index, above_freezing & (snotel["TAVG_C"] > 0)):
        for a in axes:
            a.axvspan(start, end, color="red", alpha=0.06)
    panel_idx += 1

    # ── Panel 2: Snow depth + SWE ────────────────────────────────────────
    ax = axes[panel_idx]
    ax.plot(snotel.index, snotel["SNWD"], color="tab:blue", lw=1.5,
            label="Snow depth (in)")
    ax2 = ax.twinx()
    ax2.plot(snotel.index, snotel["SWE"], color="tab:cyan", lw=1.5,
             ls="--", label="SWE (in)")
    ax.set_ylabel("Snow depth (in)", color="tab:blue")
    ax2.set_ylabel("SWE (in)", color="tab:cyan")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    panel_idx += 1

    # ── Panel 3: Scene-mean VV backscatter per track ─────────────────────
    ax = axes[panel_idx]
    track_colors = {65: "tab:green", 131: "tab:orange", 160: "tab:purple"}
    track_markers = {65: "^", 131: "s", 160: "o"}
    for tid in sorted(sar["track"].unique()):
        sub = sar[sar["track"] == tid].sort_values("date")
        color = track_colors.get(tid, "gray")
        marker = track_markers.get(tid, "o")
        ax.plot(sub["date"], sub["vv_mean"], color=color, marker=marker,
                ms=5, lw=1.2, label=f"Track {tid}")
        ax.fill_between(sub["date"], sub["vv_p05"], sub["vv_p95"],
                        alpha=0.12, color=color)
    ax.set_ylabel("VV backscatter (dB)")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=9)

    # Mark SAR acquisition dates on non-danger panels
    for _, row in sar.iterrows():
        for a in axes[1:panel_idx]:
            a.axvline(row["date"], color=track_colors.get(row["track"], "gray"),
                      alpha=0.15, lw=0.6)

    # Format x-axis
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlim(pd.Timestamp("2025-11-01"), pd.Timestamp("2026-03-13"))

    plt.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close(fig)


def plot_feb_zoom(snotel, sar, ds, fig_path, danger=None):
    """Zoomed view of the Jan 25 – Feb 20 warm event with backscatter change."""
    t_start = pd.Timestamp("2026-01-20")
    t_end = pd.Timestamp("2026-02-25")
    sn = snotel.loc[t_start:t_end]

    n_panels = 4 if danger is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 11 if danger is not None else 9),
                             sharex=True,
                             gridspec_kw={"hspace": 0.08,
                                          "height_ratios": [0.5, 1, 1, 1] if danger is not None else [1, 1, 1]})

    panel_idx = 0

    # ── Panel 0: Danger ───────────────────────────────────────────────────
    if danger is not None:
        dz = danger.loc[t_start:t_end]
        _plot_danger_bar(axes[panel_idx], dz, col="above", label="Above treeline")
        axes[panel_idx].set_title(
            "Rain-on-Snow Event: Jan 28 – Feb 8 2026 (Turnagain Pass)",
            fontsize=12, fontweight="bold")
        panel_idx += 1
    else:
        axes[panel_idx].set_title(
            "Rain-on-Snow Event: Jan 28 – Feb 8 2026 (Turnagain Pass SNOTEL + SAR)",
            fontsize=12, fontweight="bold")

    # ── Temperature ───────────────────────────────────────────────────────
    ax = axes[panel_idx]
    ax.fill_between(sn.index, sn["TMIN_C"], sn["TMAX_C"],
                    alpha=0.25, color="tab:red")
    ax.plot(sn.index, sn["TAVG_C"], color="tab:red", lw=2, label="Tavg")
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=9)

    # Shade above freezing
    for a in axes:
        a.axvspan(pd.Timestamp("2026-01-28"), pd.Timestamp("2026-02-08"),
                  color="red", alpha=0.08, label="_nolegend_")
    panel_idx += 1

    # ── Snow depth change ─────────────────────────────────────────────────
    ax = axes[panel_idx]
    ax.plot(sn.index, sn["SNWD"], color="tab:blue", lw=2, marker=".",
            label="Snow depth (in)")
    ax2 = ax.twinx()
    ax2.plot(sn.index, sn["SWE"], color="tab:cyan", lw=2, ls="--",
             marker=".", label="SWE (in)")
    ax.set_ylabel("Snow depth (in)", color="tab:blue")
    ax2.set_ylabel("SWE (in)", color="tab:cyan")

    # Annotate the key dynamics
    ax.annotate("76\" peak\n(+17\" in 4 days)",
                xy=(pd.Timestamp("2026-01-30"), 76), fontsize=8,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="tab:blue"),
                xytext=(pd.Timestamp("2026-01-28"), 82))
    ax.annotate("59\" (settlement\n+ melt)",
                xy=(pd.Timestamp("2026-02-09"), 59), fontsize=8,
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="tab:blue"),
                xytext=(pd.Timestamp("2026-02-11"), 53))
    ax2.annotate("SWE rising\nduring melt\n= rain-on-snow",
                 xy=(pd.Timestamp("2026-02-03"), 17.5), fontsize=8,
                 ha="left", color="tab:cyan",
                 xytext=(pd.Timestamp("2026-02-05"), 15.5),
                 arrowprops=dict(arrowstyle="->", color="tab:cyan"))

    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    panel_idx += 1

    # ── Scene-mean VV per track ───────────────────────────────────────────
    ax = axes[panel_idx]
    track_colors = {65: "tab:green", 131: "tab:orange", 160: "tab:purple"}
    sar_zoom = sar[(sar["date"] >= t_start) & (sar["date"] <= t_end)]
    for tid in sorted(sar_zoom["track"].unique()):
        sub = sar_zoom[sar_zoom["track"] == tid].sort_values("date")
        color = track_colors.get(tid, "gray")
        ax.errorbar(sub["date"], sub["vv_mean"],
                    yerr=[sub["vv_mean"] - sub["vv_p05"],
                          sub["vv_p95"] - sub["vv_mean"]],
                    color=color, marker="o", ms=7, lw=1.5, capsize=3,
                    label=f"Track {tid} (p5–p95)")
    ax.set_ylabel("VV backscatter (dB)")
    ax.legend(fontsize=9)

    # Mark the two SAR dates for the Feb 14 pair
    for d, label in [(pd.Timestamp("2026-02-03"), "T160 before"),
                     (pd.Timestamp("2026-02-15"), "T160 after")]:
        for a in axes:
            a.axvline(d, color="tab:purple", alpha=0.4, lw=1.2, ls=":")
        axes[-1].annotate(label, xy=(d, axes[-1].get_ylim()[1]),
                          fontsize=8, color="tab:purple", ha="center",
                          va="top", rotation=90)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close(fig)


def _contiguous_spans(index, mask):
    """Yield (start, end) for contiguous True runs in a boolean series."""
    in_span = False
    start = None
    for i, val in enumerate(mask):
        if val and not in_span:
            start = index[i]
            in_span = True
        elif not val and in_span:
            yield start, index[i - 1]
            in_span = False
    if in_span:
        yield start, index[-1]


def main():
    print("Loading SNOTEL data...")
    snotel = load_snotel()

    print("Fetching CNFAIC danger ratings...")
    danger = fetch_cnfaic_danger(start="2025-11-01", end="2026-03-13")
    print(f"  {len(danger)} danger ratings fetched")

    print("Loading SAR dataset...")
    ds = xr.open_dataset(NC)

    print("Computing scene-mean backscatter per timestep...")
    sar = load_sar_stats(ds)

    # Filter to winter season
    sar = sar[(sar["date"] >= "2025-11-01") & (sar["date"] <= "2026-03-13")]

    print(f"SAR: {len(sar)} timesteps, SNOTEL: {len(snotel)} days")

    plot_season_overview(snotel, sar, FIG_DIR / "snotel_vs_backscatter_season.png", danger=danger)
    plot_feb_zoom(snotel, sar, ds, FIG_DIR / "snotel_vs_backscatter_feb_zoom.png", danger=danger)

    print("Done!")


if __name__ == "__main__":
    main()
