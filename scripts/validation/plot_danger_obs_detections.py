"""
Plot avalanche danger vs observed avalanches vs CNN detections for a single zone.

Shows three panels:
  - Top: daily max danger rating (above-treeline), color-coded
  - Middle: 7-day rolling sum of observed avalanche count
  - Bottom: 7-day rolling sum of high-confidence CNN detection pixels

Supports multiple zone configs via --zone flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]

# Zone configurations
ZONES = {
    "sawtooth": {
        "zone_name": "Sawtooth & Western Smoky Mtns",
        "zone_dir": "Sawtooth_&_Western_Smoky_Mtns",
        "data_root": ROOT / "local" / "issw",
        "danger_csv": "snfac_dangers_2021_2025.csv",
        "obs_csv": "snfac_obs_2021_2025.csv",
        "nc_subdir": "netcdfs",
        "xlim_start": "2022-10-01",
        "out_name": "danger_obs_detections_sawtooth.png",
    },
    "turnagain": {
        "zone_name": "Turnagain Pass and Girdwood",
        "zone_dir": "Turnagain_Pass_and_Girdwood",
        "data_root": ROOT / "local" / "cnfaic",
        "danger_csv": "cnfaic_dangers_2020_2025.csv",
        "obs_csv": "cnfaic_obs_2020_2025.csv",
        "nc_subdir": "netcdfs",
        "xlim_start": "2024-10-01",
        "out_name": "danger_obs_detections_turnagain.png",
    },
}

DANGER_COLORS = {1: "#4CAF50", 2: "#FFEB3B", 3: "#FF9800", 4: "#F44336", 5: "#000000"}


def load_dangers(path: Path, zone_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["zone_name"] == zone_name]
    df = df.groupby("date")["danger_above_current"].max().reset_index()
    return df.set_index("date").sort_index()


def load_observations(path: Path, zone_name: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["zone_name"] == zone_name]
    daily = df.groupby("date").size()
    daily.name = "n_obs"
    return daily.sort_index()


def load_onset_detections(nc_dir: Path, zone_dir: str, min_confidence: float = 0.7) -> pd.Series:
    """Load high-confidence CNN onset detections, return daily pixel counts."""
    zone_path = nc_dir / zone_dir
    records = []
    for season_dir in sorted(zone_path.glob("v2_season_inference_[0-9]*")):
        if "mag" in season_dir.name:
            continue
        onset_path = season_dir / "temporal_onset.nc"
        if not onset_path.exists():
            continue
        ds = xr.open_dataset(onset_path)
        mask = ds["candidate_mask"].values
        onset = ds["onset_date"].values
        conf = ds["confidence"].values if "confidence" in ds else np.ones_like(mask, dtype=float)
        peak = ds["peak_prob"].values if "peak_prob" in ds else np.ones_like(mask, dtype=float)

        valid = mask & ~np.isnat(onset) & (conf >= min_confidence) & (peak >= 0.5)
        dates = onset[valid]
        if len(dates) == 0:
            ds.close()
            continue
        unique_dates, counts = np.unique(dates, return_counts=True)
        for d, c in zip(unique_dates, counts):
            records.append({"date": pd.Timestamp(d), "n_pixels": int(c)})
        ds.close()

    if not records:
        return pd.Series(dtype=float, name="n_pixels")
    df = pd.DataFrame(records)
    return df.groupby("date")["n_pixels"].sum().sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", choices=list(ZONES.keys()), default="sawtooth")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--min-peak-prob", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=7, help="Rolling window in days")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = ZONES[args.zone]
    data_root = cfg["data_root"]
    zone_name = cfg["zone_name"]

    dangers = load_dangers(data_root / cfg["danger_csv"], zone_name)
    obs = load_observations(data_root / cfg["obs_csv"], zone_name)
    detections = load_onset_detections(data_root / cfg["nc_subdir"], cfg["zone_dir"],
                                       min_confidence=args.min_confidence)

    # Build a common daily date range from the union of all data
    all_dates_list = list(dangers.index)
    if not obs.empty:
        all_dates_list.extend(obs.index)
    if not detections.empty:
        all_dates_list.extend(detections.index)
    date_range = pd.date_range(min(all_dates_list), max(all_dates_list), freq="D")

    # Reindex to full date range and compute rolling sums
    obs_daily = obs.reindex(date_range, fill_value=0)
    obs_rolling = obs_daily.rolling(args.window, center=True, min_periods=1).sum()

    det_daily = detections.reindex(date_range, fill_value=0)
    det_rolling = det_daily.rolling(args.window, center=True, min_periods=1).sum()

    # --- Plot ---
    fig, (ax_d, ax_o, ax_det) = plt.subplots(3, 1, figsize=(16, 7), sharex=True,
                                               gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1, 1]})

    # Danger rating bars
    dng = dangers.reindex(date_range)
    colors = [DANGER_COLORS.get(int(v), "white") if not pd.isna(v) else "white"
              for v in dng["danger_above_current"]]
    ax_d.bar(date_range, dng["danger_above_current"].fillna(0), color=colors, width=1.0, linewidth=0)
    ax_d.set_ylabel("Danger Rating", fontsize=10)
    ax_d.set_ylim(0.5, 5.5)
    ax_d.set_yticks([1, 2, 3, 4, 5])
    ax_d.set_yticklabels(["Low", "Mod", "Cons", "High", "Ext"], fontsize=8)
    ax_d.set_title(zone_name, fontsize=12, fontweight="bold")

    # Observations rolling
    ax_o.fill_between(date_range, obs_rolling, color="#FF7043", alpha=0.6, linewidth=0)
    ax_o.plot(date_range, obs_rolling, color="#D84315", linewidth=0.8)
    ax_o.set_ylabel(f"# Obs ({args.window}-day sum)", fontsize=10)

    # Detection pixels rolling
    ax_det.fill_between(date_range, det_rolling, color="#1565C0", alpha=0.5, linewidth=0)
    ax_det.plot(date_range, det_rolling, color="#0D47A1", linewidth=0.8)
    ax_det.set_ylabel(f"# Det. pixels ({args.window}-day sum)", fontsize=10)

    # Season shading and formatting
    min_year = date_range.year.min()
    max_year = date_range.year.max()
    for ax in [ax_d, ax_o, ax_det]:
        for yr in range(min_year, max_year + 1):
            ax.axvspan(pd.Timestamp(f"{yr}-11-01"), pd.Timestamp(f"{yr + 1}-04-30"),
                       alpha=0.06, color="blue", zorder=0)
        ax.grid(axis="x", alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    xlim_start = pd.Timestamp(cfg["xlim_start"])
    ax_det.set_xlim(xlim_start, date_range[-1])
    ax_det.tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"{zone_name}: Danger vs Observations vs CNN Detections\n"
        f"(confidence >= {args.min_confidence}, peak_prob >= {args.min_peak_prob}, {args.window}-day rolling)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    out_path = args.out or str(data_root / "figures" / cfg["out_name"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
