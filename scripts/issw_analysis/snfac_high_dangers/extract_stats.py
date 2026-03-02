"""
Extract summary statistics from all high-danger .nc runs and SNOTEL data.

Produces a single Parquet file (stats.parquet) consumed by the companion notebook.

Usage:
    conda run -n sarvalanche python scripts/issw_analysis/snfac_high_dangers/extract_stats.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────────
RUNS_DIR = Path(
    "/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs"
)
SNOTEL_DIR = Path("/Users/zmhoppinen/Documents/sarvalanche/local/issw/snotels")
OUT_DIR = Path(__file__).resolve().parent
STATS_OUT = OUT_DIR / "stats.parquet"
SNOTEL_OUT = OUT_DIR / "snotel_combined.parquet"


# ── SNOTEL parsing ───────────────────────────────────────────────────────────
def load_snotel(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=6, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    col_map = {}
    for c in df.columns:
        if "WTEQ" in c:
            col_map[c] = "SWE_in"
        elif "TMAX" in c:
            col_map[c] = "Tmax_C"
        elif "TMIN" in c:
            col_map[c] = "Tmin_C"
        elif "TAVG" in c:
            col_map[c] = "Tavg_C"
        elif "TOBS" in c:
            col_map[c] = "Tobs_C"
        elif "SNWD" in c:
            col_map[c] = "SnowDepth_in"
        elif "PREC" in c:
            col_map[c] = "Precip_in"
    df = df.rename(columns=col_map)
    for c in ["Tmax_C", "Tmin_C", "Tavg_C", "Tobs_C"]:
        if c in df.columns:
            df[c] = df[c].replace(-99.9, np.nan)
    return df.set_index("Date")


def snotel_window_stats(snotel: pd.DataFrame, center_date: str, days: int = 14):
    """Compute weather stats in a window around center_date."""
    center = pd.Timestamp(center_date)
    win = snotel.loc[center - pd.Timedelta(days=days) : center + pd.Timedelta(days=days)]
    if win.empty:
        return {}
    pre = snotel.loc[center - pd.Timedelta(days=days) : center]
    post = snotel.loc[center : center + pd.Timedelta(days=days)]

    result = {}
    # Temperature stats in window
    result["tavg_window_mean"] = win["Tavg_C"].mean() if "Tavg_C" in win else np.nan
    result["tmax_window_max"] = win["Tmax_C"].max() if "Tmax_C" in win else np.nan
    result["tmin_window_min"] = win["Tmin_C"].min() if "Tmin_C" in win else np.nan
    result["days_tmax_above_0"] = int((win["Tmax_C"] > 0).sum()) if "Tmax_C" in win else 0
    result["days_tmax_above_m5"] = (
        int((win["Tmax_C"] > -5).sum()) if "Tmax_C" in win else 0
    )
    # SWE change across window
    if "SWE_in" in win.columns:
        swe_vals = win["SWE_in"].dropna()
        if len(swe_vals) >= 2:
            result["swe_at_date"] = float(
                snotel.loc[center:center, "SWE_in"].iloc[0]
                if center in snotel.index
                else swe_vals.iloc[len(swe_vals) // 2]
            )
            result["swe_change_pre"] = float(swe_vals.iloc[len(pre) - 1] - swe_vals.iloc[0])
            result["swe_change_post"] = float(swe_vals.iloc[-1] - swe_vals.iloc[len(pre) - 1])
            result["max_daily_swe_change"] = float(swe_vals.diff().max())
        else:
            result["swe_at_date"] = np.nan
            result["swe_change_pre"] = np.nan
            result["swe_change_post"] = np.nan
            result["max_daily_swe_change"] = np.nan
    # Snow depth change
    if "SnowDepth_in" in win.columns:
        sd = win["SnowDepth_in"].dropna()
        if len(sd) >= 2:
            result["depth_at_date"] = float(
                snotel.loc[center:center, "SnowDepth_in"].iloc[0]
                if center in snotel.index
                else sd.iloc[len(sd) // 2]
            )
            result["depth_change_window"] = float(sd.iloc[-1] - sd.iloc[0])
            result["max_daily_depth_change"] = float(sd.diff().max())
        else:
            result["depth_at_date"] = np.nan
            result["depth_change_window"] = np.nan
            result["max_daily_depth_change"] = np.nan
    return result


# ── NC extraction ────────────────────────────────────────────────────────────
def parse_nc_stem(stem: str):
    m = re.search(r"^(.+)_(\d{4}-\d{2}-\d{2})$", stem)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def pct(arr, q):
    return float(np.nanpercentile(arr, q))


def extract_nc_stats(nc_path: Path) -> dict:
    """Extract comprehensive statistics from a single .nc file."""
    zone, date = parse_nc_stem(nc_path.stem)
    ds = xr.open_dataset(nc_path)
    row = {"zone": zone, "date": date, "nc_path": str(nc_path)}

    # ── Scene-level distance stats ──
    for var, prefix in [
        ("d_empirical", "d_emp"),
        ("distance_mahalanobis", "d_mah"),
        ("combined_distance", "d_comb"),
    ]:
        if var not in ds:
            continue
        vals = ds[var].values.ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        row[f"{prefix}_mean"] = float(vals.mean())
        row[f"{prefix}_std"] = float(vals.std())
        row[f"{prefix}_median"] = float(np.median(vals))
        row[f"{prefix}_p05"] = pct(vals, 5)
        row[f"{prefix}_p10"] = pct(vals, 10)
        row[f"{prefix}_p25"] = pct(vals, 25)
        row[f"{prefix}_p75"] = pct(vals, 75)
        row[f"{prefix}_p90"] = pct(vals, 90)
        row[f"{prefix}_p95"] = pct(vals, 95)
        row[f"{prefix}_skew"] = float(
            ((vals - vals.mean()) ** 3).mean() / (vals.std() ** 3 + 1e-12)
        )
        # Fraction of pixels above various thresholds
        if prefix == "d_emp":
            for t in [0.5, 1.0, 1.5, 2.0]:
                row[f"{prefix}_frac_gt_{t}"] = float((vals > t).mean())
            for t in [-0.5, -1.0, -1.5, -2.0]:
                row[f"{prefix}_frac_lt_{t}"] = float((vals < t).mean())

    # ── Per-orbit empirical distances ──
    orbit_emp_means = {}
    orbit_ml_means = {}
    for orbit in ["71", "93", "173"]:
        for pol in ["VV", "VH"]:
            emp_var = f"d_{orbit}_{pol}_empirical"
            ml_var = f"d_{orbit}_{pol}_ml"
            sigma_var = f"sigma_{orbit}_{pol}_ml"
            if emp_var in ds:
                vals = ds[emp_var].values.ravel()
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    row[f"d_{orbit}_{pol}_emp_mean"] = float(vals.mean())
                    row[f"d_{orbit}_{pol}_emp_std"] = float(vals.std())
                    row[f"d_{orbit}_{pol}_emp_p10"] = pct(vals, 10)
                    row[f"d_{orbit}_{pol}_emp_p90"] = pct(vals, 90)
                    row[f"d_{orbit}_{pol}_emp_nan_frac"] = float(
                        np.isnan(ds[emp_var].values).mean()
                    )
                    if pol == "VV":
                        orbit_emp_means[orbit] = float(vals.mean())
            if ml_var in ds:
                vals = ds[ml_var].values.ravel()
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    row[f"d_{orbit}_{pol}_ml_mean"] = float(vals.mean())
                    row[f"d_{orbit}_{pol}_ml_std"] = float(vals.std())
                    if pol == "VV":
                        orbit_ml_means[orbit] = float(vals.mean())
            if sigma_var in ds:
                vals = ds[sigma_var].values.ravel()
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    row[f"sigma_{orbit}_{pol}_ml_mean"] = float(vals.mean())

    # Orbit spread (max - min across orbit VV means)
    if len(orbit_emp_means) >= 2:
        vals = list(orbit_emp_means.values())
        row["orbit_emp_VV_spread"] = max(vals) - min(vals)
        row["orbit_emp_VV_min_orbit"] = min(orbit_emp_means, key=orbit_emp_means.get)
        row["orbit_emp_VV_max_orbit"] = max(orbit_emp_means, key=orbit_emp_means.get)
    if len(orbit_ml_means) >= 2:
        vals = list(orbit_ml_means.values())
        row["orbit_ml_VV_spread"] = max(vals) - min(vals)

    # ── VV vs VH comparison ──
    for orbit in ["71", "93", "173"]:
        vv_key = f"d_{orbit}_VV_emp_mean"
        vh_key = f"d_{orbit}_VH_emp_mean"
        if vv_key in row and vh_key in row:
            row[f"d_{orbit}_VV_VH_diff"] = row[vv_key] - row[vh_key]

    # ── Detection stats ──
    for var, prefix in [
        ("detections", "det"),
        ("p_pixelwise", "p_pw"),
        ("p_empirical", "p_emp"),
        ("unmasked_p_target", "p_tgt"),
    ]:
        if var not in ds:
            continue
        vals = ds[var].values.ravel()
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        row[f"{prefix}_count"] = int((valid > 0.5).sum()) if var != "p_empirical" else int(valid.sum())
        row[f"{prefix}_frac"] = float(valid.mean())
        row[f"{prefix}_n_valid"] = len(valid)

    # ── Static probability stats (how much of scene is feasible) ──
    for var in ["p_fcf", "p_runout", "p_slope"]:
        if var in ds:
            vals = ds[var].values.ravel()
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                row[f"{var}_mean"] = float(valid.mean())
                row[f"{var}_frac_gt05"] = float((valid > 0.5).mean())

    # ── Elevation stats of detections ──
    if "detections" in ds and "dem" in ds:
        det_mask = ds["detections"].values > 0.5
        dem_vals = ds["dem"].values
        if det_mask.any():
            det_elev = dem_vals[det_mask]
            det_elev = det_elev[~np.isnan(det_elev)]
            if len(det_elev) > 0:
                row["det_elev_mean"] = float(det_elev.mean())
                row["det_elev_min"] = float(det_elev.min())
                row["det_elev_max"] = float(det_elev.max())
                row["det_elev_std"] = float(det_elev.std())
    if "detections" in ds and "slope" in ds:
        det_mask = ds["detections"].values > 0.5
        slope_vals = ds["slope"].values
        if det_mask.any():
            det_slope = slope_vals[det_mask]
            det_slope = det_slope[~np.isnan(det_slope)]
            if len(det_slope) > 0:
                row["det_slope_mean_deg"] = float(np.rad2deg(det_slope.mean()))

    # ── Temporal weight info (season proxy) ──
    if "w_temporal" in ds:
        times = pd.DatetimeIndex(ds["w_temporal"].time.values)
        row["sar_start"] = str(times[0].date())
        row["sar_end"] = str(times[-1].date())
        row["n_sar_scenes"] = len(times)
        center = pd.Timestamp(date)
        row["month"] = center.month
        row["day_of_winter"] = (center - pd.Timestamp(f"{center.year - (1 if center.month < 9 else 0)}-10-01")).days

    ds.close()
    return row


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load all SNOTEL water years
    snotel_frames = []
    for csv in sorted(SNOTEL_DIR.glob("490_STAND_WATERYEAR=*.csv")):
        snotel_frames.append(load_snotel(csv))
    snotel_all = pd.concat(snotel_frames)
    snotel_all = snotel_all[~snotel_all.index.duplicated(keep="last")]
    snotel_all.sort_index(inplace=True)
    snotel_all.to_parquet(SNOTEL_OUT)
    print(f"SNOTEL: {len(snotel_all)} days, {snotel_all.index[0].date()} to {snotel_all.index[-1].date()}")

    # Extract stats from every .nc
    nc_files = sorted(RUNS_DIR.glob("*.nc"))
    print(f"Processing {len(nc_files)} .nc files...")
    rows = []
    for i, nc_path in enumerate(nc_files):
        print(f"  [{i+1}/{len(nc_files)}] {nc_path.stem}")
        row = extract_nc_stats(nc_path)
        # Add SNOTEL context
        snotel_stats = snotel_window_stats(snotel_all, row["date"], days=14)
        row.update(snotel_stats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(STATS_OUT, index=False)
    print(f"\nSaved {len(df)} rows to {STATS_OUT}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nZones: {df['zone'].nunique()}")
    print(f"Dates: {df['date'].nunique()}")
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
