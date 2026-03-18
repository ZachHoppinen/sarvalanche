"""Compare baseline CNN vs confidence-weighted CNN on 2024-2025 Sawtooth.

Spatial comparison against FlowPy avalanche paths, broken down by:
  - D-size (destructive size)
  - Observation quality (trigger type, observer type, date certainty)
  - Month
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr
from rasterio.transform import from_bounds

# ── Load observations ────────────────────────────────────────────────
obs = pd.read_csv("local/issw/snfac_obs_2021_2025.csv")
obs["date"] = pd.to_datetime(obs["date"])
mask = (obs["date"] >= "2024-11-01") & (obs["date"] <= "2025-04-30")
saw_mask = obs["zone_name"] == "Sawtooth & Western Smoky Mtns"
obs_2425 = obs[mask & saw_mask].copy().reset_index(drop=True)
print(f"Total 2024-2025 Sawtooth obs: {len(obs_2425)}")

# ── Load FlowPy paths ───────────────────────────────────────────────
paths = gpd.read_file("local/issw/observations/all_flowpy_paths.gpkg")
paths["date"] = pd.to_datetime(paths["date"])

# ── Load inference results ───────────────────────────────────────────
base_dir = Path("local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns")
prob_old = xr.open_dataset(base_dir / "v2_season_inference_2024-2025/season_v2_debris_probabilities.nc")
prob_new = xr.open_dataset(base_dir / "v2_season_inference_2024-2025_confidence/season_v2_debris_probabilities.nc")

# Inference NC may lack CRS — infer from coordinate ranges
_crs = prob_old.rio.crs
if _crs is None:
    # Geographic coords (lat/lon)
    crs = "EPSG:4326"
else:
    crs = _crs
x = prob_old.x.values
y = prob_old.y.values
H, W = len(y), len(x)
dx = abs(float(x[1] - x[0]))
dy = abs(float(y[1] - y[0]))
transform = from_bounds(
    float(x.min()) - dx / 2, float(y.min()) - dy / 2,
    float(x.max()) + dx / 2, float(y.max()) + dy / 2, W, H,
)


# ── Observation quality classification ───────────────────────────────
def classify_obs_quality(row):
    """Classify observation quality into high/medium/low.

    High: professional observer, date known, has location point
    Medium: has d_size, date known
    Low: everything else
    """
    is_pro = row.get("observer_type") in ("professional", "forecaster")
    date_known = row.get("date_known", False) is True or row.get("date_known") == "True"
    has_d_size = pd.notna(row.get("d_size"))
    has_location = pd.notna(row.get("location_point")) and row.get("location_point") != ""

    if is_pro and date_known and has_location:
        return "high"
    elif has_d_size and date_known:
        return "medium"
    else:
        return "low"


# ── Spatial detection check ──────────────────────────────────────────
def check_spatial(row, prob_ds):
    """Check if a CNN detects debris along the FlowPy path for this obs."""
    obs_date = row["date"]
    obs_id = row["id"]

    date_paths = paths[
        (paths["date"].dt.date == obs_date.date()) & (paths["obs_id"] == obs_id)
    ]
    if len(date_paths) == 0:
        return None

    path_gdf = date_paths.to_crs(crs)
    geom = path_gdf.geometry.values[0]
    if geom is None or geom.is_empty:
        return None

    try:
        path_mask = ~rasterio.features.geometry_mask(
            [geom], out_shape=(H, W), transform=transform, all_touched=True,
        )
    except Exception:
        return None

    if path_mask.sum() == 0:
        return None

    # Find closest CNN time step (within 6 days)
    cnn_times = pd.DatetimeIndex(prob_ds.time.values)
    time_diffs = np.abs(cnn_times - obs_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 6:
        return None

    prob_map = prob_ds["debris_probability"].isel(time=ci).values
    path_probs = prob_map[path_mask]

    # Background: everything outside path with nonzero prob
    bg_mask = ~path_mask & (prob_map > 0)
    bg_probs = prob_map[bg_mask] if bg_mask.any() else np.array([0.0])

    # Metrics
    mean_path = float(np.mean(path_probs))
    max_path = float(np.max(path_probs))
    p95_path = float(np.percentile(path_probs, 95))
    p50_path = float(np.percentile(path_probs, 50))
    p95_bg = float(np.percentile(bg_probs, 95))
    n_above_50 = int((path_probs > 0.5).sum())
    n_above_30 = int((path_probs > 0.3).sum())
    frac_above_30 = float((path_probs > 0.3).mean())
    n_path_px = int(path_mask.sum())

    # Pixel counts at thresholds
    n_above_20 = int((path_probs > 0.2).sum())
    frac_above_20 = float((path_probs > 0.2).mean())

    return {
        "mean_path": mean_path,
        "max_path": max_path,
        "p95_path": p95_path,
        "p50_path": p50_path,
        "n_above_50": n_above_50,
        "n_above_30": n_above_30,
        "n_above_20": n_above_20,
        "frac_above_30": frac_above_30,
        "frac_above_20": frac_above_20,
        "n_path_px": n_path_px,
        "cnn_date": str(cnn_times[ci])[:10],
    }


# ── Run comparison ───────────────────────────────────────────────────
results = []
for _, row in obs_2425.iterrows():
    r_old = check_spatial(row, prob_old)
    r_new = check_spatial(row, prob_new)
    if r_old is None or r_new is None:
        continue

    quality = classify_obs_quality(row)
    d_size = row.get("d_size", np.nan)
    month = row["date"].month

    results.append({
        "obs_id": row["id"],
        "date": row["date"].strftime("%Y-%m-%d"),
        "location": row.get("location_name", ""),
        "d_size": d_size,
        "r_size": row.get("r_size", np.nan),
        "quality": quality,
        "month": month,
        "trigger": row.get("trigger", ""),
        "avy_type": row.get("avalanche_type", ""),
        # Old model
        "old_mean": r_old["mean_path"],
        "old_max": r_old["max_path"],
        "old_n50": r_old["n_above_50"],
        "old_n30": r_old["n_above_30"],
        "old_n20": r_old["n_above_20"],
        # New model
        "new_mean": r_new["mean_path"],
        "new_max": r_new["max_path"],
        "new_n50": r_new["n_above_50"],
        "new_n30": r_new["n_above_30"],
        "new_n20": r_new["n_above_20"],
        "n_path_px": r_new["n_path_px"],
    })

df = pd.DataFrame(results)
n = len(df)
print(f"\nMatched observations with FlowPy paths: {n} / {len(obs_2425)}")

def _print_table(df, groupby_col, group_labels, title):
    """Print comparison table for a groupby column."""
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    hdr = f"{'Group':>8s} {'n':>4s}  {'Old n>0.2':>10s}  {'New n>0.2':>10s}  {'Old n>0.5':>10s}  {'New n>0.5':>10s}  {'Old mean':>9s}  {'New mean':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for key, label in group_labels:
        sub = df[df[groupby_col] == key] if key is not None else df[df[groupby_col].isna()]
        nd = len(sub)
        if nd == 0:
            continue
        # Medians of per-obs pixel counts
        old_n20 = sub["old_n20"].median()
        new_n20 = sub["new_n20"].median()
        old_n50 = sub["old_n50"].median()
        new_n50 = sub["new_n50"].median()
        old_mean = sub["old_mean"].mean()
        new_mean = sub["new_mean"].mean()
        # How many obs have ANY detection (n>0.2 >= 1)?
        old_any = (sub["old_n20"] >= 1).sum()
        new_any = (sub["new_n20"] >= 1).sum()
        print(
            f"  {label:>6s} {nd:>4d}"
            f"  {old_n20:>5.0f} ({old_any:>3d}/{nd})"
            f"  {new_n20:>5.0f} ({new_any:>3d}/{nd})"
            f"  {old_n50:>5.0f} ({(sub['old_n50']>=1).sum():>3d}/{nd})"
            f"  {new_n50:>5.0f} ({(sub['new_n50']>=1).sum():>3d}/{nd})"
            f"  {old_mean:>9.4f}"
            f"  {new_mean:>9.4f}"
        )


# ── Overall summary ──────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"OVERALL — pixel counts in avalanche paths (median across obs)")
print(f"{'='*100}")
for thresh, old_col, new_col in [("0.2", "old_n20", "new_n20"), ("0.3", "old_n30", "new_n30"), ("0.5", "old_n50", "new_n50")]:
    old_med = df[old_col].median()
    new_med = df[new_col].median()
    old_any = (df[old_col] >= 1).sum()
    new_any = (df[new_col] >= 1).sum()
    old_sum = df[old_col].sum()
    new_sum = df[new_col].sum()
    print(f"  n > {thresh}:  Old median={old_med:>5.0f}  New median={new_med:>5.0f}  |  Old any={old_any:>3d}/{n}  New any={new_any:>3d}/{n}  |  Old total={old_sum:>8d}  New total={new_sum:>8d}")

print(f"\n  Mean path prob:  Old={df['old_mean'].mean():.4f}  New={df['new_mean'].mean():.4f}  ({(df['new_mean'].mean()/max(df['old_mean'].mean(),1e-8)-1)*100:+.1f}%)")

# ── By D-size ────────────────────────────────────────────────────────
d_labels = [(d, f"D{d:.1f}") for d in sorted(df["d_size"].dropna().unique())]
if df["d_size"].isna().any():
    d_labels.append((None, "NaN"))
_print_table(df, "d_size", d_labels, "BY DESTRUCTIVE SIZE  (median n>0.2 and n>0.5 per obs, plus count with any detection)")

# ── By observation quality ───────────────────────────────────────────
q_labels = [(q, q) for q in ["high", "medium", "low"]]
_print_table(df, "quality", q_labels, "BY OBSERVATION QUALITY")

# ── By month ─────────────────────────────────────────────────────────
month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
m_labels = [(m, month_names[m]) for m in [11, 12, 1, 2, 3, 4]]
_print_table(df, "month", m_labels, "BY MONTH")

# ── By avalanche type ────────────────────────────────────────────────
at_labels = [(t, t) for t in sorted(df["avy_type"].dropna().unique()) if t]
_print_table(df, "avy_type", at_labels, "BY AVALANCHE TYPE")

# ── Detailed per-obs for D >= 2.5 ───────────────────────────────────
print(f"\n{'='*100}")
print(f"DETAIL: ALL D >= 2.5 OBSERVATIONS")
print(f"{'='*100}")
big = df[df["d_size"] >= 2.5].sort_values(["d_size", "date"], ascending=[False, True])
print(f"{'Date':>12s} {'D':>4s} {'Loc':>28s} {'Q':>4s} {'px':>5s} {'old_n20':>7s} {'new_n20':>7s} {'old_n50':>7s} {'new_n50':>7s} {'old_max':>7s} {'new_max':>7s}")
for _, r in big.iterrows():
    print(
        f"{r['date']:>12s} {r['d_size']:>4.1f} {str(r['location'])[:28]:>28s} {r['quality'][:4]:>4s}"
        f" {r['n_path_px']:>5d} {r['old_n20']:>7d} {r['new_n20']:>7d} {r['old_n50']:>7d} {r['new_n50']:>7d}"
        f" {r['old_max']:>7.3f} {r['new_max']:>7.3f}"
    )

# ── Save full results CSV ────────────────────────────────────────────
out_csv = Path("local/issw/v2_confidence_vs_baseline_comparison.csv")
df.to_csv(out_csv, index=False)
print(f"\nFull results saved to {out_csv}")
