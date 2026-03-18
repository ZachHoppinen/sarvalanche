"""Compare experiment models against avalanche observations on Sawtooth 2024-2025.

Runs the same spatial comparison as compare_confidence_vs_baseline.py but across
all experiment models + the existing confidence-weighted baseline.
"""

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

# ── Load all models ──────────────────────────────────────────────────
base_dir = Path("local/issw/netcdfs/Sawtooth_&_Western_Smoky_Mtns")

models = {
    "conf_baseline": base_dir / "v2_season_inference_2024-2025_confidence/season_v2_debris_probabilities.nc",
    "pooled_pt_ft": base_dir / "v2_season_inference_2024-2025_exp3_pretrain_finetune/season_v2_debris_probabilities.nc",
    "pair_pt_ft": base_dir / "v2_season_inference_2024-2025_pair_pretrain_ft/season_v2_debris_probabilities.nc",
    "combo_pt_ft": base_dir / "v2_season_inference_2024-2025_combo_pt_ft/season_v2_debris_probabilities.nc",
}

prob_datasets = {}
for name, path in models.items():
    if path.exists():
        prob_datasets[name] = xr.open_dataset(path)
        print(f"  Loaded {name}: {path.name}")
    else:
        print(f"  MISSING {name}: {path}")

# Use first dataset for grid info
ref_ds = next(iter(prob_datasets.values()))
crs = ref_ds.rio.crs or "EPSG:4326"
x = ref_ds.x.values
y = ref_ds.y.values
H, W = len(y), len(x)
dx = abs(float(x[1] - x[0]))
dy = abs(float(y[1] - y[0]))
transform = from_bounds(
    float(x.min()) - dx / 2, float(y.min()) - dy / 2,
    float(x.max()) + dx / 2, float(y.max()) + dy / 2, W, H,
)


def check_spatial(row, prob_ds):
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

    cnn_times = pd.DatetimeIndex(prob_ds.time.values)
    time_diffs = np.abs(cnn_times - obs_date)
    ci = time_diffs.argmin()
    if time_diffs[ci].days > 6:
        return None

    prob_map = prob_ds["debris_probability"].isel(time=ci).values
    path_probs = prob_map[path_mask]

    return {
        "mean_path": float(np.mean(path_probs)),
        "max_path": float(np.max(path_probs)),
        "n_above_50": int((path_probs > 0.5).sum()),
        "n_above_20": int((path_probs > 0.2).sum()),
        "n_path_px": int(path_mask.sum()),
    }


# ── Run comparison ───────────────────────────────────────────────────
results = []
for _, row in obs_2425.iterrows():
    all_results = {}
    skip = False
    for name, ds in prob_datasets.items():
        r = check_spatial(row, ds)
        if r is None:
            skip = True
            break
        all_results[name] = r

    if skip:
        continue

    entry = {
        "obs_id": row["id"],
        "date": row["date"].strftime("%Y-%m-%d"),
        "location": row.get("location_name", ""),
        "d_size": row.get("d_size", np.nan),
        "month": row["date"].month,
        "n_path_px": all_results[next(iter(all_results))]["n_path_px"],
    }
    for name, r in all_results.items():
        entry[f"{name}_n20"] = r["n_above_20"]
        entry[f"{name}_n50"] = r["n_above_50"]
        entry[f"{name}_mean"] = r["mean_path"]
        entry[f"{name}_max"] = r["max_path"]

    results.append(entry)

df = pd.DataFrame(results)
n = len(df)
print(f"\nMatched observations: {n}")

# ── Overall summary ──────────────────────────────────────────────────
model_names = list(prob_datasets.keys())

print(f"\n{'='*120}")
print("OVERALL COMPARISON")
print(f"{'='*120}")
hdr = f"{'Model':>22s}  {'n>0.2 (obs)':>12s}  {'n>0.5 (obs)':>12s}  {'Total px>0.2':>12s}  {'Total px>0.5':>12s}  {'Mean prob':>10s}"
print(hdr)
print("-" * len(hdr))

for name in model_names:
    n20_col = f"{name}_n20"
    n50_col = f"{name}_n50"
    mean_col = f"{name}_mean"
    any_20 = (df[n20_col] >= 1).sum()
    any_50 = (df[n50_col] >= 1).sum()
    total_20 = df[n20_col].sum()
    total_50 = df[n50_col].sum()
    mean_p = df[mean_col].mean()
    print(f"  {name:>20s}  {any_20:>5d}/{n:<5d}  {any_50:>5d}/{n:<5d}  {total_20:>12d}  {total_50:>12d}  {mean_p:>10.4f}")

# ── By D-size ────────────────────────────────────────────────────────
print(f"\n{'='*120}")
print("BY D-SIZE")
print(f"{'='*120}")

for d in sorted(df["d_size"].dropna().unique()):
    sub = df[df["d_size"] == d]
    nd = len(sub)
    print(f"\n  D{d:.1f} ({nd} obs):")
    for name in model_names:
        any_20 = (sub[f"{name}_n20"] >= 1).sum()
        any_50 = (sub[f"{name}_n50"] >= 1).sum()
        total_20 = sub[f"{name}_n20"].sum()
        mean_p = sub[f"{name}_mean"].mean()
        print(f"    {name:>20s}:  detected(>0.2)={any_20}/{nd}  detected(>0.5)={any_50}/{nd}  total_px={total_20:>6d}  mean={mean_p:.4f}")

# ── By month ─────────────────────────────────────────────────────────
print(f"\n{'='*120}")
print("BY MONTH")
print(f"{'='*120}")
month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
for m in [11, 12, 1, 2, 3, 4]:
    sub = df[df["month"] == m]
    if len(sub) == 0:
        continue
    nd = len(sub)
    print(f"\n  {month_names[m]} ({nd} obs):")
    for name in model_names:
        any_20 = (sub[f"{name}_n20"] >= 1).sum()
        any_50 = (sub[f"{name}_n50"] >= 1).sum()
        total_20 = sub[f"{name}_n20"].sum()
        mean_p = sub[f"{name}_mean"].mean()
        print(f"    {name:>20s}:  detected(>0.2)={any_20}/{nd}  detected(>0.5)={any_50}/{nd}  total_px={total_20:>6d}  mean={mean_p:.4f}")

# ── D >= 2.5 detail ──────────────────────────────────────────────────
print(f"\n{'='*120}")
print("DETAIL: D >= 2.5 OBSERVATIONS")
print(f"{'='*120}")
big = df[df["d_size"] >= 2.5].sort_values(["d_size", "date"], ascending=[False, True])
for _, r in big.iterrows():
    print(f"\n  {r['date']} D{r['d_size']:.1f} {str(r['location'])[:35]}")
    for name in model_names:
        print(f"    {name:>20s}: n>0.2={r[f'{name}_n20']:>5d}  n>0.5={r[f'{name}_n50']:>5d}  max={r[f'{name}_max']:.3f}")

# ── Save ─────────────────────────────────────────────────────────────
out = Path("local/experiments/auto_vs_human/obs_comparison.csv")
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
