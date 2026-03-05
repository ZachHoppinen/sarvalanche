#!/usr/bin/env python
"""Migrate track_labels.json from old integer track indices to new deterministic IDs.

For each zone, spatially matches old tracks to new season tracks by IoU and
remaps labels that exceed a minimum IoU threshold. Unmatched labels are
reported but preserved with a `new_id: null` field so they can be re-labeled.

Usage
-----
    conda run -n sarvalanche python scripts/issw_analysis/migrate_track_labels.py \
        --labels local/issw/track_labels.json \
        --old-dir local/issw/high_low_framing_outputs/v1_high_danger_output/sarvalanche_runs \
        --new-dir local/issw/dual_tau_output \
        --min-iou 0.3 \
        --output local/issw/track_labels_migrated.json
"""
import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

log = logging.getLogger(__name__)

ZONES = [
    "Banner_Summit",
    "Galena_Summit_&_Eastern_Mtns",
    "Sawtooth_&_Western_Smoky_Mtns",
    "Soldier_&_Wood_River_Valley_Mtns",
]


def load_old_tracks(old_dir: Path, zone: str) -> gpd.GeoDataFrame:
    """Load the old per-zone tracks gpkg (any date — geometry is the same)."""
    candidates = sorted(old_dir.glob(f"{zone}*.gpkg"))
    if not candidates:
        raise FileNotFoundError(f"No old gpkg found for zone {zone} in {old_dir}")
    gdf = gpd.read_file(candidates[0])
    log.info("Loaded %d old tracks for %s from %s", len(gdf), zone, candidates[0].name)
    return gdf


def load_new_tracks(new_dir: Path, zone: str) -> gpd.GeoDataFrame:
    """Load the new season tracks gpkg for a zone."""
    gpkg_path = new_dir / zone / "season_tracks.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"New tracks not found: {gpkg_path}")
    gdf = gpd.read_file(gpkg_path)
    log.info("Loaded %d new tracks for %s", len(gdf), zone)
    return gdf


def build_matching(
    old_gdf: gpd.GeoDataFrame,
    new_gdf: gpd.GeoDataFrame,
    min_iou: float,
) -> dict[int, tuple[str, float]]:
    """Match old track indices → new track IDs by best IoU.

    Returns {old_idx: (new_id, iou)} for matches above min_iou.
    """
    # Ensure same CRS
    if old_gdf.crs != new_gdf.crs:
        new_gdf = new_gdf.to_crs(old_gdf.crs)

    # Spatial index on new tracks for fast candidate lookup
    new_sindex = new_gdf.sindex
    mapping = {}

    for old_idx, old_row in old_gdf.iterrows():
        old_geom = old_row.geometry
        if old_geom is None or old_geom.is_empty:
            continue

        # Find candidate new tracks that intersect the bounding box
        candidates = list(new_sindex.intersection(old_geom.bounds))
        if not candidates:
            continue

        best_iou = 0.0
        best_new_id = None
        for cand_idx in candidates:
            new_geom = new_gdf.geometry.iloc[cand_idx]
            if new_geom is None or new_geom.is_empty:
                continue
            intersection = old_geom.intersection(new_geom).area
            if intersection == 0:
                continue
            union = old_geom.area + new_geom.area - intersection
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_new_id = new_gdf.iloc[cand_idx]["id"]

        if best_iou >= min_iou and best_new_id is not None:
            mapping[int(old_idx)] = (best_new_id, round(best_iou, 4))

    return mapping


def migrate_labels(
    labels: dict,
    zone_mappings: dict[str, dict[int, tuple[str, float]]],
    min_iou: float,
) -> tuple[dict, dict]:
    """Remap labels using spatial matching.

    Returns (migrated_labels, stats) where stats summarises per-zone results.
    """
    migrated = {}
    stats = {}

    for zone in ZONES:
        mapping = zone_mappings.get(zone, {})
        zone_labels = {k: v for k, v in labels.items() if v["zone"] == zone}

        matched = 0
        unmatched = 0
        by_label = {}

        for key, rec in zone_labels.items():
            old_idx = rec["track_idx"]
            label = rec["label"]
            by_label.setdefault(label, {"matched": 0, "unmatched": 0})

            if old_idx in mapping:
                new_id, iou = mapping[old_idx]
                migrated[key] = {
                    **rec,
                    "new_id": new_id,
                    "old_track_idx": old_idx,
                    "migration_iou": iou,
                }
                matched += 1
                by_label[label]["matched"] += 1
            else:
                migrated[key] = {
                    **rec,
                    "new_id": None,
                    "old_track_idx": old_idx,
                    "migration_iou": None,
                }
                unmatched += 1
                by_label[label]["unmatched"] += 1

        stats[zone] = {
            "total": len(zone_labels),
            "matched": matched,
            "unmatched": unmatched,
            "by_label": by_label,
        }

    return migrated, stats


def print_report(stats: dict, min_iou: float):
    """Print a human-readable migration report."""
    label_names = {
        -1: "unsure",
        0: "HC no avy",
        1: "LC no avy",
        2: "LC avy",
        3: "HC avy",
    }

    print(f"\n{'='*70}")
    print(f"Track Label Migration Report (min IoU = {min_iou})")
    print(f"{'='*70}")

    total_all = 0
    matched_all = 0

    for zone, s in stats.items():
        total_all += s["total"]
        matched_all += s["matched"]
        pct = 100 * s["matched"] / s["total"] if s["total"] > 0 else 0
        print(f"\n{zone}: {s['matched']}/{s['total']} matched ({pct:.1f}%)")
        for label in sorted(s["by_label"].keys()):
            bl = s["by_label"][label]
            tot = bl["matched"] + bl["unmatched"]
            name = label_names.get(label, f"L{label}")
            pct_l = 100 * bl["matched"] / tot if tot > 0 else 0
            print(f"  {name:>12s} (L{label:+d}): {bl['matched']:>4d}/{tot:<4d} ({pct_l:.0f}%)")

    pct_all = 100 * matched_all / total_all if total_all > 0 else 0
    print(f"\n{'─'*70}")
    print(f"TOTAL: {matched_all}/{total_all} matched ({pct_all:.1f}%)")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Migrate track labels to new deterministic IDs")
    parser.add_argument("--labels", type=Path, required=True, help="Input track_labels.json")
    parser.add_argument("--old-dir", type=Path, required=True, help="Dir with old per-zone gpkgs")
    parser.add_argument("--new-dir", type=Path, required=True, help="Dir with new season track dirs")
    parser.add_argument("--min-iou", type=float, default=0.3, help="Minimum IoU to accept a match")
    parser.add_argument("--output", type=Path, required=True, help="Output migrated labels JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load labels
    with open(args.labels) as f:
        labels = json.load(f)
    log.info("Loaded %d labels", len(labels))

    # Build spatial matching per zone
    zone_mappings = {}
    for zone in ZONES:
        try:
            old_gdf = load_old_tracks(args.old_dir, zone)
            new_gdf = load_new_tracks(args.new_dir, zone)
        except FileNotFoundError as e:
            log.warning("Skipping %s: %s", zone, e)
            continue

        mapping = build_matching(old_gdf, new_gdf, args.min_iou)
        zone_mappings[zone] = mapping
        log.info("%s: %d/%d old tracks matched", zone, len(mapping), len(old_gdf))

    # Migrate
    migrated, stats = migrate_labels(labels, zone_mappings, args.min_iou)

    # Report
    print_report(stats, args.min_iou)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(migrated, f, indent=2, default=str)
    log.info("Wrote migrated labels to %s", args.output)

    # Summary of unmatched for re-labeling
    unmatched = {k: v for k, v in migrated.items() if v["new_id"] is None}
    if unmatched:
        log.info("%d labels need re-labeling (new_id=null)", len(unmatched))


if __name__ == "__main__":
    main()
