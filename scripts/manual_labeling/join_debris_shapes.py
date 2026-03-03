#!/usr/bin/env python
"""
Auto-populate debris shape attributes by spatial join with track polygons.

Draw debris polygons in QGIS (no attributes needed), save to a gpkg, then
run this script to assign zone, date, track_idx, and key columns via spatial
join with the zone track gpkg files.

Usage
-----
    # Join shapes drawn for a specific date:
    conda run -n sarvalanche python scripts/join_debris_shapes.py \
        new_shapes.gpkg 2025-02-04

    # Append to the main debris_shapes.gpkg (default):
    conda run -n sarvalanche python scripts/join_debris_shapes.py \
        new_shapes.gpkg 2025-02-04

    # Write to a separate output file instead:
    conda run -n sarvalanche python scripts/join_debris_shapes.py \
        new_shapes.gpkg 2025-02-04 -o labeled_output.gpkg

    # Specify which runs dir to find zone gpkgs:
    conda run -n sarvalanche python scripts/join_debris_shapes.py \
        new_shapes.gpkg 2025-02-04 --runs-dir path/to/sarvalanche_runs
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

RUNS_DIRS = [
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/high_danger_output/sarvalanche_runs'),
    Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/low_danger_output/sarvalanche_runs'),
]
MAIN_SHAPES = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/debris_shapes.gpkg')
LABELS_PATH = Path('/Users/zmhoppinen/Documents/sarvalanche/local/issw/track_labels.json')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path,
                        help='GeoPackage with drawn debris polygons (no attributes needed)')
    parser.add_argument('date', type=str,
                        help='Avalanche date for these shapes (YYYY-MM-DD)')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help=f'Output gpkg (default: append to {MAIN_SHAPES})')
    parser.add_argument('--runs-dir', type=Path, nargs='+', default=None,
                        help='Runs dir(s) containing zone .gpkg files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print results without writing')
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    runs_dirs = [p.resolve() for p in args.runs_dir] if args.runs_dir else RUNS_DIRS
    date = args.date
    output_path = (args.output or MAIN_SHAPES).resolve()

    # Load drawn shapes
    new_shapes = gpd.read_file(input_path)
    if new_shapes.empty:
        print("No shapes found in input file.")
        sys.exit(0)
    print(f"Loaded {len(new_shapes)} drawn shapes from {input_path.name}")

    # Collect all zone gpkg files
    zone_gdfs: dict[str, gpd.GeoDataFrame] = {}
    for runs_dir in runs_dirs:
        for gpkg_path in sorted(runs_dir.glob('*.gpkg')):
            # Extract zone name (strip trailing _YYYY-MM-DD)
            import re
            zone = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', gpkg_path.stem)
            if zone not in zone_gdfs:
                gdf = gpd.read_file(gpkg_path)
                gdf['_zone'] = zone
                gdf['_track_idx'] = gdf.index
                zone_gdfs[zone] = (gdf, gpkg_path)

    if not zone_gdfs:
        print("ERROR: No zone .gpkg files found in runs dirs", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(zone_gdfs)} zones: {', '.join(sorted(zone_gdfs.keys()))}")

    # Combine all track polygons for spatial join
    all_tracks = []
    for zone, (gdf, _) in zone_gdfs.items():
        all_tracks.append(gdf)
    tracks = gpd.GeoDataFrame(pd.concat(all_tracks, ignore_index=True))

    # Reproject new shapes to match tracks if needed
    if new_shapes.crs != tracks.crs:
        new_shapes = new_shapes.to_crs(tracks.crs)

    # Fix invalid geometries before spatial join
    new_shapes['geometry'] = new_shapes.geometry.buffer(0)
    tracks['geometry'] = tracks.geometry.buffer(0)

    # Spatial join: assign shape to ALL tracks with >= 15% overlap
    MIN_OVERLAP_FRAC = 0.15
    results = []
    for i, shape in new_shapes.iterrows():
        shape_geom = shape.geometry
        if shape_geom is None or shape_geom.is_empty:
            print(f"  WARNING: shape {i} is empty after cleaning, skipping")
            continue
        # Find all intersecting tracks
        candidates = tracks[tracks.intersects(shape_geom)].copy()
        if candidates.empty:
            print(f"  WARNING: shape {i} doesn't overlap any track, skipping")
            continue

        # Compute overlap as fraction of each track's area covered by the shape
        candidates['_overlap_area'] = candidates.geometry.intersection(shape_geom).area
        candidates['_overlap_frac'] = candidates['_overlap_area'] / candidates.geometry.area

        # Keep all tracks with meaningful overlap (>= 15% of track area)
        hits = candidates[candidates['_overlap_frac'] >= MIN_OVERLAP_FRAC]
        if hits.empty:
            # Fall back to best overlap if none meet threshold
            best = candidates.loc[candidates['_overlap_area'].idxmax()]
            hits = candidates.loc[[best.name]]

        for _, track in hits.iterrows():
            zone = track['_zone']
            track_idx = int(track['_track_idx'])
            key = f"{zone}|{date}|{track_idx}"
            results.append({
                'key': key,
                'zone': zone,
                'date': date,
                'track_idx': track_idx,
                'geometry': shape_geom,
            })

    if not results:
        print("No shapes matched any tracks.")
        sys.exit(0)

    joined = gpd.GeoDataFrame(results, crs=tracks.crs)
    print(f"\nMatched {len(joined)} shapes:")
    for _, row in joined.iterrows():
        print(f"  {row['key']}")

    # Always update track labels for all matched keys (even if shapes are duplicates)
    if not args.dry_run:
        if LABELS_PATH.exists():
            with open(LABELS_PATH) as f:
                labels = json.load(f)
        else:
            labels = {}

        matched_keys = set(joined['key'].tolist())
        n_new_labels = 0
        for key in matched_keys:
            if key not in labels:
                parts = key.split('|')
                labels[key] = {
                    'label': 3,
                    'zone': parts[0],
                    'date': parts[1],
                    'track_idx': int(parts[2]),
                }
                n_new_labels += 1

        with open(LABELS_PATH, 'w') as f:
            json.dump(labels, f, indent=2)
        print(f"Added {n_new_labels} new track labels (label=3) to {LABELS_PATH.name} "
              f"({len(labels)} total)")

    # Skip duplicates with existing shapes
    existing_keys: set[str] = set()
    if output_path.exists():
        existing = gpd.read_file(output_path)
        existing_keys = set(existing['key'].tolist())
        dupes = joined[joined['key'].isin(existing_keys)]
        if not dupes.empty:
            print(f"\n  Skipping {len(dupes)} shapes already in {output_path.name}:")
            for k in dupes['key'].tolist():
                print(f"    {k}")
            joined = joined[~joined['key'].isin(existing_keys)].copy()

    if joined.empty:
        print("\nNo new shapes to add (all duplicates).")
        return

    print(f"\n{len(joined)} new shapes to add:")
    for _, row in joined.iterrows():
        print(f"  {row['key']}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write shapes
    if output_path.exists():
        combined = gpd.GeoDataFrame(pd.concat([existing, joined], ignore_index=True),
                                     crs=joined.crs)
        combined.to_file(output_path, driver='GPKG')
        print(f"\nAppended {len(joined)} shapes to {output_path} (total: {len(combined)})")
    else:
        joined.to_file(output_path, driver='GPKG')
        print(f"\nWrote {len(joined)} shapes to {output_path}")



if __name__ == '__main__':
    main()
